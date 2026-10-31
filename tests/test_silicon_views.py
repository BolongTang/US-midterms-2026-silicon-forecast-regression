#### Preamble ####
# Purpose: Test the silicon-sampling part of the website: the stored LLM results are read
#          correctly, every race is covered, favoured candidates follow the win
#          probabilities, and the page is one document with silicon sampling before regression.
# Author: Yi Zhi Zhang
# Date: 31 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: the LLM parquet files in data/02-analysis_data (tests skip without them)

#### Workspace setup ####
import importlib.util
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "02-analysis_data"
NEEDED = ["forecast_2026_summary", "forecast_2026_seats", "estimates_2026_by_model", "backtest_2024_states",
          "backtest_2024_individual", "backtest_2024_by_party", "calibration_2024", "responses_2024",
          "responses_2024_forced", "responses_2026", "responses_2026_deepseek", "responses_2026_forced"]  # fmt: skip
pytestmark = pytest.mark.skipif(not all((DATA / f"{n}.parquet").exists() for n in NEEDED), reason="LLM result files absent")

spec = importlib.util.spec_from_file_location("si", ROOT / "scripts" / "08-silicon_views.py")
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


@pytest.fixture(scope="module")
def sd():
    return si.load_silicon(ROOT)


#### Stored results ####
def test_every_2026_race_has_a_forecast_and_valid_probabilities(sd):
    s = sd["summary"]
    assert s.height == 35 and s["state"].n_unique() == 35
    assert s["p_rep_win"].is_between(0, 1).all()
    assert ((s["lower_80"] <= s["median"]) & (s["median"] <= s["upper_80"])).all()
    assert set(si.MODEL_ORDER) <= set(s.columns)  # one raw estimate column per model


def test_seats_are_valid_and_control_matches_the_threshold(sd):
    c = si.control_stats(sd["seats"])
    assert sd["seats"].height == 10_000 and 0 <= c["p_rep_control"] <= 1
    assert c["p_rep_control"] == pytest.approx(float((sd["seats"]["rep_seats"] >= si.MAJORITY_REP).mean()))
    assert sd["seats"]["rep_seats"].min() >= si.SEATS_NOT_UP_REP


def test_all_responses_have_valid_models_and_the_deepseek_rerun_replaces_the_failed_run(sd):
    free = si.free_2026(sd["resp"])
    assert set(free["model"]) == set(si.MODEL_ORDER)
    ds = free.filter(pl.col("model") == "DeepSeek V4 Flash")
    assert ds["answer"].is_not_null().mean() > 0.99  # the failed first run (33% valid) is not used
    first = sd["resp"]["responses_2026"].filter(pl.col("model") == "deepseek/deepseek-v4-flash")
    assert first["answer"].is_not_null().mean() < 0.5  # ... and it really was the failed one


#### Cards and map ####
def test_favoured_candidate_follows_the_win_probability_and_names_are_found(sd):
    races = si.silicon_races(sd["summary"], sd["races"], sd["cands"])
    assert len(races) == 35
    p = dict(zip(sd["summary"]["state"], sd["summary"]["p_rep_win"]))
    for r in races:
        assert (r["winner_party"] == "Republican") == (p[r["st"]] >= 0.5)
        assert r["rep"] is not None and r["winner_name"]
        assert r["dem2p"] == pytest.approx(1 - sd["summary"].filter(pl.col("state") == r["st"])["median"][0])
    indep = {r["st"] for r in races if r["bar_left"] == "I"}
    assert indep == {"ID", "MT", "NE", "SD"}  # independent main opponents


def test_cards_and_map_render_for_the_silicon_forecast(sd):
    races = si.silicon_races(sd["summary"], sd["races"], sd["cands"])
    html = si.views.cards_html(races)
    assert html.count('class="race-card"') == 35 and "P(Republican wins)" in html and "grid-template-columns:repeat(3" in html
    assert html.count("person win") == 35  # one larger profile per race
    if (ROOT / "data" / "01-raw_data" / "us-states.geojson").exists():
        fig = si.views.winner_map(races, title="t", legend=("a", "b", "c"))
        assert {t.get_text() for t in fig.axes[0].texts} >= {"Paxton", "Hageman"} or len(fig.axes[0].texts) > 30


def test_every_silicon_figure_and_table_builds(sd, tmp_path):
    plots = [si.fig_seats(sd["seats"]), si.fig_intervals(sd["summary"]), si.fig_models26(sd["summary"]),
             si.fig_backtest(sd["bt"], sd["calib"]), si.fig_agreement(sd["ind"]), si.fig_by_party(sd["by_party"]),
             si.fig_answers(sd["resp"]), si.fig_composition(sd["personas"])]  # fmt: skip
    for i, p in enumerate(plots):
        p.save(tmp_path / f"f{i}.png", dpi=50, verbose=False)
        assert (tmp_path / f"f{i}.png").stat().st_size > 3000
    costs = si.cost_table(sd["resp"])
    assert costs["cost_usd_total"].iloc[-1] == pytest.approx(sum(df["cost_usd"].sum() for df in sd["resp"].values()))
    assert len(si.crosstab(sd["resp"], sd["personas"], "pid7")) == 8
    assert len(si.backtest_summary(sd["bt"])) == 2


def test_silicon_and_regression_forecasts_can_be_compared(sd):
    if not (DATA / "model_selection.parquet").exists():
        pytest.skip("regression outputs absent")
    c = si.compare_with_regression(sd["summary"], ROOT)
    assert len(c) == 35 and c["reg_dem2p"].between(0, 1).all()


#### One document: silicon sampling first, regression second ####
def test_single_page_orders_silicon_before_regression():
    text = (ROOT / "paper" / "index.qmd").read_text()
    assert text.index("{#silicon-sampling}") < text.index("{#regression-forecast}")
    assert "{{< include _regression.qmd >}}" in text
    assert not (ROOT / "paper" / "regression.qmd").exists()  # no second page
    assert (ROOT / "paper" / "_regression.qmd").exists()
    cfg = (ROOT / "paper" / "_quarto.yml").read_text()
    assert "- index.qmd" in cfg and "regression.qmd" not in cfg and "navbar" not in cfg
    body, fence, h1 = (ROOT / "paper" / "_regression.qmd").read_text(), False, []
    for line in body.splitlines():
        fence ^= line.startswith("```")
        if not fence and line.startswith("# "):
            h1.append(line)
    assert h1 == []  # the regression part's headings sit one level down, under "Regression forecast"
