#### Preamble ####
# Purpose: Test the regression pipeline: helpers on small synthetic data, the
#          certified-results loader, and the real-data forecast outputs.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: data/01-raw_data/ces_2022_2024.parquet (the sample is not enough)

#### Workspace setup ####
import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("reg", ROOT / "scripts" / "05-regression.py")
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)

CES_PATH = ROOT / "data" / "01-raw_data" / "ces_2022_2024.parquet"


def toy_ces() -> pl.DataFrame:
    """Two states, one year; weights differ so weighting matters."""
    return pl.DataFrame(
        {
            "year": [2024] * 6,
            "st": ["AA", "AA", "AA", "BB", "BB", "BB"],
            "weight": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            "intent_sen": ["[Democrat / Candidate 1]", "[Republican / Candidate 2]",
                           "Not Sure", "[Democrat / Candidate 1]",
                           "[Democrat / Candidate 1]", "No One"],
        }
    )  # fmt: skip


#### Helpers ####
def test_alr_roundtrip_and_simplex():
    shares = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    back = reg.from_alr(reg.to_alr(shares))
    assert np.allclose(back, shares, atol=1e-9)
    assert np.allclose(back.sum(axis=1), 1.0)


def test_targets_are_weighted_and_sum_to_one():
    t = reg.ces_targets(toy_ces(), 2024).sort("st")
    aa = t.filter(pl.col("st") == "AA").row(0, named=True)
    assert aa["dem"] == pytest.approx(0.25)
    assert aa["rep"] == pytest.approx(0.25)
    assert aa["und"] == pytest.approx(0.5)  # weight 2 of 4
    assert (t["dem"] + t["rep"] + t["und"]).to_numpy() == pytest.approx(1.0)


def test_fit_predict_recovers_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 60)
    dem = 0.2 + 0.5 * x
    df = pl.DataFrame({"x": x, "dem": dem, "rep": 0.6 - 0.5 * x, "und": np.full(60, 0.2)})
    pred = reg.fit_predict(df[:50], df[50:], ["x"])
    assert np.abs(pred[:, 0] - df[50:]["dem"].to_numpy()).max() < 0.03
    assert pred.sum(axis=1) == pytest.approx(1.0)


#### Certified results ####
def test_load_official_missing_file_is_none(tmp_path):
    assert reg.load_official(tmp_path / "nope.csv") is None


def test_load_official_rejects_bad_columns(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("year,st,dem_votes\n2024,AA,10\n")
    with pytest.raises(ValueError, match="missing columns"):
        reg.load_official(p)


def test_load_official_two_party_share(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("year,st,dem_votes,rep_votes,other_votes\n2024,AA,60,40,5\n")
    assert reg.load_official(p)["official_dem2p"][0] == pytest.approx(0.6)


def test_score_uses_official_results_when_present():
    test = pl.DataFrame({"dem": [0.5, 0.3], "rep": [0.3, 0.5], "und": [0.2, 0.2],
                         "official_dem2p": [0.6, None]})  # fmt: skip
    pred = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2]])
    s = reg.score(test, pred, np.array([0.4, 0.4, 0.2]))
    assert s["official_mae_dem2p"] == pytest.approx(100 * abs(0.5 / 0.8 - 0.6))
    assert s["official_winner_acc"] == 1.0
    assert s["winner_acc_ces"] == 1.0


#### Real data ####
@pytest.fixture(scope="module")
def real():
    if not CES_PATH.exists():
        pytest.skip("full CES extract not present")
    ces = pl.read_parquet(CES_PATH)
    table = reg.build_training_table(ces, lag=0)  # the sample file has no pre-2022 cycles
    target = reg.build_target_table(ces)
    fsets = {"ces": reg.CES_FEATURES + ["midterm"]}
    return table, target, fsets


def test_training_table_has_no_nulls(real):
    table, _, fsets = real
    assert table.select(fsets["ces"] + reg.CLASSES).null_count().sum_horizontal()[0] == 0
    assert set(table["year"].unique()) == {2022, 2024}  # feature_year equals year at lag 0


def test_forecast_covers_every_2026_race(real):
    table, target, fsets = real
    assert set(target["st"]) == set(reg.SENATE_STATES_2026)
    sets = reg.train_sets([2022, 2024])
    f = reg.forecast(table, target, fsets, sets)
    assert f.height == len(reg.SENATE_STATES_2026) * 3
    assert set(f["train"]) == {"2022", "2024", "2022+2024"}
    assert (f["p_dem"] + f["p_rep"] + f["p_und"]).to_numpy() == pytest.approx(1.0)
    assert f.select(pl.all().null_count()).sum_horizontal()[0] == 0


def test_model_beats_mean_baseline_out_of_sample(real):
    table, _, fsets = real
    m = reg.evaluate({"contemporaneous": table}, fsets, reg.train_sets([2022, 2024]))
    # 3 sets x leave-one-state-out + forward-chaining only where an earlier cycle exists
    assert set(m["scheme"]) == {reg.LOSO, reg.FORWARD}
    assert m.filter(pl.col("scheme") == reg.FORWARD)["train"].to_list() == ["2022", "2022+2024"]
    assert (m["mae_dem"] < m["baseline_mae_dem"]).all()
    assert (m["winner_acc_ces"] > 0.7).all()


#### Training sets and simulated end-to-end ####
def test_train_sets_grow_with_available_years():
    assert reg.train_sets([2022, 2024]) == {"2022": [2022], "2024": [2024], "2022+2024": [2022, 2024]}
    wide = reg.train_sets([2014, 2016, 2018, 2020, 2022, 2024])
    assert wide["2014+2016+2018"] == [2014, 2016, 2018]
    assert wide["midterms"] == [2014, 2018, 2022]
    assert wide["all"] == [2014, 2016, 2018, 2020, 2022, 2024]
    assert {"2014", "2016", "2018", "2020", "2022", "2024", "2022+2024"} <= set(wide)


def test_simulated_pipeline_runs_and_beats_baseline(tmp_path, monkeypatch):
    out = run_simulated(tmp_path, monkeypatch)
    metrics, preds = out["metrics"], out["preds"]
    assert set(metrics["features"]) == {"ces", "ces+acs", "prior", "ces+prior"}
    assert set(metrics["timing"]) == {"lagged", "contemporaneous"}
    assert {"midterms", "all", "2014", "2014+2016+2018"} <= set(metrics["train"])
    # Known data-generating process: the model should recover the Democratic share
    assert (metrics["mae_dem"] < 0.5 * metrics["baseline_mae_dem"]).all()
    assert (metrics["official_winner_acc"] > 0.8).all()
    assert preds["st"].n_unique() == len(reg.SENATE_STATES_2026)
    assert (preds["p_dem"] + preds["p_rep"] + preds["p_und"]).to_numpy() == pytest.approx(1.0)


def run_simulated(tmp_path, monkeypatch):
    sim = importlib.util.spec_from_file_location("sim", ROOT / "scripts" / "00b-simulate_regression_inputs.py")
    mod = importlib.util.module_from_spec(sim)
    sim.loader.exec_module(mod)
    mod.simulate(tmp_path / "data" / "00-simulated_data")
    monkeypatch.setattr(reg, "ROOT", tmp_path)
    return reg.run("simulated")


#### Leakage guards ####
def toy_panel() -> pl.DataFrame:
    """Two states, cycles 2016-2022; party ID share differs by year so any mix-up shows."""
    rows = []
    for y in (2014, 2016, 2018):
        for st, n_rep in (("AA", (y - 2012) // 2), ("BB", 5)):
            rows += [{"year": y, "st": st, "weight": 1.0, "gender": "Male", "race": "White", "educ": "4-Year",
                      "age": 40, "pid3": "Republican" if i < n_rep else "Democrat", "pid7": "Strong Democrat",
                      "intent_sen": "[Democrat / Candidate 1]" if y > 2014 else None} for i in range(10)]
    return pl.DataFrame(rows)  # fmt: skip


def test_lagged_features_come_from_two_years_earlier():
    ces = toy_panel()
    lagged = reg.build_training_table(ces, lag=2)
    assert (lagged["year"] - lagged["feature_year"] == 2).all()
    assert set(lagged["year"].unique()) == {2016, 2018}  # 2014 has no 2012 features
    # AA's Republican ID share is 0.1 in 2014, 0.2 in 2016, 0.3 in 2018
    aa = lagged.filter(pl.col("st") == "AA").sort("year")
    assert aa["ces_rep"].to_list() == pytest.approx([0.1, 0.2])  # 2016 row carries 2014's value
    same_year = reg.build_training_table(ces, lag=0).filter(pl.col("st") == "AA").sort("year")
    assert same_year["ces_rep"].to_list() == pytest.approx([0.2, 0.3])  # the leaky design


def test_prior_results_use_only_earlier_years():
    table = pl.DataFrame({"st": ["AA", "AA"], "year": [2018, 2020]})
    res = pl.DataFrame({"st": ["AA", "AA", "AA"], "year": [2016, 2018, 2020], "dem2p": [0.4, 0.5, 0.6]})
    out = reg.add_prior(table, res, "prev").sort("year")
    assert out["prev"].to_list() == [0.4, 0.5]  # a year never sees its own result


def test_forward_chaining_never_trains_on_later_cycles():
    rng = np.random.default_rng(1)
    rows = []
    for y in (2014, 2016, 2018, 2020):
        for i in range(20):
            x = rng.uniform()
            rows.append({"year": y, "st": f"S{i}", "x": x, "dem": 0.3 + 0.3 * x, "rep": 0.6 - 0.3 * x, "und": 0.1})
    table = pl.DataFrame(rows)
    test, pred, base = reg.forward_chaining(table, ["x"], [2014, 2016, 2018, 2020])
    assert sorted(set(test["year"].to_list())) == [2016, 2018, 2020]  # first cycle is never a test cycle
    single = reg.forward_chaining(table, ["x"], [2018])
    assert set(single[0]["year"].to_list()) == {2020}  # 2018-only model scores only the future


def test_fold_baseline_excludes_the_held_out_state():
    rows = [{"year": 2020, "st": f"S{i}", "x": 0.0, "dem": 0.9 if i == 0 else 0.1, "rep": 0.1, "und": 0.0} for i in range(12)]
    table = pl.DataFrame(rows)
    _, base = reg.leave_one_state_out(table, ["x"])
    # held-out S0 (dem 0.9) must not be in its own baseline: the mean of the other 11 is 0.1
    assert base[0, 0] == pytest.approx(0.1)


def test_baselines_are_defined_and_use_only_history(tmp_path, monkeypatch):
    out = run_simulated(tmp_path, monkeypatch)
    cmp = out["comparison"]
    assert {f"baseline: {b}" for b in reg.BASELINES} <= set(cmp["method"])
    summary = reg.summarise_comparison(cmp)
    # common rows: every method is scored on the same number of state-races
    assert summary["n"].n_unique() == 1
    assert summary.filter(pl.col("method") == "baseline: majority class")["mae_dem2p"].is_null().all()
    assert summary.filter(pl.col("method") == "baseline: mean share")["winner_acc"][0] < 0.9
    assert out["baseline_forecast"]["st"].n_unique() == len(reg.SENATE_STATES_2026)


def test_prior_features_fall_back_to_president_and_flag_it():
    table = pl.DataFrame({"prev_senate": [0.6, None], "prev_president": [0.5, 0.4]})
    out = reg.add_prior_features(table)
    assert out["prev_senate_f"].to_list() == [0.6, 0.4]  # missing Senate share -> presidential share
    assert out["has_prev_senate"].to_list() == [1.0, 0.0]


def test_prior_feature_sets_exist_and_forecast_every_race(tmp_path, monkeypatch):
    out = run_simulated(tmp_path, monkeypatch)
    f = out["preds"].filter(pl.col("features").is_in(["prior", "ces+prior"]))
    assert f["features"].n_unique() == 2
    assert f.group_by("features", "train").agg(n=pl.col("st").n_unique())["n"].to_list() == [len(reg.SENATE_STATES_2026)] * (2 * len(out["sets"]))
    # the simulated state lean persists, so previous results should call winners well
    fwd = out["metrics"].filter(pl.col("scheme") == reg.FORWARD, pl.col("timing") == "lagged", pl.col("features") == "prior", pl.col("train") == "all")
    assert fwd["official_winner_acc"][0] > 0.85
    # targets never feed the priors: no prior feature equals the same row's certified share
    tbl = out["tables"]["lagged"]
    assert not (tbl["prev_president"] == tbl["official_dem2p"]).any()


def test_model_is_selected_on_validation_and_scored_on_untouched_holdout(tmp_path, monkeypatch):
    out = run_simulated(tmp_path, monkeypatch)
    sel = out["selection"]
    years = sorted(out["tables"]["lagged"]["year"].unique().to_list())
    holdout = str(years[-1])
    val_years = set(sel.filter(pl.col("phase") == "validation")["test_years"].str.split(",").explode())
    assert holdout not in val_years  # the holdout never enters validation
    assert set(sel.filter(pl.col("phase") == "holdout")["test_years"]) == {holdout}
    chosen = sel.filter(pl.col("selected"), pl.col("phase") == "validation")
    assert chosen.height == 1
    val = sel.filter(pl.col("phase") == "validation")
    assert chosen["official_mae_dem2p"][0] == val["official_mae_dem2p"].min()  # lowest validation error
    fs, window = out["selected"]
    assert fs in out["feature_sets"] and window in reg.WINDOWS
    assert out["headline_train"] in set(out["preds"]["train"])
    assert out["headline_train"] in out["sets"]
