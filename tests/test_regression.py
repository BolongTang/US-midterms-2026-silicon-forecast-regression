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
    table = reg.build_training_table(ces)
    target = reg.build_target_table(ces)
    fsets = {"ces": reg.CES_FEATURES + ["midterm"]}
    return table, target, fsets


def test_training_table_has_no_nulls(real):
    table, _, fsets = real
    assert table.select(fsets["ces"] + reg.CLASSES).null_count().sum_horizontal()[0] == 0
    assert set(table["year"].unique()) == {2022, 2024}


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
    m = reg.evaluate(table, fsets, reg.train_sets([2022, 2024]))
    assert m.height == 6  # 3 leave-one-state-out + 2 cross-cycle + 1 leave-one-cycle-out
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
    sim = importlib.util.spec_from_file_location("sim", ROOT / "scripts" / "00b-simulate_regression_inputs.py")
    mod = importlib.util.module_from_spec(sim)
    sim.loader.exec_module(mod)
    mod.simulate(tmp_path / "data" / "00-simulated_data")
    monkeypatch.setattr(reg, "ROOT", tmp_path)
    metrics, preds = reg.run("simulated")
    assert set(metrics["features"]) == {"ces", "ces+acs"}
    assert {"midterms", "all"} <= set(metrics["train"])
    # Known data-generating process: the model should recover the Democratic share
    assert (metrics["mae_dem"] < 0.5 * metrics["baseline_mae_dem"]).all()
    assert (metrics["official_winner_acc"] > 0.85).all()
    assert len(preds["st"].unique()) == len(reg.SENATE_STATES_2026)
    assert (preds["p_dem"] + preds["p_rep"] + preds["p_und"]).to_numpy() == pytest.approx(1.0)
