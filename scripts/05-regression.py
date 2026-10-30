#### Preamble ####
# Purpose: Regress state-level Senate vote shares (Democratic / Republican /
#          Undecided-or-other) on state demographics and partisanship, test the
#          fit, and forecast the 2026 Senate races. Three training sets are
#          compared: 2022 only, 2024 only, and 2022 + 2024.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run 01-download_data.py. Optional inputs that add features and
#   an independent accuracy check (the script runs without them):
#   - data/01-raw_data/ces_2014_2020.parquet from 01-download_data.py
#   - data/01-raw_data/acs_state.parquet    from 01b-download_acs.py
#   - data/01-raw_data/senate_results.csv   certified results, columns:
#       year, st, dem_votes, rep_votes, other_votes
#
# Design notes:
#   - Senate seats are decided by statewide popular vote, so vote share is a
#     valid target (no Electoral College).
#   - Targets come from CES pre-election Senate intent, the only source with an
#     "undecided" answer. Certified results (when present) check the
#     Democratic share of the two-party vote.
#   - The 2026 features carry the 2024 CES (and 2024 ACS) forward: no 2026
#     survey exists.
#
# Years: every even year found in the data is used (2014-2024 when the extra
#   CES extract and ACS years are downloaded). Training sets: every single cycle,
#   2022+2024 (the requested pair), 2014+2016+2018, "midterms" and "all".
#
# Run with: uv run scripts/05-regression.py            (real data if present,
#                                                       else simulated)
#           uv run scripts/05-regression.py --source simulated

#### Workspace setup ####
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")

CLASSES = ["dem", "rep", "und"]  # und = Not Sure, No One, third party
CES_FEATURES = [
    "ces_male", "ces_white", "ces_black", "ces_hispanic", "ces_college",
    "ces_age", "ces_rep", "ces_dem",
]  # fmt: skip
ACS_FEATURES = [
    "acs_male", "acs_white", "acs_black", "acs_hispanic", "acs_college",
    "acs_median_age", "acs_log_income",
]  # fmt: skip
TARGET_YEAR = 2026
FEATURE_YEAR = 2024  # CES / ACS year carried forward to 2026
ALPHAS = np.logspace(-2, 3, 20)

# Same list as scripts/02-build_personas.py
SENATE_STATES_2026 = [
    "AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS", "KY", "LA",
    "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM", "NC", "OH", "OK",
    "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
]  # fmt: skip


def paths(source: str) -> dict[str, Path]:
    if source == "simulated":
        d = ROOT / "data" / "00-simulated_data"
        return {"ces": [d / "simulated_ces.parquet"], "acs": d / "simulated_acs.parquet",
                "results": d / "simulated_senate_results.csv", "out": d}  # fmt: skip
    raw = ROOT / "data" / "01-raw_data"
    return {"ces": [raw / "ces_2014_2020.parquet", raw / "ces_2022_2024.parquet"],
            "acs": raw / "acs_state.parquet", "results": raw / "senate_results.csv",
            "out": ROOT / "data" / "02-analysis_data"}  # fmt: skip


def train_sets(years: list[int]) -> dict[str, list[int]]:
    """Every single cycle, then the combinations of interest, where the data allow."""
    sets = {str(y): [y] for y in years}
    combos = {
        "2014+2016+2018": [2014, 2016, 2018],
        "2022+2024": [2022, 2024],
        "midterms": [y for y in years if y % 4 == 2],
        "all": list(years),
    }
    for label, ys in combos.items():
        ys = [y for y in ys if y in years]
        if len(ys) > 1 and ys not in sets.values() and (label != "2014+2016+2018" or len(ys) == 3):
            sets[label] = ys
    return sets


#### Build the state table ####
def wmean(expr: pl.Expr) -> pl.Expr:
    """Survey-weighted mean of a 0/1 or numeric expression."""
    return (expr * pl.col("weight")).sum() / pl.col("weight").sum()


def ces_features(ces: pl.DataFrame, year: int) -> pl.DataFrame:
    """Weighted state profile from all CES respondents of one year."""
    return (
        ces.filter(pl.col("year") == year, pl.col("weight").is_not_null())
        .group_by("st")
        .agg(
            ces_male=wmean((pl.col("gender") == "Male").cast(pl.Float64)),
            ces_white=wmean((pl.col("race") == "White").cast(pl.Float64)),
            ces_black=wmean((pl.col("race") == "Black").cast(pl.Float64)),
            ces_hispanic=wmean((pl.col("race") == "Hispanic").cast(pl.Float64)),
            ces_college=wmean(
                pl.col("educ").is_in(["4-Year", "Post-Grad"]).cast(pl.Float64)
            ),
            ces_age=wmean(pl.col("age").cast(pl.Float64)),
            ces_rep=wmean((pl.col("pid3") == "Republican").cast(pl.Float64)),
            ces_dem=wmean((pl.col("pid3") == "Democrat").cast(pl.Float64)),
            n_respondents=pl.len(),
        )
        .with_columns(year=pl.lit(year))
    )


def ces_targets(ces: pl.DataFrame, year: int) -> pl.DataFrame:
    """Weighted Dem / Rep / Undecided-or-other shares of Senate intent."""
    answered = ces.filter(
        pl.col("year") == year,
        pl.col("intent_sen").is_not_null(),
        pl.col("weight").is_not_null(),
    )
    is_dem = pl.col("intent_sen").str.starts_with("[Democrat")
    is_rep = pl.col("intent_sen").str.starts_with("[Republican")
    return answered.group_by("st").agg(
        dem=wmean(is_dem.cast(pl.Float64)),
        rep=wmean(is_rep.cast(pl.Float64)),
        und=wmean((~is_dem & ~is_rep).cast(pl.Float64)),
        n_intent=pl.len(),
    ).with_columns(year=pl.lit(year))  # fmt: skip


def load_acs(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def load_official(path: Path) -> pl.DataFrame | None:
    """Certified results -> Democratic share of the two-party vote."""
    if not path.exists():
        return None
    df = pl.read_csv(path)
    needed = {"year", "st", "dem_votes", "rep_votes", "other_votes"}
    if missing := needed - set(df.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df.select(
        "year",
        "st",
        official_dem2p=pl.col("dem_votes") / (pl.col("dem_votes") + pl.col("rep_votes")),
    )


def add_acs(table: pl.DataFrame, acs: pl.DataFrame | None, acs_year_for) -> pl.DataFrame:
    if acs is None:
        return table
    acs = acs.rename({"acs_year": "_acs_year"})
    return (
        table.with_columns(_acs_year=acs_year_for(pl.col("year")))
        .join(acs, on=["st", "_acs_year"], how="left")
        .drop("_acs_year")
    )


def build_training_table(ces, acs=None, official=None) -> pl.DataFrame:
    # Only cycles where CES asked about the Senate race
    years = sorted(ces.filter(pl.col("intent_sen").is_not_null())["year"].unique().to_list())
    table = pl.concat(
        [ces_features(ces, y).join(ces_targets(ces, y), on=["st", "year"]) for y in years]
    )
    table = add_acs(table, acs, lambda y: y)
    if official is not None:
        table = table.join(official, on=["st", "year"], how="left")
    return table.with_columns(midterm=(pl.col("year") % 4 == 2).cast(pl.Float64))


def build_target_table(ces, acs=None) -> pl.DataFrame:
    table = ces_features(ces, FEATURE_YEAR).filter(pl.col("st").is_in(SENATE_STATES_2026))
    table = add_acs(table, acs, lambda y: pl.lit(FEATURE_YEAR))
    return table.with_columns(year=pl.lit(TARGET_YEAR), midterm=pl.lit(1.0))


#### Model ####
def to_alr(shares: np.ndarray) -> np.ndarray:
    """Additive log-ratio with 'und' as reference; keeps predictions on the simplex."""
    s = np.clip(shares, 0.005, None)
    return np.log(s[:, :2] / s[:, [2]])


def from_alr(z: np.ndarray) -> np.ndarray:
    e = np.exp(np.column_stack([z, np.zeros(len(z))]))
    return e / e.sum(axis=1, keepdims=True)


def fit_predict(train: pl.DataFrame, test: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Ridge regression on log-ratios, penalty chosen by leave-one-out CV."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(train.select(features).to_numpy(), to_alr(train.select(CLASSES).to_numpy()))
    return from_alr(model.predict(test.select(features).to_numpy()))


def leave_one_state_out(table: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Predict each row from a model that never saw that state (in any cycle)."""
    pred = np.zeros((table.height, 3))
    states = table["st"].to_numpy()
    for s in np.unique(states):
        held = states == s
        pred[held] = fit_predict(table.filter(~pl.Series(held)), table.filter(pl.Series(held)), features)
    return pred


#### Metrics ####
def score(test: pl.DataFrame, pred: np.ndarray, baseline: np.ndarray) -> dict:
    """MAE in percentage points, winner accuracy, and the same checks against
    certified results if the test rows have them."""
    true = test.select(CLASSES).to_numpy()
    out = {
        "n": test.height,
        "mae_dem": 100 * np.abs(pred[:, 0] - true[:, 0]).mean(),
        "mae_rep": 100 * np.abs(pred[:, 1] - true[:, 1]).mean(),
        "mae_und": 100 * np.abs(pred[:, 2] - true[:, 2]).mean(),
        "baseline_mae_dem": 100 * np.abs(baseline[0] - true[:, 0]).mean(),
        "winner_acc_ces": float(((pred[:, 0] > pred[:, 1]) == (true[:, 0] > true[:, 1])).mean()),
        "official_mae_dem2p": None,
        "official_winner_acc": None,
    }
    if "official_dem2p" in test.columns:
        off = test["official_dem2p"].to_numpy().astype(float)
        ok = ~np.isnan(off)
        if ok.any():
            two_party = pred[:, 0] / (pred[:, 0] + pred[:, 1])
            out["official_mae_dem2p"] = 100 * np.abs(two_party[ok] - off[ok]).mean()
            out["official_winner_acc"] = float(((two_party[ok] > 0.5) == (off[ok] > 0.5)).mean())
    return out


def evaluate(table: pl.DataFrame, feature_sets: dict[str, list[str]], sets: dict) -> pl.DataFrame:
    rows = []
    for fs_name, features in feature_sets.items():
        complete = table.drop_nulls(features)  # e.g. ACS missing for a year
        for label, years in sets.items():
            train = complete.filter(pl.col("year").is_in(years))
            if train["st"].n_unique() < 10:
                continue
            mean_shares = train.select(CLASSES).mean().to_numpy().ravel()
            base = {"train": label, "features": fs_name}
            # Leave-one-state-out within the training cycles
            pred = leave_one_state_out(train, features)
            rows.append({**base, "scheme": "leave-one-state-out",
                         **score(train, pred, mean_shares)})  # fmt: skip
            if len(years) == 1:
                # Out-of-cycle: a single-cycle model predicts every other cycle
                test = complete.filter(~pl.col("year").is_in(years))
                if test.height:
                    pred = fit_predict(train, test, features)
                    rows.append({**base, "scheme": "predict other years",
                                 **score(test, pred, mean_shares)})  # fmt: skip
            else:
                # Leave-one-cycle-out: hold out each cycle in turn
                held, preds = [], []
                for y in years:
                    sub = train.filter(pl.col("year") != y)
                    test = train.filter(pl.col("year") == y)
                    held.append(test)
                    preds.append(fit_predict(sub, test, features))
                rows.append({**base, "scheme": "leave-one-cycle-out",
                             **score(pl.concat(held), np.vstack(preds), mean_shares)})  # fmt: skip
    return pl.DataFrame(rows)


def forecast(table: pl.DataFrame, target: pl.DataFrame, feature_sets, sets: dict) -> pl.DataFrame:
    parts = []
    for fs_name, features in feature_sets.items():
        tgt = target.drop_nulls(features)
        complete = table.drop_nulls(features)
        for label, years in sets.items():
            train = complete.filter(pl.col("year").is_in(years))
            if train["st"].n_unique() < 10:
                continue
            pred = fit_predict(train, tgt, features)
            parts.append(
                tgt.select("st").with_columns(
                    train=pl.lit(label),
                    features=pl.lit(fs_name),
                    p_dem=pl.Series(pred[:, 0]),
                    p_rep=pl.Series(pred[:, 1]),
                    p_und=pl.Series(pred[:, 2]),
                )
            )
    return (
        pl.concat(parts)
        .with_columns(
            winner=pl.when(pl.col("p_dem") > pl.col("p_rep")).then(pl.lit("Democratic")).otherwise(pl.lit("Republican")),
            margin_dem=pl.col("p_dem") - pl.col("p_rep"),
        )
        .sort("st", "features", "train")
    )  # fmt: skip


def load_script(name: str):
    spec = importlib.util.spec_from_file_location("_s", Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    p = paths(source)
    if source == "simulated" and not p["ces"][0].exists():
        load_script("00b-simulate_regression_inputs.py").simulate(p["out"])
    p["out"].mkdir(parents=True, exist_ok=True)
    ces = pl.concat([pl.read_parquet(f) for f in p["ces"] if f.exists()], how="diagonal_relaxed")
    acs = load_acs(p["acs"])
    official = load_official(p["results"])
    print(f"Source: {source} | years: {sorted(ces['year'].unique().to_list())}")
    print(f"ACS features: {'yes' if acs is not None else 'NO (run 01b-download_acs.py)'}")
    print(f"Certified results: {'yes' if official is not None else 'NO (senate_results.csv missing)'}")

    feature_sets = {"ces": CES_FEATURES + ["midterm"]}
    if acs is not None:
        feature_sets["ces+acs"] = CES_FEATURES + ACS_FEATURES + ["midterm"]

    table = build_training_table(ces, acs, official)
    target = build_target_table(ces, acs)
    sets = train_sets(sorted(table["year"].unique().to_list()))

    metrics = evaluate(table, feature_sets, sets)
    preds = forecast(table, target, feature_sets, sets)
    metrics.write_parquet(p["out"] / "regression_metrics.parquet")
    preds.write_parquet(p["out"] / "regression_forecast_2026.parquet")
    return metrics, preds


#### Run ####
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "simulated"], default=None)
    args = ap.parse_args()
    source = args.source or (
        "real" if paths("real")["ces"][-1].exists() else "simulated"
    )
    metrics, preds = run(source)
    with pl.Config(tbl_rows=60, tbl_cols=20, fmt_float="mixed"):
        print(metrics.with_columns(pl.col(pl.Float64).round(2)))
        last = preds["features"].unique().sort()[-1]
        wide = preds.filter(pl.col("features") == last).pivot(
            on="train", index="st", values="p_dem"
        ).sort("st")
        print(f"\nPredicted Democratic share ({last} features), by training set:")
        print(wide.with_columns(pl.col(pl.Float64).round(3)))
