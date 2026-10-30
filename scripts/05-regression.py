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
# Run with: uv run scripts/05-regression.py

#### Workspace setup ####
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("data/01-raw_data")
OUT_DIR = Path("data/02-analysis_data")

CLASSES = ["dem", "rep", "und"]  # und = Not Sure, No One, third party
CES_FEATURES = [
    "ces_male", "ces_white", "ces_black", "ces_hispanic", "ces_college",
    "ces_age", "ces_rep", "ces_dem",
]  # fmt: skip
ACS_FEATURES = [
    "acs_male", "acs_white", "acs_black", "acs_hispanic", "acs_college",
    "acs_median_age", "acs_log_income",
]  # fmt: skip
TRAIN_SETS = {"2022": [2022], "2024": [2024], "2022+2024": [2022, 2024]}
TARGET_YEAR = 2026
FEATURE_YEAR = 2024  # CES / ACS year carried forward to 2026
ALPHAS = np.logspace(-2, 3, 20)

# Same list as scripts/02-build_personas.py
SENATE_STATES_2026 = [
    "AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS", "KY", "LA",
    "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM", "NC", "OH", "OK",
    "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
]  # fmt: skip


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
    years = [2022, 2024]
    table = pl.concat(
        [ces_features(ces, y).join(ces_targets(ces, y), on=["st", "year"]) for y in years]
    )
    table = add_acs(table, acs, lambda y: y)
    if official is not None:
        table = table.join(official, on=["st", "year"], how="left")
    return table.with_columns(midterm=(pl.col("year") == 2022).cast(pl.Float64))


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


def evaluate(table: pl.DataFrame, feature_sets: dict[str, list[str]]) -> pl.DataFrame:
    rows = []
    for fs_name, features in feature_sets.items():
        for label, years in TRAIN_SETS.items():
            train = table.filter(pl.col("year").is_in(years))
            mean_shares = train.select(CLASSES).mean().to_numpy().ravel()
            # Leave-one-state-out within the training cycles
            pred = leave_one_state_out(train, features)
            rows.append({"train": label, "features": fs_name, "scheme": "leave-one-state-out",
                         **score(train, pred, mean_shares)})  # fmt: skip
            # Out-of-cycle: a single-cycle model predicts the other cycle
            if len(years) == 1:
                test = table.filter(~pl.col("year").is_in(years))
                pred = fit_predict(train, test, features)
                rows.append({"train": label, "features": fs_name,
                             "scheme": f"predict {test['year'][0]}",
                             **score(test, pred, mean_shares)})  # fmt: skip
    return pl.DataFrame(rows)


def forecast(table: pl.DataFrame, target: pl.DataFrame, feature_sets) -> pl.DataFrame:
    parts = []
    for fs_name, features in feature_sets.items():
        for label, years in TRAIN_SETS.items():
            pred = fit_predict(table.filter(pl.col("year").is_in(years)), target, features)
            parts.append(
                target.select("st").with_columns(
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


#### Run ####
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ces = pl.read_parquet(RAW_DIR / "ces_2022_2024.parquet")
    acs = load_acs(RAW_DIR / "acs_state.parquet")
    official = load_official(RAW_DIR / "senate_results.csv")
    print(f"ACS features: {'yes' if acs is not None else 'NO (run 01b-download_acs.py)'}")
    print(f"Certified results: {'yes' if official is not None else 'NO (senate_results.csv missing)'}")

    feature_sets = {"ces": CES_FEATURES + ["midterm"]}
    if acs is not None:
        feature_sets["ces+acs"] = CES_FEATURES + ACS_FEATURES + ["midterm"]

    table = build_training_table(ces, acs, official)
    target = build_target_table(ces, acs)

    metrics = evaluate(table, feature_sets)
    preds = forecast(table, target, feature_sets)
    metrics.write_parquet(OUT_DIR / "regression_metrics.parquet")
    preds.write_parquet(OUT_DIR / "regression_forecast_2026.parquet")

    with pl.Config(tbl_rows=50, tbl_cols=20, fmt_float="mixed"):
        print(metrics.with_columns(pl.col(pl.Float64).round(2)))
        wide = preds.filter(pl.col("features") == list(feature_sets)[-1]).pivot(
            on="train", index="st", values="p_dem"
        ).sort("st")
        print("\nPredicted Democratic share, by training set:")
        print(wide.with_columns(pl.col(pl.Float64).round(3)))
