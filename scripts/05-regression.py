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
#   - data/01-raw_data/ces_2012_2020.parquet from 01-download_data.py
#   - data/01-raw_data/president_results.csv from 01d-download_president.py (baseline)
#   - data/01-raw_data/acs_state.parquet    from 01b-download_acs.py
#   - data/01-raw_data/senate_results.csv   certified results, columns:
#       year, st, dem_votes, rep_votes, other_votes
#
# Design notes:
#   - Senate seats are decided by statewide popular vote, so vote share is a
#     valid target (no Electoral College).
#   - Targets come from CES pre-election Senate intent, the only source with an
#     "undecided" answer. Certified results check the Democratic share of the
#     two-party vote and are NEVER used to fit a model.
#   - 2026 features carry the 2024 CES (and ACS) forward: no 2026 survey exists.
#
# Leakage audit (see paper/regression.qmd for the numbers):
#   1. Timing: a forecast can only use information from before the election. The
#      primary design therefore LAGS every feature by two years (cycle y is
#      predicted from the CES and ACS of cycle y - 2), exactly the situation in
#      2026. "contemporaneous" tables (features from the same year as the target,
#      same respondents) are kept only to measure how much that leaked.
#   2. Same-cycle information: leave-one-state-out lets the other states' results
#      from the SAME election (the national swing) inform the held-out state. It is
#      kept, labelled optimistic. The honest test is forward-chaining: train on
#      earlier cycles only, predict the next one.
#   3. Time travel removed: "leave-one-cycle-out" and "predict other years" let
#      later cycles train a model that scores earlier ones; both were dropped.
#   4. Baselines (mean shares) are computed inside each training fold only.
#   5. ACS 5-year estimates for year Y are published after the year-Y election;
#      lagging them by two years fixes that too.
#   6. The 2026 forecast uses models trained on lagged features, matching the
#      two-year-old features it is applied to.
#
# Feature sets: ces (survey profile), ces+acs, prior (previous presidential and Senate
#   results only), ces+prior. All lagged; priors come from strictly earlier years.
#
# Baselines (simple, no fitting): party-ID majority (pid3, and pid7 with leaners),
#   last Senate result, last presidential result, majority class, mean share.
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
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="Sortedness")  # join_asof with `by`; inputs are sorted
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
# Previous results as features: the state's last presidential and last Senate two-party
# Democratic share, each from a year strictly before the target election. Where a state
# had no earlier Senate race, the Senate share falls back to the presidential one and
# has_prev_senate marks it.
PRIOR_FEATURES = ["prev_president", "prev_senate_f", "has_prev_senate"]
TARGET_YEAR = 2026
FEATURE_YEAR = 2024  # CES / ACS year carried forward to 2026
LAG = TARGET_YEAR - FEATURE_YEAR  # features lag the target by two years
TIMINGS = {"lagged": LAG, "contemporaneous": 0}  # contemporaneous = leaky, for comparison
ALPHAS = np.logspace(-2, 3, 20)
LOSO = "leave-one-state-out (same-cycle info)"
FORWARD = "forward-chaining"
D7 = ["Strong Democrat", "Not Very Strong Democrat", "Lean Democrat"]
R7 = ["Strong Republican", "Not Very Strong Republican", "Lean Republican"]

# Same list as scripts/02-build_personas.py
SENATE_STATES_2026 = [
    "AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS", "KY", "LA",
    "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM", "NC", "OH", "OK",
    "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
]  # fmt: skip


def paths(source: str) -> dict:
    if source == "simulated":
        d = ROOT / "data" / "00-simulated_data"
        return {"ces": [d / "simulated_ces.parquet"], "acs": d / "simulated_acs.parquet",
                "results": d / "simulated_senate_results.csv",
                "president": d / "simulated_president_results.csv", "out": d}  # fmt: skip
    raw = ROOT / "data" / "01-raw_data"
    return {"ces": [raw / "ces_2012_2020.parquet", raw / "ces_2022_2024.parquet"],
            "acs": raw / "acs_state.parquet", "results": raw / "senate_results.csv",
            "president": raw / "president_results.csv",
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
    aggs = dict(
        ces_male=wmean((pl.col("gender") == "Male").cast(pl.Float64)),
        ces_white=wmean((pl.col("race") == "White").cast(pl.Float64)),
        ces_black=wmean((pl.col("race") == "Black").cast(pl.Float64)),
        ces_hispanic=wmean((pl.col("race") == "Hispanic").cast(pl.Float64)),
        ces_college=wmean(pl.col("educ").is_in(["4-Year", "Post-Grad"]).cast(pl.Float64)),
        ces_age=wmean(pl.col("age").cast(pl.Float64)),
        ces_rep=wmean((pl.col("pid3") == "Republican").cast(pl.Float64)),
        ces_dem=wmean((pl.col("pid3") == "Democrat").cast(pl.Float64)),
        n_respondents=pl.len(),
    )
    if "pid7" in ces.columns:  # party ID with leaners, used only by a baseline
        aggs["ces_dem7"] = wmean(pl.col("pid7").is_in(D7).cast(pl.Float64))
        aggs["ces_rep7"] = wmean(pl.col("pid7").is_in(R7).cast(pl.Float64))
    return (
        ces.filter(pl.col("year") == year, pl.col("weight").is_not_null())
        .group_by("st").agg(**aggs).with_columns(year=pl.lit(year))
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


def load_results(path: Path) -> pl.DataFrame | None:
    """Certified returns -> Democratic share of the two-party vote (year, st, dem2p)."""
    if not path.exists():
        return None
    df = pl.read_csv(path)
    needed = {"year", "st", "dem_votes", "rep_votes", "other_votes"}
    if missing := needed - set(df.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df.select(
        pl.col("year").cast(pl.Int64), "st",
        dem2p=pl.col("dem_votes") / (pl.col("dem_votes") + pl.col("rep_votes")),
    )  # fmt: skip


def load_official(path: Path) -> pl.DataFrame | None:
    out = load_results(path)
    return None if out is None else out.rename({"dem2p": "official_dem2p"})


def add_acs(table: pl.DataFrame, acs: pl.DataFrame | None, acs_year_for) -> pl.DataFrame:
    if acs is None:
        return table
    acs = acs.rename({"acs_year": "_acs_year"})
    return (
        table.with_columns(_acs_year=acs_year_for(pl.col("year")))
        .join(acs, on=["st", "_acs_year"], how="left")
        .drop("_acs_year")
    )


def add_prior(table: pl.DataFrame, results: pl.DataFrame | None, name: str) -> pl.DataFrame:
    """Attach each state's most recent result from a year STRICTLY BEFORE the row's year."""
    if results is None:
        return table.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    r = results.select("st", pl.col("year").alias(f"{name}_year"), pl.col("dem2p").alias(name)).sort(f"{name}_year")
    left = table.with_columns(_y=(pl.col("year") - 1).cast(pl.Int64)).sort("_y")
    with warnings.catch_warnings():  # polars cannot verify sortedness with `by`; both sides are sorted
        warnings.simplefilter("ignore")
        out = left.join_asof(r, left_on="_y", right_on=f"{name}_year", by="st", strategy="backward")
    return out.drop("_y", f"{name}_year")


def add_prior_features(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_columns(
        prev_senate_f=pl.coalesce("prev_senate", "prev_president"),
        has_prev_senate=pl.col("prev_senate").is_not_null().cast(pl.Float64),
    )


def build_training_table(ces, acs=None, official=None, president=None, lag: int = LAG) -> pl.DataFrame:
    """One row per state-race. Features come from the cycle `lag` years before the
    target, so lag=2 mimics forecasting 2026 from 2024 and lag=0 is the leaky,
    same-year design. Cycles whose feature year is not in the data are dropped."""
    have = set(ces["year"].unique().to_list())
    years = sorted(ces.filter(pl.col("intent_sen").is_not_null())["year"].unique().to_list())
    parts = []
    for y in years:
        if y - lag not in have:
            continue
        feats = ces_features(ces, y - lag).with_columns(feature_year=pl.lit(y - lag), year=pl.lit(y))
        parts.append(feats.join(ces_targets(ces, y), on=["st", "year"]))
    table = pl.concat(parts, how="diagonal_relaxed").with_columns(pl.col("year").cast(pl.Int64))
    table = add_acs(table, acs, lambda y: y - lag)
    if official is not None:
        table = table.join(official, on=["st", "year"], how="left")
    table = add_prior(table, load_prior(official, "official_dem2p"), "prev_senate")
    table = add_prior_features(add_prior(table, president, "prev_president"))
    return table.with_columns(midterm=(pl.col("year") % 4 == 2).cast(pl.Float64))


def load_prior(official: pl.DataFrame | None, col: str) -> pl.DataFrame | None:
    return None if official is None else official.rename({col: "dem2p"})


def build_target_table(ces, acs=None, official=None, president=None) -> pl.DataFrame:
    table = ces_features(ces, FEATURE_YEAR).filter(pl.col("st").is_in(SENATE_STATES_2026))
    table = table.with_columns(feature_year=pl.lit(FEATURE_YEAR), year=pl.lit(TARGET_YEAR).cast(pl.Int64))
    table = add_acs(table, acs, lambda y: pl.lit(FEATURE_YEAR))
    table = add_prior(table, load_prior(official, "official_dem2p"), "prev_senate")
    table = add_prior_features(add_prior(table, president, "prev_president"))
    return table.with_columns(midterm=pl.lit(1.0))


#### Model ####
def to_alr(shares: np.ndarray) -> np.ndarray:
    """Additive log-ratio with 'und' as reference; keeps predictions on the simplex."""
    s = np.clip(shares, 0.005, None)
    return np.log(s[:, :2] / s[:, [2]])


def from_alr(z: np.ndarray) -> np.ndarray:
    e = np.exp(np.column_stack([z, np.zeros(len(z))]))
    return e / e.sum(axis=1, keepdims=True)


def fit_predict(train: pl.DataFrame, test: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Ridge regression on log-ratios, penalty chosen by leave-one-out CV on `train` only."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(train.select(features).to_numpy(), to_alr(train.select(CLASSES).to_numpy()))
    return from_alr(model.predict(test.select(features).to_numpy()))


def mean_shares(train: pl.DataFrame, n: int) -> np.ndarray:
    return np.tile(train.select(CLASSES).mean().to_numpy().ravel(), (n, 1))


def leave_one_state_out(table: pl.DataFrame, features: list[str]):
    """Predict each row from a model that never saw that state (in any cycle), with
    the mean-share baseline computed on the same training fold. OPTIMISTIC: the other
    states of the same cycle are in the training data."""
    pred, base = np.zeros((table.height, 3)), np.zeros((table.height, 3))
    states = table["st"].to_numpy()
    for s in np.unique(states):
        held = pl.Series(states == s)
        train, test = table.filter(~held), table.filter(held)
        pred[held.to_numpy()] = fit_predict(train, test, features)
        base[held.to_numpy()] = mean_shares(train, test.height)
    return pred, base


def forward_chaining(table: pl.DataFrame, features: list[str], years: list[int], min_states: int = 10):
    """Train only on EARLIER cycles of `years`, predict the next one. A single-cycle
    set predicts every later cycle in the table. Returns (test rows, predictions,
    fold baselines) pooled over test cycles, or None."""
    years = sorted(years)
    if len(years) == 1:
        tests = [(y, years) for y in sorted(table["year"].unique().to_list()) if y > years[0]]
    else:
        tests = [(y, [t for t in years if t < y]) for y in years[1:]]
    held, preds, bases = [], [], []
    for y, train_years in tests:
        train = table.filter(pl.col("year").is_in(train_years))
        test = table.filter(pl.col("year") == y)
        if train["st"].n_unique() < min_states or test.height == 0:
            continue
        held.append(test)
        preds.append(fit_predict(train, test, features))
        bases.append(mean_shares(train, test.height))
    if not held:
        return None
    return pl.concat(held), np.vstack(preds), np.vstack(bases)


#### Metrics ####
def score(test: pl.DataFrame, pred: np.ndarray, baseline: np.ndarray) -> dict:
    """MAE in percentage points, winner accuracy, and the same checks against
    certified results if the test rows have them. `baseline` is (n, 3) or (3,)."""
    true = test.select(CLASSES).to_numpy()
    base = np.broadcast_to(baseline, true.shape)
    out = {
        "n": test.height,
        "mae_dem": 100 * np.abs(pred[:, 0] - true[:, 0]).mean(),
        "mae_rep": 100 * np.abs(pred[:, 1] - true[:, 1]).mean(),
        "mae_und": 100 * np.abs(pred[:, 2] - true[:, 2]).mean(),
        "baseline_mae_dem": 100 * np.abs(base[:, 0] - true[:, 0]).mean(),
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


def evaluate(tables: dict, feature_sets: dict[str, list[str]], sets: dict) -> pl.DataFrame:
    """`tables` maps a timing name ('lagged', 'contemporaneous') to a training table."""
    rows = []
    for timing, table in tables.items():
        for fs_name, features in feature_sets.items():
            complete = table.drop_nulls(features)  # e.g. ACS missing for a year
            for label, years in sets.items():
                train = complete.filter(pl.col("year").is_in(years))
                if train["st"].n_unique() < 10:
                    continue
                base = {"timing": timing, "train": label, "features": fs_name}
                pred, bl = leave_one_state_out(train, features)
                rows.append({**base, "scheme": LOSO, **score(train, pred, bl)})
                if (fc := forward_chaining(complete, features, years)) is not None:
                    rows.append({**base, "scheme": FORWARD, **score(*fc)})
    return pl.DataFrame(rows)


def forecast(table: pl.DataFrame, target: pl.DataFrame, feature_sets, sets: dict) -> pl.DataFrame:
    """2026 forecast from models trained on the LAGGED table, as the target features are."""
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


#### Simple baselines ####
# Ideas considered, in order of how little they assume:
#   1. party-ID majority (pid3): the party with more self-identifiers in the state wins
#   2. party-ID majority with leaners (pid7): same, but leaners count for their party
#   3. last Senate result: the state's most recent certified Senate winner (persistence)
#   4. last presidential result: the state's most recent presidential winner
#   5. majority class: every state goes to the party that won most earlier races
#   6. mean share: every state gets the average two-party share of earlier races
# All use only information from before the target election.
BASELINES = [
    "party ID (pid3)", "party ID + leaners (pid7)", "last Senate result",
    "last presidential result", "majority class", "mean share",
]


def baseline_dem2p(name: str, test: pl.DataFrame, history: pl.DataFrame) -> np.ndarray:
    """Predicted Democratic two-party share (NaN where the baseline is undefined).
    `history` holds the earlier cycles' rows, with their certified shares."""
    def ratio(d, r):
        return (test[d] / (test[d] + test[r])).to_numpy().astype(float) if d in test.columns else np.full(test.height, np.nan)
    hist = history["official_dem2p"].drop_nulls().to_numpy() if "official_dem2p" in history.columns else np.array([])
    if name == "party ID (pid3)":
        return ratio("ces_dem", "ces_rep")
    if name == "party ID + leaners (pid7)":
        return ratio("ces_dem7", "ces_rep7")
    if name == "last Senate result":
        return test["prev_senate"].to_numpy().astype(float)
    if name == "last presidential result":
        return test["prev_president"].to_numpy().astype(float)
    if name == "majority class":  # winner only; 0.75 / 0.25 are placeholders, not shares
        return np.full(test.height, np.nan if hist.size == 0 else (0.75 if (hist > 0.5).mean() > 0.5 else 0.25))
    if name == "mean share":
        return np.full(test.height, np.nan if hist.size == 0 else hist.mean())
    raise ValueError(name)


def compare_by_year(tables: dict, feature_sets: dict[str, list[str]]) -> pl.DataFrame:
    """Row-level predictions for each test cycle: the ridge model trained on ALL earlier
    cycles and every baseline. One row per method, state and test year."""
    rows = []
    for timing, table in tables.items():
        years = sorted(table["year"].unique().to_list())
        for y in years[1:]:
            train_all, test = table.filter(pl.col("year") < y), table.filter(pl.col("year") == y)
            off = test["official_dem2p"].to_numpy().astype(float) if "official_dem2p" in test.columns else np.full(test.height, np.nan)
            ces_dem2p = (test["dem"] / (test["dem"] + test["rep"])).to_numpy()
            sts = test["st"].to_list()

            def record(method, dem2p, has_share=True):
                rows.extend(
                    {"timing": timing, "method": method, "test_year": y, "st": st, "pred_dem2p": None if np.isnan(d) else float(d),
                     "has_share": has_share, "official_dem2p": None if np.isnan(o) else float(o), "ces_dem2p": float(c)}
                    for st, d, o, c in zip(sts, dem2p, off, ces_dem2p)
                )  # fmt: skip

            for fs_name, features in feature_sets.items():
                comp_train, comp_test = train_all.drop_nulls(features), test.drop_nulls(features)
                if comp_train["st"].n_unique() >= 10 and comp_test.height:
                    p = fit_predict(comp_train, comp_test, features)
                    full = np.full(test.height, np.nan)
                    full[np.isin(test["st"].to_numpy(), comp_test["st"].to_numpy())] = p[:, 0] / (p[:, 0] + p[:, 1])
                    record(f"ridge, all earlier cycles ({fs_name})", full)
            for name in BASELINES:
                record(f"baseline: {name}", baseline_dem2p(name, test, train_all), has_share=name != "majority class")
    return pl.DataFrame(rows, schema_overrides={"pred_dem2p": pl.Float64, "official_dem2p": pl.Float64})


def summarise_comparison(cmp: pl.DataFrame, timing: str = "lagged", by_year: bool = False, common: bool = True) -> pl.DataFrame:
    """Winner accuracy and error per method. With common=True every method is scored on
    exactly the same state-years (those where ALL methods make a prediction and a
    certified result exists), so a baseline that is undefined for some states cannot
    look better or worse just because it saw different rows."""
    d = cmp.filter(pl.col("timing") == timing, pl.col("pred_dem2p").is_not_null(), pl.col("official_dem2p").is_not_null())
    if common:
        keys = d.group_by("st", "test_year").agg(k=pl.col("method").n_unique()).filter(pl.col("k") == d["method"].n_unique())
        d = d.join(keys.select("st", "test_year"), on=["st", "test_year"])
    group = ["method", "test_year"] if by_year else ["method"]
    return (
        d.group_by(group).agg(
            n=pl.len(),
            winner_acc=((pl.col("pred_dem2p") > 0.5) == (pl.col("official_dem2p") > 0.5)).mean(),
            mae_dem2p=pl.when(pl.col("has_share")).then(100 * (pl.col("pred_dem2p") - pl.col("official_dem2p")).abs()).mean(),
        ).sort(*group)
    )  # fmt: skip


def forecast_baselines(target: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for name in BASELINES:
        d = baseline_dem2p(name, target, table)
        parts.append(
            target.select("st").with_columns(
                baseline=pl.lit(name), dem2p=pl.Series(d),
                winner=pl.when(pl.Series(d).is_null() | pl.Series(d).is_nan()).then(pl.lit(None, dtype=pl.String))
                         .when(pl.Series(d) > 0.5).then(pl.lit("Democratic")).otherwise(pl.lit("Republican")),
            )
        )  # fmt: skip
    return pl.concat(parts).with_columns(pl.col("dem2p").fill_nan(None)).sort("st", "baseline")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location("_s", Path(__file__).parent / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(source: str) -> dict:
    p = paths(source)
    if source == "simulated" and not p["ces"][0].exists():
        load_script("00b-simulate_regression_inputs.py").simulate(p["out"])
    p["out"].mkdir(parents=True, exist_ok=True)
    ces = pl.concat([pl.read_parquet(f) for f in p["ces"] if f.exists()], how="diagonal_relaxed")
    acs = load_acs(p["acs"])
    official = load_official(p["results"])
    president = load_results(p["president"])
    print(f"Source: {source} | CES years: {sorted(ces['year'].unique().to_list())}")
    print(f"ACS features: {'yes' if acs is not None else 'NO (run 01b-download_acs.py)'}")
    print(f"Certified Senate results: {'yes' if official is not None else 'NO (senate_results.csv missing)'}")
    print(f"Presidential results: {'yes' if president is not None else 'NO (run 01d-download_president.py)'}")

    feature_sets = {"ces": CES_FEATURES + ["midterm"]}
    if acs is not None:
        feature_sets["ces+acs"] = CES_FEATURES + ACS_FEATURES + ["midterm"]
    if president is not None:  # previous results as features (no survey / with survey)
        feature_sets["prior"] = PRIOR_FEATURES + ["midterm"]
        feature_sets["ces+prior"] = CES_FEATURES + PRIOR_FEATURES + ["midterm"]

    lagged = build_training_table(ces, acs, official, president, lag=LAG)
    years = sorted(lagged["year"].unique().to_list())
    tables = {"lagged": lagged}
    tables["contemporaneous"] = build_training_table(ces, acs, official, president, lag=0).filter(pl.col("year").is_in(years))
    target = build_target_table(ces, acs, official, president)
    sets = train_sets(years)

    metrics = evaluate(tables, feature_sets, sets)
    preds = forecast(lagged, target, feature_sets, sets)
    comparison = compare_by_year(tables, feature_sets)
    base_forecast = forecast_baselines(target, lagged)
    for name, df in [("regression_metrics", metrics), ("regression_forecast_2026", preds),
                     ("baseline_comparison", comparison), ("baseline_forecast_2026", base_forecast)]:  # fmt: skip
        df.write_parquet(p["out"] / f"{name}.parquet")
    return {"metrics": metrics, "preds": preds, "comparison": comparison,
            "baseline_forecast": base_forecast, "tables": tables, "sets": sets,
            "feature_sets": feature_sets, "paths": p}  # fmt: skip


#### Run ####
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "simulated"], default=None)
    args = ap.parse_args()
    source = args.source or ("real" if paths("real")["ces"][-1].exists() else "simulated")
    out = run(source)
    with pl.Config(tbl_rows=80, tbl_cols=20, fmt_float="mixed", fmt_str_lengths=40):
        m = out["metrics"].filter(pl.col("timing") == "lagged", pl.col("train") == "all", pl.col("scheme") == FORWARD)
        print(m.select("features", "train", "scheme", "n", pl.col("official_mae_dem2p").round(1), pl.col("official_winner_acc").round(2)))
        c = summarise_comparison(out["comparison"]).sort("winner_acc", descending=True)
        print(c)
