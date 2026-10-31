#### Preamble ####
# Purpose: Turn the LLM answers into (1) a 2024 backtest against the real
#          results and (2) a 2026 Senate forecast whose bias correction and
#          uncertainty come from that backtest.
# Author: Yi Zhi Zhang
# Date: 31 Oct 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run 01, 02 and 03 (main 2024 and 2026 runs, the 2026 DeepSeek
#                 rerun, and the forced-choice runs for pure independents).
#
# Steps
#   1. Each persona's synthetic Republican share (p_hat) per model: Republican
#      answers / answers for the Republican or the race's main opponent.
#      Pure independents use their forced-choice answers.
#   2. Each persona's weight: CES survey weight x turnout. 2024 uses validated
#      2024 turnout; 2026 uses midterm turnout rates estimated from CES 2022.
#   3. State estimate = weighted mean of p_hat (post-stratification).
#   4. 2024: compare with real results -> per-model bias and state-level error.
#   5. 2026: correct each backtested model by its 2024 bias, then simulate
#      elections with shared national error, state error, model choice and
#      sampling error. Save the draws as parquet.
#
# Run with: uv run scripts/04-forecast.py

#### Workspace setup ####
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path("data/01-raw_data")
DATA = Path("data/02-analysis_data")
SEED = 853
N_DRAWS = 10_000
N_BOOT = 200  # bootstrap resamples of personas for sampling error

# Models with a 2024 backtest (knowledge cutoffs before Nov 2024). Only these
# feed the headline forecast, because only they have a measured bias.
BACKTESTED_MODELS = ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]
# Current model: 2026 only, reported for comparison (no backtest possible).
CURRENT_MODEL = "deepseek/deepseek-v4-flash"

# ASSUMPTION: one election gives only one draw of the national error, so its
# size cannot be estimated from our backtest. We set it here and say so in the
# paper; change it to see how sensitive the forecast is.
NATIONAL_SD = 0.03

# Seats NOT up in 2026 (119th Congress: 53 Republicans, 47 Democrats and
# independents who caucus with them; 22 Republican-held and 13 non-Republican
# seats are up). Check before publishing.
SEATS_NOT_UP_REP = 31
SEATS_NOT_UP_OTHER = 34
MAJORITY_REP = 50  # Republicans control with 50 seats (the Vice President breaks ties)

PARTY_SIDE = {
    "Strong Democrat": "D",
    "Not Very Strong Democrat": "D",
    "Lean Democrat": "D",
    "Strong Republican": "R",
    "Not Very Strong Republican": "R",
    "Lean Republican": "R",
}

rng = np.random.default_rng(SEED)


#### Loading ####
def load_races(year: int) -> pl.DataFrame:
    """State and the label a vote for the main opponent has in our answers."""
    return (
        pl.read_csv(RAW / "senate_races.csv")
        .filter(pl.col("year") == year)
        .select(
            "state",
            opponent=pl.when(pl.col("main_opponent") == "Democratic")
            .then(pl.lit("Democrat"))
            .otherwise(pl.lit("Independent")),
        )
    )


def load_responses(year: int) -> pl.DataFrame:
    """All answers for a year, labelled free or forced choice."""
    cols = ["persona_id", "state", "model", "repeat", "answer", "cost_usd"]
    free = pl.read_parquet(DATA / f"responses_{year}.parquet")
    if year == 2026:
        # DeepSeek's first run failed (reasoning used up the token limit);
        # replace it with the rerun made with reasoning turned off.
        free = free.filter(pl.col("model") != CURRENT_MODEL)
        rerun = DATA / "responses_2026_deepseek.parquet"
        if rerun.exists():
            free = pl.concat([free.select(cols), pl.read_parquet(rerun).select(cols)])
    parts = [free.select(cols).with_columns(source=pl.lit("free"))]
    forced = DATA / f"responses_{year}_forced.parquet"
    if forced.exists():
        parts.append(
            pl.read_parquet(forced).select(cols).with_columns(source=pl.lit("forced"))
        )
    else:
        print(
            f"Note: no forced-choice file for {year}; pure independents use free answers"
        )
    return pl.concat(parts)


#### Step 1: persona shares ####
def persona_shares(responses: pl.DataFrame, personas: pl.DataFrame, year: int):
    """p_hat per persona and model. Pure independents use forced choice."""
    races = load_races(year)
    counts = (
        responses.join(races, on="state")
        .group_by("persona_id", "model", "source")
        .agg(
            n_answers=pl.len(),
            n_rep=(pl.col("answer") == "Republican").sum(),
            n_opp=(pl.col("answer") == pl.col("opponent")).sum(),
        )
        .join(personas.select("persona_id", "pid7"), on="persona_id")
    )
    has_forced = counts.filter(pl.col("source") == "forced").height > 0
    use_forced = (pl.col("pid7") == "Independent") & pl.lit(has_forced)
    chosen = counts.filter(
        pl.when(use_forced)
        .then(pl.col("source") == "forced")
        .otherwise(pl.col("source") == "free")
    )
    return chosen.with_columns(
        n_two_party=pl.col("n_rep") + pl.col("n_opp"),
        p_hat=pl.when(pl.col("n_rep") + pl.col("n_opp") > 0).then(
            pl.col("n_rep") / (pl.col("n_rep") + pl.col("n_opp"))
        ),
    ).drop("pid7")


#### Step 2: weights ####
def turnout_cells(df: pl.DataFrame) -> pl.DataFrame:
    """Coarse cells for midterm turnout: age group x college x news interest."""
    return df.with_columns(
        age_group=pl.when(pl.col("age") < 30)
        .then(pl.lit("18-29"))
        .when(pl.col("age") < 45)
        .then(pl.lit("30-44"))
        .when(pl.col("age") < 65)
        .then(pl.lit("45-64"))
        .otherwise(pl.lit("65+")),
        college=pl.col("educ").is_in(["4-Year", "Post-Grad"]).fill_null(False),
        follows_news=(pl.col("newsint") == "Most of the time").fill_null(False),
    )


def weights_2024(personas: pl.DataFrame, ces: pl.DataFrame) -> pl.DataFrame:
    """CES weight x validated 2024 turnout (1 if voted, 0 if no record)."""
    voted = ces.filter(pl.col("year") == 2024).select(
        persona_id="case_id",
        voted=(pl.col("vv_turnout_gvm") == "Voted").cast(pl.Float64),
    )
    return personas.join(voted, on="persona_id", how="left").select(
        "persona_id", w=pl.col("weight").fill_null(0) * pl.col("voted").fill_null(0)
    )


def weights_2026(personas: pl.DataFrame, ces: pl.DataFrame) -> pl.DataFrame:
    """CES weight x midterm turnout rate of the persona's cell, from CES 2022."""
    ces22 = turnout_cells(ces.filter(pl.col("year") == 2022)).with_columns(
        voted=(pl.col("vv_turnout_gvm") == "Voted").cast(pl.Float64),
        weight=pl.col("weight").fill_null(0),
    )
    keys = ["age_group", "college", "follows_news"]
    rates = ces22.group_by(keys).agg(
        turnout=(pl.col("voted") * pl.col("weight")).sum() / pl.col("weight").sum()
    )
    overall = (ces22["voted"] * ces22["weight"]).sum() / ces22["weight"].sum()
    return (
        turnout_cells(personas)
        .join(rates, on=keys, how="left")
        .select(
            "persona_id",
            w=pl.col("weight").fill_null(0) * pl.col("turnout").fill_null(overall),
        )
    )


#### Step 3: state estimates ####
def state_estimates(
    shares: pl.DataFrame, weights: pl.DataFrame, personas: pl.DataFrame
):
    """Weighted mean of p_hat per state and model, with a bootstrap standard error."""
    d = (
        shares.join(weights, on="persona_id")
        .join(personas.select("persona_id", "state"), on="persona_id")
        .filter(pl.col("p_hat").is_not_null(), pl.col("w") > 0)
    )
    rows = []
    for (state, model), g in d.group_by("state", "model"):
        p, w = g["p_hat"].to_numpy(), g["w"].to_numpy()
        boot = [
            np.average(p[i], weights=w[i])
            for i in (rng.integers(0, len(p), len(p)) for _ in range(N_BOOT))
        ]
        rows.append(
            {
                "state": state,
                "model": model,
                "rep_share": float(np.average(p, weights=w)),
                "se": float(np.std(boot)),
                "n_personas": len(p),
            }
        )
    return pl.DataFrame(rows).sort("state", "model")


#### Step 4: 2024 backtest ####
def individual_agreement(shares, personas, year=2024):
    """Does the persona's synthetic vote match their real reported vote?"""
    races = load_races(year)
    real = (
        personas.join(races, on="state")
        .with_columns(
            actual=pl.when(pl.col("voted_sen_party") == "Republican")
            .then(pl.lit("R"))
            .when(
                pl.col("voted_sen_party")
                == pl.when(pl.col("opponent") == "Democrat")
                .then(pl.lit("Democratic"))
                .otherwise(pl.lit("Independent"))
            )
            .then(pl.lit("O"))
        )
        .filter(pl.col("actual").is_not_null())
        .select("persona_id", "pid7", "actual")
    )
    llm = shares.filter(pl.col("p_hat").is_not_null()).select(
        "persona_id",
        "model",
        predicted=pl.when(pl.col("p_hat") >= 0.5)
        .then(pl.lit("R"))
        .otherwise(pl.lit("O")),
    )
    by_person = llm.join(real, on="persona_id")
    # Party-line baseline: vote with your side; no prediction for pure independents
    baseline = real.with_columns(
        model=pl.lit("party-line baseline"),
        predicted=pl.col("pid7")
        .replace_strict(PARTY_SIDE, default=None)
        .replace({"D": "O"}),
    ).filter(pl.col("predicted").is_not_null())
    both = pl.concat(
        [by_person, baseline.select(by_person.columns)], how="vertical_relaxed"
    ).with_columns(correct=pl.col("predicted") == pl.col("actual"))
    overall = both.group_by("model").agg(agreement=pl.col("correct").mean(), n=pl.len())
    by_party = both.group_by("model", "pid7").agg(
        agreement=pl.col("correct").mean(), n=pl.len()
    )
    return overall.sort("model"), by_party.sort("model", "pid7")


def backtest_states(estimates: pl.DataFrame) -> pl.DataFrame:
    results = pl.read_csv(RAW / "senate_results_2024.csv").select(
        "state",
        actual=pl.col("republican_pct")
        / (pl.col("republican_pct") + pl.col("main_opponent_pct")),
    )
    return estimates.join(results, on="state").with_columns(
        error=pl.col("actual") - pl.col("rep_share"),
        winner_right=(pl.col("actual") > 0.5) == (pl.col("rep_share") > 0.5),
    )


#### Step 5: 2026 forecast ####
def simulate(estimates, bias, state_sd):
    """Draws of the Republican two-party share in each 2026 state."""
    est = estimates.filter(pl.col("model").is_in(BACKTESTED_MODELS)).join(
        bias, on="model"
    )
    est = est.with_columns(corrected=pl.col("rep_share") + pl.col("bias"))
    states = sorted(est["state"].unique())
    models = BACKTESTED_MODELS
    centre = np.array(
        [
            [est.filter(state=s, model=m)["corrected"].item() for s in states]
            for m in models
        ]
    )
    se = np.array(
        [[est.filter(state=s, model=m)["se"].item() for s in states] for m in models]
    )
    model_pick = rng.integers(0, len(models), N_DRAWS)  # model uncertainty
    national = rng.normal(0, NATIONAL_SD, N_DRAWS)  # shared by all states
    state_err = rng.normal(0, state_sd, (N_DRAWS, len(states)))
    sampling = rng.normal(0, 1, (N_DRAWS, len(states))) * se[model_pick]
    share = np.clip(centre[model_pick] + national[:, None] + state_err + sampling, 0, 1)
    return pl.DataFrame(
        {
            "draw": np.repeat(np.arange(N_DRAWS), len(states)),
            "state": np.tile(states, N_DRAWS),
            "model": np.repeat(np.array(models)[model_pick], len(states)),
            "rep_share": share.ravel(),
        }
    ).with_columns(opp_share=1 - pl.col("rep_share"))


#### Main ####
ces = pl.read_parquet(RAW / "ces_2022_2024.parquet")
personas = {y: pl.read_parquet(DATA / f"personas_{y}.parquet") for y in (2024, 2026)}

# --- 2024 backtest ---
shares_24 = persona_shares(load_responses(2024), personas[2024], 2024)
est_24 = state_estimates(shares_24, weights_2024(personas[2024], ces), personas[2024])
bt = backtest_states(est_24)
bias = bt.group_by("model").agg(bias=pl.col("error").mean())
residual_sd = float(
    bt.join(bias, on="model").select((pl.col("error") - pl.col("bias")).std()).item()
)
overall, by_party = individual_agreement(shares_24, personas[2024])

no_answer_24 = shares_24.filter(pl.col("p_hat").is_null()).height
print(f"2024: {no_answer_24} persona-model pairs had no two-party answer (dropped)")
print("\nIndividual agreement with real 2024 votes:")
print(overall)
print("\nState-level backtest by model:")
print(
    bt.group_by("model").agg(
        bias=pl.col("error").mean(),
        mean_abs_error=pl.col("error").abs().mean(),
        winners_right=pl.col("winner_right").sum(),
        races=pl.len(),
    )
)
print(f"State-level error SD after removing bias: {residual_sd:.3f}")

# --- 2026 forecast ---
shares_26 = persona_shares(load_responses(2026), personas[2026], 2026)
est_26 = state_estimates(shares_26, weights_2026(personas[2026], ces), personas[2026])
draws = simulate(est_26, bias, residual_sd)

summary = (
    draws.group_by("state")
    .agg(
        p_rep_win=(pl.col("rep_share") > 0.5).mean(),
        median=pl.col("rep_share").median(),
        lower_80=pl.col("rep_share").quantile(0.1),
        upper_80=pl.col("rep_share").quantile(0.9),
    )
    .join(
        est_26.pivot(on="model", index="state", values="rep_share"),
        on="state",
    )
    .sort("p_rep_win")
)
seats = (
    draws.group_by("draw")
    .agg(rep_seats=(pl.col("rep_share") > 0.5).sum() + SEATS_NOT_UP_REP)
    .sort("draw")
)
p_control = (seats["rep_seats"] >= MAJORITY_REP).mean()

print(
    f"\n2026: {shares_26.filter(pl.col('p_hat').is_null()).height} persona-model pairs dropped"
)
print(f"P(Republicans keep Senate control): {p_control:.0%}")
print(f"Median Republican seats: {seats['rep_seats'].median():.0f}")
print(summary.select("state", "p_rep_win", "median", "lower_80", "upper_80"))

#### Save ####
bt.write_parquet(DATA / "backtest_2024_states.parquet")
overall.write_parquet(DATA / "backtest_2024_individual.parquet")
by_party.write_parquet(DATA / "backtest_2024_by_party.parquet")
est_26.write_parquet(DATA / "estimates_2026_by_model.parquet")
draws.write_parquet(DATA / "forecast_2026_draws.parquet")
summary.write_parquet(DATA / "forecast_2026_summary.parquet")
seats.write_parquet(DATA / "forecast_2026_seats.parquet")
print("\nSaved backtest and forecast files to", DATA)
