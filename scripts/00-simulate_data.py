#### Preamble ####
# Purpose: Simulate the data we expect to end up with for the silicon-sampling
#          Senate forecast: (1) CES-like respondents who become personas,
#          (2) LLM vote responses for those personas, (3) the post-stratification
#          frame, and (4) the forecast distribution over outcomes.
# Author: Yi Zhi Zhang
# Date: 28 Oct 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: None
#
# The simulation deliberately builds in two failure modes documented in the
# literature, so that our tests and analysis can catch them on real data:
#   - Mode collapse / overconfidence (Bisbee et al. 2024; Heath and Alexander
#     2026): LLM vote probabilities are pushed towards 0 or 1.
#   - Systematic bias (Argyle et al. 2023): each model leans slightly toward one
#     party, and pure independents are predicted poorly.

#### Workspace setup ####
from pathlib import Path

import numpy as np
import polars as pl

rng = np.random.default_rng(seed=853)
OUT_DIR = Path("data/00-simulated_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_STATE = 120  # personas per state, similar to what we will sample from CES
N_REPEATS = 5  # times each persona is sent to each model
N_DRAWS = 1000  # simulated elections in the forecast

# The 35 states with a Senate race in 2026 (33 regular + FL and OH specials)
STATES = [
    "AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS", "KY",
    "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM", "NC",
    "OH", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
]  # fmt: skip

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AR": "Arkansas", "CO": "Colorado",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "ID": "Idaho",
    "IL": "Illinois", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MT": "Montana", "NE": "Nebraska",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NC": "North Carolina", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "VA": "Virginia", "WV": "West Virginia",
    "WY": "Wyoming",
}  # fmt: skip

# Two hypothetical models with different failure modes.
# overconfidence > 1 pushes probabilities toward 0/1 (mode collapse);
# bias < 0 shifts every persona toward the Democrat (on the logit scale).
MODELS = {
    "provider-a/model-small": {"overconfidence": 2.0, "bias": -0.3},
    "provider-b/model-large": {"overconfidence": 1.5, "bias": 0.1},
}


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


#### Simulate state leans ####
# Republican lean of each state on the logit scale. Positive = more Republican.
state_lean = dict(zip(STATES, rng.normal(loc=0.3, scale=0.8, size=len(STATES))))

#### Simulate respondents (future personas) ####
n = N_PER_STATE * len(STATES)
state = np.repeat(STATES, N_PER_STATE)
lean = np.array([state_lean[s] for s in state])

age = np.clip(rng.normal(50, 17, n).round(), 18, 90).astype(int)
gender = rng.choice(["man", "woman"], n, p=[0.48, 0.52])
race = rng.choice(
    ["white", "Black", "Hispanic", "Asian", "other"],
    n,
    p=[0.65, 0.12, 0.14, 0.05, 0.04],
)
education = rng.choice(
    [
        "high school or less",
        "some college",
        "a bachelor's degree",
        "a postgraduate degree",
    ],
    n,
    p=[0.35, 0.30, 0.22, 0.13],
)

# Party ID depends on demographics and the state's lean (the interactions the
# rubric asks for). Latent Republican-ness, then cut into three groups.
race_effect = {
    "white": 0.4,
    "Black": -1.8,
    "Hispanic": -0.5,
    "Asian": -0.6,
    "other": 0.0,
}
educ_effect = {
    "high school or less": 0.4,
    "some college": 0.2,
    "a bachelor's degree": -0.2,
    "a postgraduate degree": -0.6,
}
latent_r = (
    lean
    + np.array([race_effect[r] for r in race])
    + np.array([educ_effect[e] for e in education])
    + 0.01 * (age - 50)
    + np.where(gender == "man", 0.3, -0.3)
    + rng.normal(0, 1.2, n)
)
party = np.select(
    [latent_r < -0.6, latent_r > 0.6], ["Democrat", "Republican"], default="independent"
)

# Ideology follows party, with noise
ideology_scale = np.clip((latent_r + rng.normal(0, 0.8, n)).round(), -2, 2).astype(int)
ideology_labels = {
    -2: "very liberal",
    -1: "liberal",
    0: "moderate",
    1: "conservative",
    2: "very conservative",
}
ideology = np.array([ideology_labels[i] for i in ideology_scale])

# True probability of voting Republican: party dominates, ideology and state
# lean matter, and independents are close to a coin flip.
party_effect = {"Democrat": -3.0, "independent": 0.0, "Republican": 3.0}
true_logit_r = (
    np.array([party_effect[p] for p in party]) + 0.6 * ideology_scale + 0.5 * lean
)
true_p_r = logistic(true_logit_r)
true_vote = np.where(rng.random(n) < true_p_r, "Republican", "Democrat")

# Midterm turnout propensity: older and more educated people vote more.
turnout_p = logistic(
    -0.5
    + 0.03 * (age - 50)
    + np.array([{"high school or less": -0.5, "some college": 0.0,
                 "a bachelor's degree": 0.4, "a postgraduate degree": 0.6}[e]
                for e in education])
)  # fmt: skip


def make_backstory(row: dict) -> str:
    """First-person backstory in the style of Argyle et al. (2023)."""
    party_text = (
        "an independent" if row["party"] == "independent" else f"a {row['party']}"
    )
    return (
        f"I am a {row['age']}-year-old {row['race']} {row['gender']} living in "
        f"{STATE_NAMES[row['state']]}. My highest level of education is {row['education']}. "
        f"Politically, I consider myself {party_text} and {row['ideology']}."
    )


respondents = pl.DataFrame(
    {
        "persona_id": np.arange(n),
        "state": state,
        "age": age,
        "gender": gender,
        "race": race,
        "education": education,
        "party": party,
        "ideology": ideology,
        "true_p_republican": true_p_r,
        "true_vote": true_vote,
        "turnout_propensity": turnout_p,
    }
)
respondents = respondents.with_columns(
    pl.struct(pl.all())
    .map_elements(make_backstory, return_dtype=pl.String)
    .alias("backstory")
)

#### Simulate LLM responses ####
# Each persona goes to each model N_REPEATS times. The model's probability is a
# distorted version of the truth: stretched (overconfident) and shifted (bias).
# Independents get extra noise, mirroring Argyle et al.'s weak fit for them.
response_frames = []
for model, failure in MODELS.items():
    extra_noise = np.where(party == "independent", rng.normal(0, 1.0, n), 0.0)
    llm_p_r = logistic(
        failure["overconfidence"] * true_logit_r + failure["bias"] + extra_noise
    )
    for repeat in range(N_REPEATS):
        draw = rng.random(n)
        # ~2% of answers are unusable (refusals or bad formatting)
        answer = np.where(draw < llm_p_r, "Republican", "Democrat")
        unusable = rng.random(n) < 0.02
        response_frames.append(
            pl.DataFrame(
                {
                    "persona_id": np.arange(n),
                    "model": model,
                    "repeat": repeat,
                    "answer": answer,
                    "unusable": unusable,
                }
            )
            .with_columns(
                answer=pl.when(pl.col("unusable"))
                .then(None)
                .otherwise(pl.col("answer"))
            )
            .drop("unusable")
        )
responses = pl.concat(response_frames)

#### Simulate post-stratification frame ####
# Each persona's weight is their turnout propensity, normalised within state,
# so weights sum to 1 in every state.
frame = respondents.select("persona_id", "state", "turnout_propensity").with_columns(
    weight=pl.col("turnout_propensity")
    / pl.col("turnout_propensity").sum().over("state")
)

#### Simulate forecast ####
# 1. Share of valid answers that are Republican, per persona (pooling models).
persona_share = (
    responses.drop_nulls("answer")
    .group_by("persona_id")
    .agg(p_republican=(pl.col("answer") == "Republican").mean())
)
# 2. Weighted average within state = point estimate of Republican two-party share.
state_estimate = (
    frame.join(persona_share, on="persona_id")
    .group_by("state")
    .agg(
        rep_share=(pl.col("weight") * pl.col("p_republican")).sum()
        / pl.col("weight").sum()
    )
    .sort("state")
)
# 3. Simulate elections: a national swing shared by all states plus state noise.
national_error = rng.normal(0, 0.03, N_DRAWS)
draws = []
for row in state_estimate.iter_rows(named=True):
    state_error = rng.normal(0, 0.02, N_DRAWS)
    share = np.clip(row["rep_share"] + national_error + state_error, 0, 1)
    draws.append(
        pl.DataFrame(
            {
                "draw": np.arange(N_DRAWS),
                "state": row["state"],
                "rep_share": share,
                "dem_share": 1 - share,
            }
        )
    )
forecast = pl.concat(draws)

#### Save data ####
respondents.write_parquet(OUT_DIR / "simulated_personas.parquet")
responses.write_parquet(OUT_DIR / "simulated_responses.parquet")
frame.write_parquet(OUT_DIR / "simulated_frame.parquet")
forecast.write_parquet(OUT_DIR / "simulated_forecast.parquet")

summary = (
    forecast.group_by("state")
    .agg(
        p_rep_win=(pl.col("rep_share") > 0.5).mean(),
        median_rep=pl.col("rep_share").median(),
    )
    .sort("state")
)
print(summary.head(10))
