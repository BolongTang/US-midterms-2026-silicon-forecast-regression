#### Preamble ####
# Purpose: Test the simulated personas, LLM responses, frame, and forecast.
#          The same checks will be adapted for the real data.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run scripts/00-simulate_data.py
#
# Run with: uv run pytest tests/test_simulated_data.py

#### Workspace setup ####
from pathlib import Path

import pandera.polars as pa
import polars as pl
import pytest

DATA = Path("data/00-simulated_data")
N_STATES = 35
VALID_ANSWERS = ["Republican", "Democrat"]


@pytest.fixture(scope="module")
def personas():
    return pl.read_parquet(DATA / "simulated_personas.parquet")


@pytest.fixture(scope="module")
def responses():
    return pl.read_parquet(DATA / "simulated_responses.parquet")


@pytest.fixture(scope="module")
def frame():
    return pl.read_parquet(DATA / "simulated_frame.parquet")


@pytest.fixture(scope="module")
def forecast():
    return pl.read_parquet(DATA / "simulated_forecast.parquet")


#### Personas ####
persona_schema = pa.DataFrameSchema(
    {
        "persona_id": pa.Column(int, unique=True),
        "state": pa.Column(str, pa.Check.str_length(2, 2)),
        "age": pa.Column(int, pa.Check.in_range(18, 90)),
        "gender": pa.Column(str, pa.Check.isin(["man", "woman"])),
        "race": pa.Column(
            str, pa.Check.isin(["white", "Black", "Hispanic", "Asian", "other"])
        ),
        "party": pa.Column(
            str, pa.Check.isin(["Democrat", "independent", "Republican"])
        ),
        "true_p_republican": pa.Column(float, pa.Check.in_range(0, 1)),
        "turnout_propensity": pa.Column(float, pa.Check.in_range(0, 1)),
        "backstory": pa.Column(str, pa.Check.str_length(min_value=50)),
    },
    strict=False,  # other columns allowed
)


def test_personas_schema(personas):
    persona_schema.validate(personas)


def test_every_race_has_personas(personas):
    assert personas["state"].n_unique() == N_STATES


def test_backstory_has_no_missing_fields(personas):
    # A missing value would print as "None" inside the persona text
    assert not personas["backstory"].str.contains("None").any()


def test_party_varies_within_states(personas):
    # Every state should contain all three party groups
    assert (
        personas.group_by("state").agg(pl.col("party").n_unique())["party"] == 3
    ).all()


#### Responses ####
def test_answers_are_valid_or_missing(responses):
    valid = responses.drop_nulls("answer")["answer"]
    assert valid.is_in(VALID_ANSWERS).all()


def test_missing_answer_rate_is_low(responses):
    # An NA flowing into the forecast is the main risk the assignment names
    assert responses["answer"].null_count() / responses.height < 0.05


def test_every_persona_answered_by_every_model(personas, responses):
    counts = responses.group_by("persona_id").agg(pl.col("model").n_unique())
    assert counts.height == personas.height
    assert (counts["model"] == responses["model"].n_unique()).all()


def test_missingness_does_not_depend_on_party(personas, responses):
    # Bisbee et al. (2024) found formatting failures were not random across groups
    rates = (
        responses.join(personas.select("persona_id", "party"), on="persona_id")
        .group_by("party")
        .agg(na_rate=pl.col("answer").is_null().mean())
    )
    assert rates["na_rate"].max() - rates["na_rate"].min() < 0.02


#### Frame ####
def test_weights_sum_to_one_in_each_state(frame):
    sums = frame.group_by("state").agg(pl.col("weight").sum())
    assert sums["weight"].is_between(0.999, 1.001).all()


def test_weights_are_positive(frame):
    assert (frame["weight"] > 0).all()


#### Forecast ####
def test_forecast_has_every_state(forecast):
    assert forecast["state"].n_unique() == N_STATES


def test_shares_between_zero_and_one(forecast):
    assert forecast["rep_share"].is_between(0, 1).all()


def test_shares_sum_to_one(forecast):
    total = forecast["rep_share"] + forecast["dem_share"]
    assert (total - 1).abs().max() < 1e-9


def test_no_missing_values_in_forecast(forecast):
    assert forecast.null_count().sum_horizontal().item() == 0


def test_same_number_of_draws_per_state(forecast):
    assert forecast.group_by("state").len()["len"].n_unique() == 1


def test_state_errors_are_correlated(forecast):
    # Gelman et al. (2020): states should move together because of the shared
    # national swing. Average pairwise correlation across draws should be positive.
    wide = forecast.pivot(on="state", index="draw", values="rep_share").drop("draw")
    corr = wide.corr().to_numpy()
    mean_off_diagonal = (corr.sum() - N_STATES) / (N_STATES * (N_STATES - 1))
    assert mean_off_diagonal > 0.3
