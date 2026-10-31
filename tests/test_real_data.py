#### Preamble ####
# Purpose: Test the real data: the labelled CES 2022/2024 subset and the
#          personas built from it.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run scripts/01-download_data.py and scripts/02-build_personas.py
#
# Run with: uv run pytest tests/test_real_data.py

#### Workspace setup ####
from pathlib import Path

import pandera.polars as pa
import polars as pl
import pytest

RAW = Path("data/01-raw_data/ces_2022_2024.parquet")
PERSONAS = {
    2024: Path("data/02-analysis_data/personas_2024.parquet"),
    2026: Path("data/02-analysis_data/personas_2026.parquet"),
}

# Written out independently of the scripts, so the tests check the scripts
# against what we expect rather than against themselves.
SENATE_STATES = {
    2024: {
        "AZ", "CA", "CT", "DE", "FL", "HI", "IN", "ME", "MD", "MA", "MI",
        "MN", "MS", "MO", "MT", "NE", "NV", "NJ", "NM", "NY", "ND", "OH",
        "PA", "RI", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    },
    2026: {
        "AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS",
        "KY", "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ",
        "NM", "NC", "OH", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA",
        "WV", "WY",
    },
}  # fmt: skip
RACES = ["White", "Black", "Hispanic", "Asian", "Native American", "Mixed",
         "Other", "Middle Eastern"]  # fmt: skip
EDUCATION = ["No HS", "High School Graduate", "Some College", "2-Year", "4-Year",
             "Post-Grad"]  # fmt: skip
PARTY_ID = ["Strong Democrat", "Not Very Strong Democrat", "Lean Democrat",
            "Independent", "Lean Republican", "Not Very Strong Republican",
            "Strong Republican", "Not Sure", "Don't Know"]  # fmt: skip
IDEOLOGY = ["Very Liberal", "Liberal", "Moderate", "Conservative",
            "Very Conservative", "Not Sure"]  # fmt: skip
# A few states to spot-check that backstories name the right state
SPOT_CHECK = {"OH": "Ohio", "TX": "Texas", "ME": "Maine", "NE": "Nebraska",
              "MI": "Michigan", "WY": "Wyoming"}  # fmt: skip


@pytest.fixture(scope="module")
def ces():
    return pl.read_parquet(RAW)


@pytest.fixture(scope="module", params=[2024, 2026])
def personas(request):
    return request.param, pl.read_parquet(PERSONAS[request.param])


#### CES subset ####
ces_schema = pa.DataFrameSchema(
    {
        "year": pa.Column(pl.Int32, pa.Check.isin([2022, 2024])),
        "case_id": pa.Column(pl.Int64),
        "st": pa.Column(str, pa.Check.str_length(2, 2)),
        "age": pa.Column(pl.Int32, pa.Check.in_range(18, 110)),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female"]), nullable=True),
        "race": pa.Column(str, pa.Check.isin(RACES), nullable=True),
        "educ": pa.Column(str, pa.Check.isin(EDUCATION), nullable=True),
        "pid7": pa.Column(str, pa.Check.isin(PARTY_ID), nullable=True),
        "ideo5": pa.Column(str, pa.Check.isin(IDEOLOGY), nullable=True),
        "weight": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    strict=False,
)


def test_ces_schema(ces):
    # Also catches numeric codes that were not converted to labels
    ces_schema.validate(ces)


def test_ces_has_both_years(ces):
    counts = dict(ces.group_by("year").len().iter_rows())
    assert set(counts) == {2022, 2024}
    assert all(n > 50_000 for n in counts.values())


def test_case_id_unique_within_year(ces):
    assert (
        not ces.select(pl.struct("year", "case_id").is_duplicated()).to_series().any()
    )


def test_ces_covers_all_states(ces):
    # 50 states plus DC
    assert ces.filter(pl.col("year") == 2024)["st"].n_unique() == 51


#### Personas ####
def test_persona_states_match_senate_races(personas):
    year, df = personas
    assert set(df["state"]) == set(SENATE_STATES[year])


def test_at_most_150_per_state(personas):
    _, df = personas
    assert df.group_by("state").len()["len"].max() <= 150


def test_enough_personas_per_state(personas):
    # Small states may have fewer than 150, but too few makes estimates useless
    _, df = personas
    assert df.group_by("state").len()["len"].min() >= 50


def test_persona_ids_unique(personas):
    _, df = personas
    assert df["persona_id"].is_unique().all()


def test_target_year_column(personas):
    year, df = personas
    assert (df["target_year"] == year).all()


def test_backstory_has_no_placeholders(personas):
    # Missing answers must be skipped, never written into the text
    _, df = personas
    for bad in ["None", "null", "Don't Know", "Not Sure", "Prefer not to say"]:
        assert not df["backstory"].str.contains(bad, literal=True).any(), bad


def test_backstory_names_the_right_state(personas):
    _, df = personas
    for abbrev, name in SPOT_CHECK.items():
        rows = df.filter(pl.col("state") == abbrev)
        assert rows["backstory"].str.contains(f"living in {name}.").all(), abbrev


def test_backstory_starts_first_person(personas):
    _, df = personas
    assert df["backstory"].str.starts_with("I am a").all()


def test_every_persona_has_a_party_sentence(personas):
    _, df = personas
    assert df["backstory"].str.contains("In politics, I").all()


def test_weights_positive(personas):
    _, df = personas
    assert (df["weight"].drop_nulls() > 0).all()


def test_2024_answer_key_present():
    # The backtest needs real votes for a good share of 2024 personas
    df = pl.read_parquet(PERSONAS[2024])
    share_with_vote = df["voted_sen_party"].is_not_null().mean()
    assert share_with_vote > 0.4


#### Senate races table ####
@pytest.fixture(scope="module")
def races():
    return pl.read_csv("data/01-raw_data/senate_races.csv")


def test_races_cover_the_right_states(races):
    for year, states in SENATE_STATES.items():
        assert set(races.filter(pl.col("year") == year)["state"]) == states


def test_one_row_per_race(races):
    assert (
        not races.select(pl.struct("year", "state").is_duplicated()).to_series().any()
    )


def test_every_race_has_a_republican_and_an_opponent(races):
    assert races["republican"].is_not_null().all()
    has_opponent = races["democrat"].is_not_null() | races["independent"].is_not_null()
    assert has_opponent.all()


def test_main_opponent_is_on_the_ballot(races):
    for row in races.iter_rows(named=True):
        column = "democrat" if row["main_opponent"] == "Democratic" else "independent"
        assert row[column], (row["year"], row["state"])


def test_every_race_has_a_source(races):
    assert races["source"].str.starts_with("https://").all()


def test_personas_match_races(personas, races):
    year, df = personas
    assert set(df["state"]) == set(races.filter(pl.col("year") == year)["state"])
