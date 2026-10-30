#### Preamble ####
# Purpose: Turn CES 2024 respondents into first-person backstories (personas)
#          for (1) the 2024 Senate backtest and (2) the 2026 Senate forecast.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Run 01-download_data.py
#
# Backstories follow Argyle et al. (2023): first person, built from the
# respondent's own survey answers. Both persona sets use CES 2024 respondents;
# what differs is which states are included (the states with a Senate race that
# year). The 2024 set keeps each respondent's reported Senate vote, which is the
# answer key for the backtest.
#
# Run with: uv run scripts/02-build_personas.py

#### Workspace setup ####
from pathlib import Path

import polars as pl

IN_PATH = Path("data/01-raw_data/ces_2022_2024.parquet")
OUT_DIR = Path("data/02-analysis_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_STATE = 150  # personas per state (fewer if the state has fewer respondents)
SEED = 853

# States with a Senate race. Nebraska had two races in 2024 (regular and special);
# CES records one Senate vote per respondent, so we treat it as one race for now.
SENATE_STATES = {
    2024: [
        "AZ", "CA", "CT", "DE", "FL", "HI", "IN", "ME", "MD", "MA", "MI",
        "MN", "MS", "MO", "MT", "NE", "NV", "NJ", "NM", "NY", "ND", "OH",
        "PA", "RI", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    ],
    2026: [
        "AL", "AK", "AR", "CO", "DE", "FL", "GA", "ID", "IL", "IA", "KS",
        "KY", "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ",
        "NM", "NC", "OH", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA",
        "WV", "WY",
    ],
}  # fmt: skip

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "the District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}  # fmt: skip

#### Text for each answer ####
# None means "leave this detail out of the backstory" (e.g. "Don't Know").
RACE = {
    "White": "white", "Black": "Black", "Hispanic": "Hispanic", "Asian": "Asian",
    "Native American": "Native American", "Mixed": "multiracial",
    "Middle Eastern": "Middle Eastern", "Other": None,
}  # fmt: skip
GENDER = {"Male": "man", "Female": "woman"}
EDUC = {
    "No HS": "I did not finish high school.",
    "High School Graduate": "My highest level of education is a high school diploma.",
    "Some College": "I have some college education but no degree.",
    "2-Year": "My highest level of education is a two-year college degree.",
    "4-Year": "My highest level of education is a four-year college degree.",
    "Post-Grad": "My highest level of education is a postgraduate degree.",
}
PARTY = {
    "Strong Democrat": "a strong Democrat",
    "Not Very Strong Democrat": "a Democrat, but not a strong one",
    "Lean Democrat": "an independent who leans toward the Democratic Party",
    "Independent": "an independent who does not lean toward either party",
    "Lean Republican": "an independent who leans toward the Republican Party",
    "Not Very Strong Republican": "a Republican, but not a strong one",
    "Strong Republican": "a strong Republican",
    "Not Sure": None,
    "Don't Know": None,
}
IDEOLOGY = {
    "Very Liberal": "very liberal",
    "Liberal": "liberal",
    "Moderate": "moderate",
    "Conservative": "conservative",
    "Very Conservative": "very conservative",
    "Not Sure": None,
}
CHURCH = {
    "More Than Once a Week": "more than once a week",
    "Once a Week": "once a week",
    "Once or Twice a Month": "once or twice a month",
    "A Few Times a Year": "a few times a year",
    "Seldom": "seldom",
    "Never": "never",
    "Don't Know": None,
}
NEWS = {
    "Most of the time": "most of the time",
    "Some of the time": "some of the time",
    "Only now and then": "only now and then",
    "Hardly at all": "hardly at all",
    "Don't Know": None,
}


def income_text(faminc: str | None) -> str | None:
    """'20k - 30k' -> 'between $20,000 and $30,000 a year'."""
    if faminc is None or faminc == "Prefer not to say":
        return None

    def dollars(k: str) -> str:
        return f"${int(k.strip().rstrip('k+')) * 1000:,}"

    if faminc.startswith("Less than"):
        return f"less than {dollars(faminc.removeprefix('Less than'))} a year"
    if faminc.endswith("+"):
        return f"{dollars(faminc)} or more a year"
    low, high = faminc.split(" - ")
    return f"between {dollars(low)} and {dollars(high)} a year"


def make_backstory(row: dict) -> str:
    """Build a first-person backstory, skipping any detail that is missing."""
    race = RACE.get(row["race"])
    gender = GENDER.get(row["gender"], "person")
    who = " ".join(x for x in [f"{row['age']}-year-old", race, gender] if x)
    parts = [f"I am a {who} living in {STATE_NAMES[row['st']]}."]

    if EDUC.get(row["educ"]):
        parts.append(EDUC[row["educ"]])
    if income := income_text(row["faminc"]):
        parts.append(f"My family income is {income}.")

    party = PARTY.get(row["pid7"])
    parts.append(
        f"In politics, I consider myself {party}."
        if party
        else "In politics, I am not sure which party I identify with."
    )
    if ideology := IDEOLOGY.get(row["ideo5"]):
        parts.append(f"I would describe my political views as {ideology}.")
    if church := CHURCH.get(row["relig_church"]):
        parts.append(
            "I never attend religious services."
            if church == "never"
            else f"I attend religious services {church}."
        )
    if news := NEWS.get(row["newsint"]):
        parts.append(
            f"I follow what's going on in government and public affairs {news}."
        )
    return " ".join(parts)


#### Build persona sets ####
ces = pl.read_parquet(IN_PATH).filter(
    pl.col("year") == 2024, pl.col("age").is_not_null(), pl.col("st").is_not_null()
)

for target_year, states in SENATE_STATES.items():
    personas = (
        ces.filter(pl.col("st").is_in(states))
        # Shuffle with a seed, then keep the first N in each state
        .sample(fraction=1.0, shuffle=True, seed=SEED)
        .group_by("st", maintain_order=True)
        .head(N_PER_STATE)
        .with_columns(
            backstory=pl.struct(pl.all()).map_elements(
                make_backstory, return_dtype=pl.String
            )
        )
        .select(
            persona_id="case_id",
            target_year=pl.lit(target_year),
            state="st",
            age="age",
            gender="gender",
            race="race",
            educ="educ",
            faminc="faminc",
            pid7="pid7",
            ideo5="ideo5",
            relig_church="relig_church",
            newsint="newsint",
            weight="weight",
            # Only meaningful for 2024: the backtest answer key
            voted_sen_party="voted_sen_party",
            backstory="backstory",
        )
        .sort("state", "persona_id")
    )

    path = OUT_DIR / f"personas_{target_year}.parquet"
    personas.write_parquet(path)
    per_state = personas.group_by("state").len()
    print(
        f"{target_year}: {personas.height:,} personas in {per_state.height} states "
        f"(min {per_state['len'].min()} per state) -> {path}"
    )

print("\nExample backstories:")
for b in personas.sample(n=3, seed=SEED)["backstory"]:
    print(" -", b)
