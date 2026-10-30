#### Preamble ####
# Purpose: Download state-level demographics from the Census Bureau ACS 5-year
#          estimates for 2014-2024 election cycles (2024 is reused for the 2026 forecast).
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Internet access (api.census.gov) and a free Census API key
#   (https://api.census.gov/data/key_signup.html), supplied as the environment
#   variable CENSUS_API_KEY (or in .env; see .env.example).
#
# Outputs (in data/01-raw_data/):
#   - acs_state.parquet: one row per state and ACS year, with shares in [0, 1]
#
# Run with: uv run scripts/01b-download_acs.py

#### Workspace setup ####
from pathlib import Path

import os

import httpx
import polars as pl
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/01-raw_data")
RAW_DIR.mkdir(parents=True, exist_ok=True)
ACS_YEARS = [2014, 2016, 2018, 2020, 2022, 2024]  # 5-year ACS ending in these years

# Variable codes from tables B01001, B01002, B02001, B03003, B15003, B19013
VARIABLES = {
    "B01003_001E": "pop",
    "B01001_002E": "male",
    "B01002_001E": "median_age",
    "B02001_002E": "white_alone",
    "B02001_003E": "black_alone",
    "B03003_003E": "hispanic",
    "B15003_001E": "educ_total",
    "B15003_022E": "bachelors",
    "B15003_023E": "masters",
    "B15003_024E": "professional",
    "B15003_025E": "doctorate",
    "B19013_001E": "median_income",
}

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip


#### Download ####
MAX_VARIABLES = 50  # the Census API returns at most 50 variables per request


def chunks(codes: list[str], size: int = MAX_VARIABLES - 1) -> list[list[str]]:
    """Split variable codes into request-sized groups (NAME takes one slot)."""
    return [codes[i : i + size] for i in range(0, len(codes), size)]


def fetch_year(year: int) -> pl.DataFrame:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise SystemExit("CENSUS_API_KEY is not set (environment variable or .env).")
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    frames = []
    for group in chunks(list(VARIABLES)):
        params = {"get": "NAME," + ",".join(group), "for": "state:*", "key": key}
        r = httpx.get(url, params=params, timeout=60)
        if r.status_code != 200:  # never print the URL: it contains the key
            hint = " (Census rejected the key: check it and activate it via the email link)" if "invalid_key" in r.headers.get("location", "") else ""
            raise SystemExit(f"ACS {year}: HTTP {r.status_code}{hint}")
        header, *rows = r.json()
        frames.append(pl.DataFrame(rows, schema=header, orient="row").drop("state"))
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, on="NAME")
    df = df.filter(pl.col("NAME").is_in(list(STATES)))  # drops Puerto Rico
    return df.select(
        acs_year=pl.lit(year),
        st=pl.col("NAME").replace_strict(STATES),
        **{new: pl.col(old).cast(pl.Float64) for old, new in VARIABLES.items()},
    )


def to_shares(df: pl.DataFrame) -> pl.DataFrame:
    """Turn counts into the shares used as regression features."""
    return df.select(
        "acs_year",
        "st",
        acs_male=pl.col("male") / pl.col("pop"),
        acs_white=pl.col("white_alone") / pl.col("pop"),
        acs_black=pl.col("black_alone") / pl.col("pop"),
        acs_hispanic=pl.col("hispanic") / pl.col("pop"),
        acs_college=(
            pl.col("bachelors") + pl.col("masters")
            + pl.col("professional") + pl.col("doctorate")
        ) / pl.col("educ_total"),
        acs_median_age="median_age",
        acs_log_income=pl.col("median_income").log(),
    )  # fmt: skip


if __name__ == "__main__":
    out = pl.concat([to_shares(fetch_year(y)) for y in ACS_YEARS])
    out.write_parquet(RAW_DIR / "acs_state.parquet")
    print(f"Saved {out.height} state-years to {RAW_DIR / 'acs_state.parquet'}")
