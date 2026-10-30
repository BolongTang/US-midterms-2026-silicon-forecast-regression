#### Preamble ####
# Purpose: Download certified statewide U.S. Senate returns and keep the general-
#          election Democratic vs Republican contest for each state and cycle, for
#          checking the regression against real results.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Internet access. No account needed.
#
# Source: MIT Election Data and Science Lab, "U.S. Senate statewide 1976-2024",
#         Harvard Dataverse, doi:10.7910/DVN/PEJ5QU
#
# Coding rules (per the project owner):
#   - Independent senators who caucus with a party are coded by that party. Their
#     affiliation rarely changes: Angus King (ME) and Bernie Sanders (VT) caucus with
#     the Democrats. Independents who do not caucus with a party (e.g. Dan Osborn in
#     Nebraska 2024, Evan McMullin, Greg Orman) stay "other".
#   - Special and regular races are treated alike: the rules and dates are the same.
#   - General elections only (runoffs excluded). A race counts only if, after the coding
#     above, it has both a Democratic and a Republican candidate (this drops California
#     2016/2018, where two Democrats advanced, and Nebraska 2024's regular race, where
#     Osborn ran against Fischer). A state with two qualifying races in one year (e.g.
#     Georgia 2020) pools their votes.
#
# Outputs (in data/01-raw_data/):
#   - 1976-2024-senate-state.csv   the file as downloaded
#   - senate_results.csv           year, st, dem_votes, rep_votes, other_votes, n_races
#
# Run with: uv run scripts/01c-download_results.py

#### Workspace setup ####
from pathlib import Path

import httpx
import polars as pl

RAW_DIR = Path("data/01-raw_data")
FILE_ID = 13887039  # 1976-2024-senate-state.csv
YEARS = [2014, 2016, 2018, 2020, 2022, 2024]


#### Build results ####
# Independents who caucus with a party: (state, candidate-name fragment) -> party
CAUCUS_PARTY = {("ME", "KING"): "DEMOCRAT", ("VT", "SANDERS"): "DEMOCRAT"}


def code_party(raw: pl.DataFrame, caucus: dict = CAUCUS_PARTY) -> pl.DataFrame:
    """party_simplified, with caucusing independents recoded to their caucus party."""
    expr = pl.col("party_simplified")
    for (st, name), party in caucus.items():
        hit = (pl.col("state_po") == st) & pl.col("candidate").str.to_uppercase().str.contains(name) & (pl.col("party_detailed") == "INDEPENDENT")
        expr = pl.when(hit).then(pl.lit(party)).otherwise(expr)
    return raw.with_columns(party_coded=expr)


def build_results(raw: pl.DataFrame, years: list[int] = YEARS, caucus: dict = CAUCUS_PARTY) -> pl.DataFrame:
    races = (
        code_party(raw, caucus)
        .filter(pl.col("year").is_in(years), pl.col("stage").str.to_lowercase() == "gen")
        .group_by("year", "state_po", "special")
        .agg(
            dem_votes=pl.col("candidatevotes").filter(pl.col("party_coded") == "DEMOCRAT").sum(),
            rep_votes=pl.col("candidatevotes").filter(pl.col("party_coded") == "REPUBLICAN").sum(),
            other_votes=pl.col("candidatevotes").filter(~pl.col("party_coded").is_in(["DEMOCRAT", "REPUBLICAN"])).sum(),
        )
        .filter(pl.col("dem_votes") > 0, pl.col("rep_votes") > 0)
    )  # fmt: skip
    return (
        races.group_by("year", "state_po")  # pool a state's qualifying races, special or regular
        .agg(pl.col("dem_votes").sum(), pl.col("rep_votes").sum(), pl.col("other_votes").sum(), n_races=pl.len())
        .rename({"state_po": "st"})
        .cast({"dem_votes": pl.Int64, "rep_votes": pl.Int64, "other_votes": pl.Int64})
        .sort("year", "st")
    )


def download(path: Path) -> pl.DataFrame:
    if not path.exists():
        url = f"https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}"
        r = httpx.get(url, params={"format": "original"}, timeout=120, follow_redirects=True)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pl.read_csv(path, infer_schema_length=10000)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = download(RAW_DIR / "1976-2024-senate-state.csv")
    out = build_results(raw)
    out.write_csv(RAW_DIR / "senate_results.csv")
    print(out.group_by("year").agg(races=pl.len(), pooled=(pl.col("n_races") > 1).sum()).sort("year"))
    print(f"Saved {out.height} state-races to {RAW_DIR / 'senate_results.csv'}")
