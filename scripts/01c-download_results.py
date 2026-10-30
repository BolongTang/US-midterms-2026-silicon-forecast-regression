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
# Rules for choosing one race per state and year:
#   - general elections only (runoffs excluded)
#   - the race must have both a Democratic and a Republican candidate
#     (drops, e.g., California 2016 with two Democrats, Maine 2018 and Vermont
#     2018/2024 where an independent replaced the Democrat on the ballot)
#   - if a state had a regular and a special race, use the regular one unless
#     only the special has both parties (Nebraska 2024: Ricketts v. Love)
#
# Outputs (in data/01-raw_data/):
#   - 1976-2024-senate-state.csv   the file as downloaded
#   - senate_results.csv           year, st, dem_votes, rep_votes, other_votes, special
#
# Run with: uv run scripts/01c-download_results.py

#### Workspace setup ####
from pathlib import Path

import httpx
import polars as pl

RAW_DIR = Path("data/01-raw_data")
FILE_ID = 13887039  # 1976-2024-senate-state.csv
YEARS = [2016, 2018, 2020, 2022, 2024]


#### Build results ####
def build_results(raw: pl.DataFrame, years: list[int] = YEARS) -> pl.DataFrame:
    races = (
        raw.filter(pl.col("year").is_in(years), pl.col("stage").str.to_lowercase() == "gen")
        .group_by("year", "state_po", "special")
        .agg(
            dem_votes=pl.col("candidatevotes").filter(pl.col("party_simplified") == "DEMOCRAT").sum(),
            rep_votes=pl.col("candidatevotes").filter(pl.col("party_simplified") == "REPUBLICAN").sum(),
            other_votes=pl.col("candidatevotes").filter(~pl.col("party_simplified").is_in(["DEMOCRAT", "REPUBLICAN"])).sum(),
        )
        .filter(pl.col("dem_votes") > 0, pl.col("rep_votes") > 0)
    )  # fmt: skip
    return (
        races.sort("year", "state_po", "special")  # regular (False) before special (True)
        .group_by("year", "state_po", maintain_order=True)
        .first()
        .rename({"state_po": "st"})
        .select("year", "st", "dem_votes", "rep_votes", "other_votes", "special")
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
    print(out.group_by("year").agg(races=pl.len(), special=pl.col("special").sum()).sort("year"))
    print(f"Saved {out.height} state-races to {RAW_DIR / 'senate_results.csv'}")
