#### Preamble ####
# Purpose: Download certified statewide presidential returns and keep the Democratic
#          and Republican totals per state and election, for the "last presidential
#          result" baseline.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Internet access. No account needed.
#
# Source: MIT Election Data and Science Lab, "U.S. President 1976-2024",
#         Harvard Dataverse, doi:10.7910/DVN/42MVDX
#
# Outputs (in data/01-raw_data/):
#   - 1976-2024-president.csv   the file as downloaded
#   - president_results.csv     year, st, dem_votes, rep_votes, other_votes (2008-2024)
#
# Run with: uv run scripts/01d-download_president.py

#### Workspace setup ####
from pathlib import Path

import httpx
import polars as pl

RAW_DIR = Path("data/01-raw_data")
FILE_ID = 13887042  # 1976-2024-president.csv
YEARS = [2008, 2012, 2016, 2020, 2024]


#### Build results ####
def build_president(raw: pl.DataFrame, years: list[int] = YEARS) -> pl.DataFrame:
    return (
        raw.filter(pl.col("year").is_in(years))
        .group_by("year", "state_po")
        .agg(
            dem_votes=pl.col("candidatevotes").filter(pl.col("party_simplified") == "DEMOCRAT").sum(),
            rep_votes=pl.col("candidatevotes").filter(pl.col("party_simplified") == "REPUBLICAN").sum(),
            other_votes=pl.col("candidatevotes").filter(~pl.col("party_simplified").is_in(["DEMOCRAT", "REPUBLICAN"]) | pl.col("party_simplified").is_null()).sum(),
        )
        .rename({"state_po": "st"})
        .sort("year", "st")
    )  # fmt: skip


def download(path: Path) -> pl.DataFrame:
    if not path.exists():
        url = f"https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}"
        r = httpx.get(url, params={"format": "original"}, timeout=120, follow_redirects=True)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pl.read_csv(path, infer_schema_length=10000)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = build_president(download(RAW_DIR / "1976-2024-president.csv"))
    out.write_csv(RAW_DIR / "president_results.csv")
    print(out.group_by("year").len().sort("year"))
    print(f"Saved {out.height} state-elections to {RAW_DIR / 'president_results.csv'}")
