#### Preamble ####
# Purpose: Download the Cumulative CES Common Content (2006-2025) from Harvard
#          Dataverse and save the 2022 and 2024 respondents we need.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Internet access. No account needed.
#
# Source: Kuriwaki, Shiro. Cumulative CES Common Content. Harvard Dataverse.
#         doi:10.7910/DVN/II2DB6
#
# Outputs (in data/01-raw_data/):
#   - ces_2016_2020.parquet: earlier cycles for the regression (committed)
#   - the full cumulative file as downloaded (large, gitignored)
#   - ces_2022_2024.parquet: 2022 and 2024 rows, selected columns (committed)
#   - ces_2022_2024_sample.parquet: 2,000 rows so scripts run without the full file
#
# Run with: uv run scripts/01-download_data.py

#### Workspace setup ####
from pathlib import Path

import httpx
import polars as pl
import pyarrow as pa
from pyarrow import feather

RAW_DIR = Path("data/01-raw_data")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATAVERSE = "https://dataverse.harvard.edu"
CES_DOI = "doi:10.7910/DVN/II2DB6"
YEARS = [2016, 2018, 2020, 2022, 2024]  # 2016-2020 feed the regression only

# Columns we keep. 2024 gives personas and the backtest answer key
# (voted_sen_party); 2022 gives validated midterm turnout (vv_turnout_gvm).
COLUMNS = [
    "year", "case_id", "st", "age", "gender", "race", "educ", "faminc",
    "pid3", "pid7", "ideo5", "relig_church", "newsint",
    "weight", "weight_post", "vv_turnout_gvm", "voted_turnout_self",
    "intent_sen", "intent_sen_party", "voted_sen", "voted_sen_party",
]  # fmt: skip


#### Find the data file ####
def list_files() -> list[dict]:
    """Ask the Dataverse API which files the latest version of the dataset has."""
    url = f"{DATAVERSE}/api/datasets/:persistentId/versions/:latest/files"
    r = httpx.get(url, params={"persistentId": CES_DOI}, timeout=60)
    r.raise_for_status()
    files = []
    for f in r.json()["data"]:
        df = f["dataFile"]
        files.append(
            {
                "id": df["id"],
                # Dataverse converts .dta to .tab; originalFileName keeps the real name
                "name": df.get("originalFileName", df["filename"]),
                "size_mb": round(df.get("originalFileSize", df["filesize"]) / 1e6, 1),
            }
        )
    return files


def choose_file(files: list[dict]) -> dict:
    """Use the feather file: smaller than .dta and readable without R or Stata."""
    for ext in (".feather",):
        matches = [f for f in files if f["name"].endswith(ext)]
        if matches:
            return max(matches, key=lambda f: f["size_mb"])
    raise RuntimeError("No .feather file found; see the list above.")


#### Download ####
def download(file: dict) -> Path:
    path = RAW_DIR / file["name"]
    if path.exists():
        print(f"Already downloaded: {path}")
        return path
    url = f"{DATAVERSE}/api/access/datafile/{file['id']}"
    print(f"Downloading {file['name']} ({file['size_mb']} MB)...")
    with httpx.stream(
        "GET", url, params={"format": "original"}, timeout=None, follow_redirects=True
    ) as r:
        r.raise_for_status()
        with open(path, "wb") as out:
            out.writelines(r.iter_bytes(chunk_size=1 << 20))
    return path


#### Read and subset ####
# The feather file was written from R. Columns with value labels (haven_labelled)
# arrive as integer codes, so we map them to text here. Codes and labels are
# from the label metadata stored in the file and the CES cumulative guide.
LABELS = {
    "gender": {1: "Male", 2: "Female"},
    "race": {
        1: "White", 2: "Black", 3: "Hispanic", 4: "Asian",
        5: "Native American", 6: "Mixed", 7: "Other", 8: "Middle Eastern",
    },
    "educ": {
        1: "No HS", 2: "High School Graduate", 3: "Some College",
        4: "2-Year", 5: "4-Year", 6: "Post-Grad",
    },
    "pid3": {1: "Democrat", 2: "Republican", 3: "Independent", 4: "Other", 5: "Not Sure"},
    "pid7": {
        1: "Strong Democrat", 2: "Not Very Strong Democrat", 3: "Lean Democrat",
        4: "Independent", 5: "Lean Republican", 6: "Not Very Strong Republican",
        7: "Strong Republican", 8: "Not Sure", 9: "Don't Know",
    },
    "relig_church": {
        1: "More Than Once a Week", 2: "Once a Week", 3: "Once or Twice a Month",
        4: "A Few Times a Year", 5: "Seldom", 6: "Never", 7: "Don't Know",
    },
    "newsint": {
        1: "Most of the time", 2: "Some of the time", 3: "Only now and then",
        4: "Hardly at all", 7: "Don't Know",
    },
}  # fmt: skip


def read_subset(path: Path) -> pl.DataFrame:
    """Read the columns we need for 2022 and 2024, with every column as plain text
    or numbers (no R-specific types)."""
    available = pa.ipc.open_file(path).schema.names
    keep = [c for c in COLUMNS if c in available]
    missing = sorted(set(COLUMNS) - set(keep))
    if missing:
        print(f"Note: columns not in this file and skipped: {missing}")

    table = feather.read_table(path, columns=keep)
    # Drop R metadata and turn R factors (dictionary columns) into text
    plain = pa.schema(
        [
            pa.field(f.name, pa.string() if pa.types.is_dictionary(f.type) else f.type)
            for f in table.schema
        ]
    )
    ces = pl.from_arrow(table.cast(plain)).filter(pl.col("year").is_in(YEARS))

    # Replace integer codes with their labels
    for col, mapping in LABELS.items():
        if col in ces.columns:
            ces = ces.with_columns(
                pl.col(col).replace_strict(
                    mapping, default=None, return_dtype=pl.String
                )
            )
    return ces.with_columns(pl.col("case_id").cast(pl.Int64))


#### Main ####
files = list_files()
print("Files in the dataset:")
for f in files:
    print(f"  {f['name']:<45} {f['size_mb']:>8} MB")

path = download(choose_file(files))
ces = read_subset(path)

# Persona scripts and their tests use only 2022 and 2024; the earlier cycles go
# to their own file for the regression (scripts/05-regression.py).
extra = ces.filter(pl.col("year") < 2022)
if extra.height:
    extra.write_parquet(RAW_DIR / "ces_2016_2020.parquet")
    print(f"Saved {extra.height:,} rows to {RAW_DIR / 'ces_2016_2020.parquet'}")
ces = ces.filter(pl.col("year") >= 2022)

ces.write_parquet(RAW_DIR / "ces_2022_2024.parquet")
ces.sample(n=min(2000, ces.height), seed=853).write_parquet(
    RAW_DIR / "ces_2022_2024_sample.parquet"
)

print(f"Saved {ces.height:,} rows to {RAW_DIR / 'ces_2022_2024.parquet'}")
print(ces.group_by("year").len().sort("year"))
