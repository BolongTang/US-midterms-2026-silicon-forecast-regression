#### Preamble ####
# Purpose: Download raw data (CES, candidates, polls, past results) into data/01-raw_data
# Author: AUTHOR
# Date: DATE
# Contact: EMAIL
# License: MIT
# Pre-requisites: None

#### Workspace setup ####
from pathlib import Path

import httpx
import polars as pl

RAW_DIR = Path("data/01-raw_data")

#### Download data ####

