#### Preamble ####
# Purpose: Test forecast outputs: shares sum to 1, no NAs, probabilities in [0, 1], every race present
# Author: AUTHOR
# Date: DATE
# Contact: EMAIL
# License: MIT
# Pre-requisites: Run scripts/04-forecast.py

#### Workspace setup ####
import pandera.polars as pa
import polars as pl
import pytest

#### Tests ####

