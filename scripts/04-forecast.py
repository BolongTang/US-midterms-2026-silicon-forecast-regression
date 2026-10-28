#### Preamble ####
# Purpose: Post-stratify/combine responses into a distribution over outcomes; save as parquet
# Author: AUTHOR
# Date: DATE
# Contact: EMAIL
# License: MIT
# Pre-requisites: Run 03-query_models.py

#### Workspace setup ####
import numpy as np
import polars as pl

rng = np.random.default_rng(seed=853)

#### Forecast ####

