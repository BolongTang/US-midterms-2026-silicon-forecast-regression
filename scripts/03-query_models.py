#### Preamble ####
# Purpose: Query LLMs from at least two providers via OpenRouter
# Author: AUTHOR
# Date: DATE
# Contact: EMAIL
# License: MIT
# Pre-requisites: Run 02-build_personas.py; OPENROUTER_API_KEY in .env

#### Workspace setup ####
import os

import httpx
import polars as pl
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODELS = []  # at least two providers, e.g. "openai/...", "anthropic/..."

#### Query models ####

