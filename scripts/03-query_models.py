#### Preamble ####
# Purpose: Ask LLMs (through OpenRouter) how each persona votes in their state's
#          Senate race. Every call is saved with model, timestamp, tokens and cost.
# Author: AUTHOR
# Date: DATE
# Contact: EMAIL
# License: MIT
# Pre-requisites: A personas parquet with columns persona_id, state, backstory
#                 (from 00-simulate_data.py or 02-build_personas.py), and
#                 OPENROUTER_API_KEY in .env
#
# Examples:
#   Print three prompts, no API calls:
#     uv run scripts/03-query_models.py --dry-run --limit 3
#   Small paid pilot on simulated personas:
#     uv run scripts/03-query_models.py --limit 20 --repeats 2
#   Real run:
#     uv run scripts/03-query_models.py \
#       --personas data/02-analysis_data/personas_2026.parquet --year 2026

#### Workspace setup ####
import argparse
import asyncio
import os
import random
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
from dotenv import load_dotenv

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Check each model's page on openrouter.ai for availability, price, and
# knowledge cutoff before using it. Use at least two providers.
DEFAULT_MODELS = ["openai/gpt-4o-mini", "meta-llama/llama-3.1-8b-instruct"]

VALID_ANSWERS = ["Republican", "Democrat", "Other", "Would not vote"]

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AR": "Arkansas", "CO": "Colorado",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "ID": "Idaho",
    "IL": "Illinois", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MT": "Montana", "NE": "Nebraska",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NC": "North Carolina", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "VA": "Virginia", "WV": "West Virginia",
    "WY": "Wyoming",
}  # fmt: skip

# Wording follows Bisbee et al. (2024): answer as the person, using only
# knowledge they would have.
SYSTEM_PROMPT = (
    "You will be given a description of a person. Answer from this person's "
    "perspective, using only knowledge about politics that they would have."
)


#### Prompt building ####
def build_prompt(backstory: str, state: str, year: int) -> str:
    """Backstory followed by a forced-choice ballot question."""
    state_name = STATE_NAMES[state]
    if year == 2024:
        question = (
            f"It is November 2024. In the 2024 U.S. Senate election in {state_name}, "
            "between the Republican candidate and the Democratic candidate, I voted for"
        )
    else:
        question = (
            f"It is November {year}. In this year's midterm U.S. Senate election in "
            f"{state_name}, between the Republican candidate and the Democratic "
            "candidate, I will vote for"
        )
    options = ", ".join(VALID_ANSWERS)
    return (
        f"{backstory}\n\n{question} ...\n\n"
        f"Reply with exactly one of: {options}. No other text."
    )


def parse_answer(text: str | None) -> str | None:
    """Map a raw reply to one of VALID_ANSWERS, or None if unusable.

    Looks for each option anywhere in the reply, so "I will vote for the
    Democrat." counts. A reply that mentions more than one option (e.g.
    "Democrat or Republican") is ambiguous and returns None.
    """
    if not text:
        return None
    t = text.lower()
    if re.search(r"would not|wouldn't|won't vote|not vote|not going to vote", t):
        return "Would not vote"
    patterns = {
        "Republican": r"\brepublican",  # also matches "Republicans"
        "Democrat": r"\bdemocrat",  # also matches "Democratic"
        "Other": r"\bother\b",
    }
    found = [label for label, pattern in patterns.items() if re.search(pattern, t)]
    return found[0] if len(found) == 1 else None  # None = NA, caught by tests


#### API calls ####
async def ask(client, semaphore, model, prompt, temperature, max_retries=4):
    """One API call with retries on rate limits and server errors."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 25,
        "usage": {"include": True},  # ask OpenRouter to report cost
    }
    async with semaphore:
        for attempt in range(max_retries):
            try:
                r = await client.post(API_URL, json=payload)
                if r.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2**attempt + random.random())
                    continue
                r.raise_for_status()
                body = r.json()
                usage = body.get("usage", {})
                return {
                    "raw_reply": body["choices"][0]["message"].get("content"),
                    "model_returned": body.get("model"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cost_usd": usage.get("cost"),
                    "error": None,
                }
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_error = repr(e)
                await asyncio.sleep(2**attempt)
        return {
            "raw_reply": None,
            "model_returned": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
            "error": locals().get("last_error", "rate limited"),
        }


async def run(personas, models, repeats, year, temperature, concurrency):
    load_dotenv()
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    semaphore = asyncio.Semaphore(concurrency)
    jobs, meta = [], []
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        for row in personas.iter_rows(named=True):
            prompt = build_prompt(row["backstory"], row["state"], year)
            for model in models:
                for repeat in range(repeats):
                    jobs.append(ask(client, semaphore, model, prompt, temperature))
                    meta.append(
                        {
                            "persona_id": row["persona_id"],
                            "state": row["state"],
                            "model": model,
                            "repeat": repeat,
                            "year": year,
                            "temperature": temperature,
                        }
                    )
        print(f"Sending {len(jobs)} requests...")
        results = await asyncio.gather(*jobs)
    queried_at = datetime.now(UTC).isoformat()
    rows = [{**m, **r, "queried_at": queried_at} for m, r in zip(meta, results)]
    return pl.DataFrame(rows, infer_schema_length=None)


#### Main ####
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--personas", default="data/00-simulated_data/simulated_personas.parquet"
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--year", type=int, default=2026, choices=[2024, 2026])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="only use the first N personas (pilots)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="print prompts, call nothing")
    args = p.parse_args()

    personas = pl.read_parquet(args.personas)
    if args.limit:
        # Sample across states rather than taking the first N rows of one state
        personas = personas.sample(n=args.limit, seed=853)

    if args.dry_run:
        for row in personas.head(3).iter_rows(named=True):
            print(build_prompt(row["backstory"], row["state"], args.year), "\n---")
        n_calls = personas.height * len(args.models) * args.repeats
        print(f"Would send {n_calls} requests to: {', '.join(args.models)}")
        return

    out = asyncio.run(
        run(
            personas,
            args.models,
            args.repeats,
            args.year,
            args.temperature,
            args.concurrency,
        )
    )
    out_dir = Path("data/02-analysis_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"responses_{args.year}_{stamp}.parquet"

    # Save the raw replies first, so a parsing bug can never lose paid calls
    out.write_parquet(path)
    print(f"Saved {out.height} raw replies to {path}")

    # Then parse and save again with the answer column added
    out = out.with_columns(
        answer=pl.col("raw_reply").map_elements(parse_answer, return_dtype=pl.String)
    )
    out.write_parquet(path)
    print(
        out.group_by("model").agg(
            valid=pl.col("answer").is_not_null().mean(),
            cost_usd=pl.col("cost_usd").sum(),
        )
    )
    print(out.group_by("answer").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
