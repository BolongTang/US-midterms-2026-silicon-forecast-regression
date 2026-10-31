#### Preamble ####
# Purpose: Ask LLMs (through OpenRouter) how each persona votes in their state's
#          Senate race. Every call is saved with model, timestamp, tokens and cost.
# Author: Yi Zhi Zhang
# Date: 28 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: A personas parquet with columns persona_id, state, backstory
#                 (from 00-simulate_data.py or 02-build_personas.py), and
#                 OPENROUTER_API_KEY in .env
#
# Examples:
#   Print three prompts, no API calls:
#     uv run scripts/03-query_models.py --dry-run --limit 3
#   Small paid pilot (give it its own run name):
#     uv run scripts/03-query_models.py --limit 20 --repeats 2 --run-name pilot
#   Real run:
#     uv run scripts/03-query_models.py \
#       --personas data/02-analysis_data/personas_2026.parquet --year 2026
#   Name the candidates in the prompt (default is party labels only):
#     ... --names
#
# The ballot for each race (which parties have a candidate) comes from
# data/01-raw_data/senate_races.csv.

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
DEFAULT_MODELS = ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]

RACES_PATH = Path("data/01-raw_data/senate_races.csv")
VALID_ANSWERS = ["Republican", "Democrat", "Independent", "Other", "Would not vote"]
PARTY_LABEL = {
    "republican": "Republican",
    "democrat": "Democrat",
    "independent": "Independent",
}

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "the District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}  # fmt: skip

# Wording follows Bisbee et al. (2024): answer as the person, using only
# knowledge they would have.
SYSTEM_PROMPT = (
    "You will be given a description of a person. Answer from this person's "
    "perspective, using only knowledge about politics that they would have."
)


#### Prompt building ####
def load_races(year: int) -> dict[str, dict]:
    """One dict per state: which parties have a candidate, and their names."""
    races = pl.read_csv(RACES_PATH).filter(pl.col("year") == year)
    return {row["state"]: row for row in races.iter_rows(named=True)}


def describe_ballot(race: dict, names: bool) -> tuple[str, list[str]]:
    """e.g. ("a Republican and an independent", ["Republican", "Independent"]).
    With names=True: "Republican Pete Ricketts and independent Dan Osborn"."""
    parts, options = [], []
    for column, label in PARTY_LABEL.items():
        if race[column]:
            options.append(label)
            if names:
                party = label if label != "Independent" else "independent"
                parts.append(f"{party} {race[column]}")
            else:
                article = "an" if label == "Independent" else "a"
                parts.append(
                    f"{article} {label if label != 'Independent' else 'independent'}"
                )
    if len(parts) == 2:
        text = f"{parts[0]} and {parts[1]}"
    else:
        text = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return text, options


def build_prompt(
    backstory: str, race: dict, year: int, names: bool = False, forced: bool = False
) -> str:
    """Backstory followed by a ballot question for that race.

    forced=True asks a follow-up that allows only the Republican and the main
    opponent, like a survey's "if you had to choose" question for undecided
    respondents. We use it for pure independents, who otherwise mostly answer
    "Other".
    """
    state_name = STATE_NAMES[race["state"]]
    ballot, options = describe_ballot(race, names)
    election = "special election" if race["special"] else "election"
    if year == 2024:
        question = (
            f"It is November 2024. In the 2024 U.S. Senate {election} in "
            f"{state_name}, the main candidates were {ballot}. I voted for"
        )
    else:
        question = (
            f"It is November {year}. In this year's midterm U.S. Senate {election} "
            f"in {state_name}, the main candidates are {ballot}. I will vote for"
        )
    if forced:
        democratic = race["main_opponent"] == "Democratic"
        opponent = "Democrat" if democratic else "Independent"
        described = "Democratic" if democratic else "independent"
        verb = "would you have voted" if year == 2024 else "would you vote"
        follow_up = (
            f"If you had to choose between the Republican candidate and the "
            f"{described} candidate, which {verb} for?"
        )
        return (
            f"{backstory}\n\n{question} ...\n\n{follow_up}\n\n"
            f"Reply with exactly one of: Republican, {opponent}. No other text."
        )
    choices = ", ".join([*options, "Other", "Would not vote"])
    return (
        f"{backstory}\n\n{question} ...\n\n"
        f"Reply with exactly one of: {choices}. No other text."
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
        "Independent": r"\bindependent",
        "Other": r"\bother\b",
    }
    found = [label for label, pattern in patterns.items() if re.search(pattern, t)]
    return found[0] if len(found) == 1 else None  # None = NA, caught by tests


#### API calls ####
async def ask(client, semaphore, model, prompt, temperature, settings, max_retries=4):
    """One API call with retries on rate limits and server errors."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": settings["max_tokens"],
        "usage": {"include": True},  # ask OpenRouter to report cost
    }
    if settings["no_reasoning"]:
        # Some models (e.g. DeepSeek V4) "think" before answering, which can use
        # up the whole token budget and leave an empty reply. Turn that off.
        payload["reasoning"] = {"enabled": False}
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


async def run(
    personas,
    races,
    models,
    repeats,
    year,
    temperature,
    concurrency,
    names,
    forced,
    settings,
):
    load_dotenv()
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    semaphore = asyncio.Semaphore(concurrency)
    jobs, meta = [], []
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        for row in personas.iter_rows(named=True):
            prompt = build_prompt(
                row["backstory"], races[row["state"]], year, names, forced
            )
            for model in models:
                for repeat in range(repeats):
                    jobs.append(
                        ask(client, semaphore, model, prompt, temperature, settings)
                    )
                    meta.append(
                        {
                            "persona_id": row["persona_id"],
                            "state": row["state"],
                            "model": model,
                            "repeat": repeat,
                            "year": year,
                            "temperature": temperature,
                            "names_in_prompt": names,
                            "forced_choice": forced,
                            "max_tokens": settings["max_tokens"],
                            "no_reasoning": settings["no_reasoning"],
                        }
                    )
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
    p.add_argument("--names", action="store_true", help="name candidates in prompt")
    p.add_argument("--max-tokens", type=int, default=25, help="reply length limit")
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="ask reasoning models to answer without thinking first",
    )
    p.add_argument(
        "--forced-choice",
        action="store_true",
        help="ask 'if you had to choose' between the Republican and main opponent",
    )
    p.add_argument(
        "--only-pid7",
        nargs="+",
        help="only query personas with this party ID, e.g. --only-pid7 Independent",
    )
    p.add_argument(
        "--batch-size", type=int, default=200, help="personas per saved batch"
    )
    p.add_argument(
        "--run-name",
        help="output name (default: the year, e.g. responses_2024.parquet). "
        "Use a new name for pilots so they don't mix with the full run.",
    )
    args = p.parse_args()

    personas = pl.read_parquet(args.personas)
    races = load_races(args.year)
    unmatched = set(personas["state"]) - set(races)
    if unmatched:
        raise SystemExit(f"No {args.year} race in senate_races.csv for: {unmatched}")
    if args.only_pid7:
        personas = personas.filter(pl.col("pid7").is_in(args.only_pid7))
    if args.limit:
        # Sample across states rather than taking the first N rows of one state
        personas = personas.sample(n=args.limit, seed=853)

    if args.dry_run:
        for row in personas.head(3).iter_rows(named=True):
            race = races[row["state"]]
            prompt = build_prompt(
                row["backstory"], race, args.year, args.names, args.forced_choice
            )
            print(prompt, "\n---")
        n_calls = personas.height * len(args.models) * args.repeats
        print(f"Would send {n_calls} requests to: {', '.join(args.models)}")
        return

    # Query in batches of personas. Each batch is saved as soon as it finishes,
    # so a crash loses at most one batch, and rerunning the same command skips
    # personas that are already done.
    out_dir = Path("data/02-analysis_data")
    run_name = args.run_name or (
        f"{args.year}{'_names' if args.names else ''}"
        f"{'_forced' if args.forced_choice else ''}"
    )
    parts_dir = out_dir / f"parts_{run_name}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    for part in parts_dir.glob("batch_*.parquet"):
        done |= set(pl.read_parquet(part, columns=["persona_id"])["persona_id"])
    todo = personas.filter(~pl.col("persona_id").is_in(list(done)))
    if done:
        print(f"Resuming: {len(done)} personas already done, {todo.height} to go")

    n_batches = -(-todo.height // args.batch_size)  # ceiling division
    for i, batch in enumerate(todo.iter_slices(args.batch_size), start=1):
        result = asyncio.run(
            run(
                batch,
                races,
                args.models,
                args.repeats,
                args.year,
                args.temperature,
                args.concurrency,
                args.names,
                args.forced_choice,
                {"max_tokens": args.max_tokens, "no_reasoning": args.no_reasoning},
            )
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        result.write_parquet(parts_dir / f"batch_{stamp}.parquet")
        spent = result["cost_usd"].sum() or 0
        print(f"  batch {i}/{n_batches} saved ({result.height} replies, ${spent:.4f})")

    # Combine all batches into one file and parse the answers
    out = pl.concat(
        [pl.read_parquet(f) for f in sorted(parts_dir.glob("batch_*.parquet"))],
        how="diagonal_relaxed",
    ).with_columns(
        answer=pl.col("raw_reply").map_elements(parse_answer, return_dtype=pl.String)
    )
    path = out_dir / f"responses_{run_name}.parquet"
    out.write_parquet(path)
    print(f"Saved {out.height} replies to {path}")
    print(
        out.group_by("model").agg(
            valid=pl.col("answer").is_not_null().mean(),
            errors=pl.col("error").is_not_null().sum(),
            cost_usd=pl.col("cost_usd").sum(),
        )
    )
    print(out.group_by("answer").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
