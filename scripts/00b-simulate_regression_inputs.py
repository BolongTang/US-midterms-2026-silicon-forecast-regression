#### Preamble ####
# Purpose: Simulate the inputs of scripts/05-regression.py (CES-like respondents,
#          ACS-like state features, certified-style Senate results) so the
#          regression can run end to end before the real downloads exist.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: None
#
# The simulated world has a known structure: each state has a latent partisan
# lean, party ID and demographics follow from it, and vote intent follows party
# ID, race, education, the lean, and a year-specific national swing. Fitting the
# regression to these data therefore shows whether the code works; it says
# nothing about the real 2026 races.
#
# Outputs (in data/00-simulated_data/):
#   - simulated_ces.parquet           respondents, same columns the regression uses
#   - simulated_acs.parquet           state features by ACS year
#   - simulated_senate_results.csv    year, st, dem_votes, rep_votes, other_votes
#
# Run with: uv run scripts/00b-simulate_regression_inputs.py

#### Workspace setup ####
from pathlib import Path

import numpy as np
import polars as pl

OUT_DIR = Path("data/00-simulated_data")
YEARS = [2014, 2016, 2018, 2020, 2022, 2024]
N_PER_STATE = 300
SEED = 853

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]  # fmt: skip
# National swing toward Democrats by year (logit scale)
SWING = {2014: -0.3, 2016: -0.1, 2018: 0.3, 2020: 0.1, 2022: 0.1, 2024: -0.15}


def logistic(x):
    return 1 / (1 + np.exp(-x))


def simulate(out_dir: Path = OUT_DIR, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # State-level truth: lean (+ = Democratic) and demographic profile
    n_states = len(STATES)
    lean = rng.normal(0, 1, n_states)
    white = np.clip(0.72 - 0.08 * lean + rng.normal(0, 0.1, n_states), 0.25, 0.95)
    college = np.clip(0.33 + 0.05 * lean + rng.normal(0, 0.04, n_states), 0.15, 0.55)
    male = rng.normal(0.49, 0.01, n_states)
    race_mix = {}  # state -> (white, black, hispanic) probabilities
    for i, s in enumerate(STATES):
        black = (1 - white[i]) * 0.45
        race_mix[s] = [white[i], black, (1 - white[i]) * 0.4, (1 - white[i]) * 0.15]

    # Which states have a Senate race each year (about 34 of 51)
    races = {y: set(rng.choice(STATES, 34, replace=False)) for y in YEARS}

    resp, truth = [], []
    case_id = 0
    for y in YEARS:
        for i, s in enumerate(STATES):
            n = N_PER_STATE
            race = rng.choice(
                ["White", "Black", "Hispanic", "Asian"], n, p=race_mix[s]
            )
            gender = np.where(rng.random(n) < male[i], "Male", "Female")
            educ = np.where(
                rng.random(n) < college[i],
                rng.choice(["4-Year", "Post-Grad"], n),
                rng.choice(["No HS", "High School Graduate", "Some College", "2-Year"], n),
            )
            age = rng.integers(18, 90, n)
            # Party ID: lean, race, and education shift the odds of Democrat
            z = 0.9 * lean[i] - 0.7 * (race == "White") + 0.3 * np.isin(educ, ["4-Year", "Post-Grad"])
            p_ind = 0.18
            p_dem = (1 - p_ind) * logistic(z)
            u = rng.random(n)
            pid3 = np.where(u < p_dem, "Democrat",
                            np.where(u < p_dem + (1 - p_ind) * (1 - logistic(z)), "Republican", "Independent"))  # fmt: skip
            # Vote: party ID dominates; undecided mostly among independents
            util = (2.4 * (pid3 == "Democrat") - 2.4 * (pid3 == "Republican")
                    + 0.3 * lean[i] + SWING[y] + rng.normal(0, 0.5, n))  # fmt: skip
            pd = logistic(util)
            p_und = np.where(pid3 == "Independent", 0.35, 0.06)
            r = rng.random(n)
            choice = np.where(r < p_und, "und", np.where(rng.random(n) < pd, "dem", "rep"))
            has_race = s in races[y]
            intent = np.where(
                choice == "dem", "[Democrat / Candidate 1]",
                np.where(choice == "rep", "[Republican / Candidate 2]",
                         rng.choice(["Not Sure", "No One", "[Other / Candidate 3]"], n)),  # fmt: skip
            ).astype(object)
            if not has_race:
                intent = np.full(n, None, dtype=object)
            weight = rng.lognormal(0, 0.4, n)
            resp.append(
                pl.DataFrame(
                    {
                        "year": np.full(n, y, dtype=np.int32),
                        "case_id": np.arange(case_id, case_id + n),
                        "st": s, "age": age.astype(np.int32), "gender": gender,
                        "race": race, "educ": educ, "pid3": pid3,
                        "weight": weight, "intent_sen": intent,
                    },
                    schema_overrides={"intent_sen": pl.String},
                )
            )  # fmt: skip
            case_id += n
            if has_race:
                # Population two-party Democratic share implied by the simulation
                # (many draws, so sampling noise in the CES is not in the truth)
                m = 20000
                pid = rng.choice(["D", "R", "I"], m, p=[p_dem.mean(), (1 - p_ind) - p_dem.mean(), p_ind])  # fmt: skip
                ut = (2.4 * (pid == "D") - 2.4 * (pid == "R") + 0.3 * lean[i] + SWING[y]
                      + rng.normal(0, 0.5, m))  # fmt: skip
                dem2p = logistic(ut).mean()
                total = int(rng.integers(8e5, 9e6))
                dem2p = float(np.clip(dem2p + rng.normal(0, 0.01), 0.05, 0.95))
                truth.append(
                    {"year": y, "st": s,
                     "dem_votes": int(total * 0.97 * dem2p),
                     "rep_votes": int(total * 0.97 * (1 - dem2p)),
                     "other_votes": int(total * 0.03)}  # fmt: skip
                )

    pl.concat(resp).write_parquet(out_dir / "simulated_ces.parquet")
    pl.DataFrame(truth).write_csv(out_dir / "simulated_senate_results.csv")

    # ACS-like features: the true profile plus measurement noise
    acs = []
    for y in YEARS:
        for i, s in enumerate(STATES):
            acs.append(
                {"acs_year": y, "st": s,
                 "acs_male": male[i] + rng.normal(0, 0.002),
                 "acs_white": white[i] + rng.normal(0, 0.01),
                 "acs_black": (1 - white[i]) * 0.45 + rng.normal(0, 0.01),
                 "acs_hispanic": (1 - white[i]) * 0.4 + rng.normal(0, 0.01),
                 "acs_college": college[i] + rng.normal(0, 0.01),
                 "acs_median_age": 38 + rng.normal(0, 1.5),
                 "acs_log_income": 11.1 + 0.4 * (college[i] - 0.33) / 0.05 * 0.1 + rng.normal(0, 0.08)}  # fmt: skip
            )
    pl.DataFrame(acs).write_parquet(out_dir / "simulated_acs.parquet")
    print(f"Simulated inputs written to {out_dir}")


if __name__ == "__main__":
    simulate()
