#### Preamble ####
# Purpose: Test how certified Senate returns are reduced to one Dem-v-Rep race per
#          state and year, and that the real file (if present) fits the regression.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: None (real-file checks skip if senate_results.csv is absent)

#### Workspace setup ####
import importlib.util
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("res", ROOT / "scripts" / "01c-download_results.py")
res = importlib.util.module_from_spec(spec)
spec.loader.exec_module(res)


def row(year, st, party, votes, special=False, stage="gen"):
    return {"year": year, "state_po": st, "party_simplified": party, "candidatevotes": votes,
            "special": special, "stage": stage}  # fmt: skip


def test_builds_one_two_party_race_per_state_year():
    raw = pl.DataFrame([
        row(2024, "AA", "DEMOCRAT", 60), row(2024, "AA", "REPUBLICAN", 40), row(2024, "AA", "OTHER", 5),
        # two Democrats and no Republican: dropped
        row(2024, "BB", "DEMOCRAT", 60), row(2024, "BB", "DEMOCRAT", 40),
        # runoff: excluded
        row(2024, "CC", "DEMOCRAT", 1, stage="runoff"), row(2024, "CC", "REPUBLICAN", 1, stage="runoff"),
    ])  # fmt: skip
    out = res.build_results(raw, [2024])
    assert out["st"].to_list() == ["AA"]
    assert out.row(0, named=True)["other_votes"] == 5


def test_prefers_regular_race_but_falls_back_to_special():
    raw = pl.DataFrame([
        row(2020, "GA", "DEMOCRAT", 10), row(2020, "GA", "REPUBLICAN", 12),
        row(2020, "GA", "DEMOCRAT", 7, special=True), row(2020, "GA", "REPUBLICAN", 9, special=True),
        # regular race has no Democrat (Nebraska 2024 pattern); special does
        row(2024, "NE", "REPUBLICAN", 50), row(2024, "NE", "OTHER", 40),
        row(2024, "NE", "REPUBLICAN", 58, special=True), row(2024, "NE", "DEMOCRAT", 35, special=True),
    ])  # fmt: skip
    out = res.build_results(raw, [2020, 2024]).sort("year")
    assert out["dem_votes"].to_list() == [10, 35]
    assert out["special"].to_list() == [False, True]


def test_real_results_file_is_usable():
    path = ROOT / "data" / "01-raw_data" / "senate_results.csv"
    if not path.exists():
        pytest.skip("senate_results.csv not downloaded")
    reg_spec = importlib.util.spec_from_file_location("reg", ROOT / "scripts" / "05-regression.py")
    reg = importlib.util.module_from_spec(reg_spec)
    reg_spec.loader.exec_module(reg)
    off = reg.load_official(path)
    assert off["official_dem2p"].is_between(0.05, 0.95).all()
    assert off.group_by("year", "st").len()["len"].max() == 1
