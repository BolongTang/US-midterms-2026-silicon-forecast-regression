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


def row(year, st, party, votes, special=False, stage="gen", candidate="X", detailed=None):
    return {"year": year, "state_po": st, "party_simplified": party, "party_detailed": detailed or party,
            "candidate": candidate, "candidatevotes": votes, "special": special, "stage": stage}  # fmt: skip


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


def test_caucusing_independents_are_coded_by_party_but_others_are_not():
    raw = pl.DataFrame([
        # Maine: King (independent, caucuses with Democrats) v. a Republican
        row(2018, "ME", "OTHER", 60, candidate="ANGUS S. KING, JR.", detailed="INDEPENDENT"),
        row(2018, "ME", "REPUBLICAN", 40),
        # Nebraska: Osborn does not caucus with a party; no Democrat in this race
        row(2024, "NE", "OTHER", 46, candidate="DAN OSBORN", detailed="BY PETITION"),
        row(2024, "NE", "REPUBLICAN", 54),
        # an independent named King in another state is not recoded
        row(2018, "ZZ", "OTHER", 30, candidate="JOHN KING", detailed="INDEPENDENT"), row(2018, "ZZ", "REPUBLICAN", 70),
    ])  # fmt: skip
    out = res.build_results(raw, [2018, 2024]).sort("st")
    assert out["st"].to_list() == ["ME"]  # NE and ZZ have no Democrat, so no race
    assert out["dem_votes"].to_list() == [60]  # King counted as a Democrat


def test_special_and_regular_races_are_pooled_alike():
    raw = pl.DataFrame([
        row(2020, "GA", "DEMOCRAT", 10), row(2020, "GA", "REPUBLICAN", 12),
        row(2020, "GA", "DEMOCRAT", 7, special=True), row(2020, "GA", "REPUBLICAN", 9, special=True),
        # regular race has no Democrat (Nebraska 2024 pattern); the special race does
        row(2024, "NE", "REPUBLICAN", 50), row(2024, "NE", "OTHER", 40, detailed="BY PETITION"),
        row(2024, "NE", "REPUBLICAN", 58, special=True), row(2024, "NE", "DEMOCRAT", 35, special=True),
    ])  # fmt: skip
    out = res.build_results(raw, [2020, 2024]).sort("year")
    assert out["dem_votes"].to_list() == [17, 35]  # Georgia pools both races; Nebraska keeps the one with a Democrat
    assert out["rep_votes"].to_list() == [21, 58]
    assert out["n_races"].to_list() == [2, 1]


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
