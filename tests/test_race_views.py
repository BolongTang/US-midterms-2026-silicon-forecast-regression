#### Preamble ####
# Purpose: Test the candidate parser, the portrait matcher, the race cards and the winner map.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: None (the geometry test skips if data/01-raw_data/us-states.geojson is absent)

#### Workspace setup ####
import importlib.util
import re
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cand = load("cand", "01e-download_candidates.py")
views = load("views", "07-race_views.py")

INFOBOX = """{{Infobox election
| election_name = 2026 United States Senate election in Texas
| nominee1 = [[Ken Paxton]]
| party1 = Republican Party (United States)
| nominee2 = [[James Talarico]] <ref>cite</ref>
| party2 = Democratic Party (United States)
| nominee3 = Troy Jackson (replacing Graham Platner)
| party3 = Democratic-Farmer-Labor
| nominee4 = Todd Achilles
| party4 = Independent politician
| before_election = [[Dan Sullivan (U.S. senator)|Dan S. Sullivan]]
}}
text"""


#### Candidate parsing ####
def test_infobox_parsing_cleans_names_and_maps_parties():
    out = cand.parse_infobox(INFOBOX, "Texas", "regular")
    assert [(r["name"], r["party"]) for r in out] == [
        ("Ken Paxton", "Republican"), ("James Talarico", "Democratic"),
        ("Troy Jackson", "Democratic"), ("Todd Achilles", "Other")]  # fmt: skip


def test_incumbent_matches_across_middle_initials():
    assert cand.same_person("Dan S. Sullivan", "Dan Sullivan")
    assert not cand.same_person("Darline Graham", "Lindsey Graham")
    out = cand.parse_infobox(INFOBOX.replace("[[Ken Paxton]]", "Dan Sullivan"), "Alaska", "regular")
    assert out[0]["is_incumbent"] and not out[1]["is_incumbent"]


def test_legislator_matching_uses_state_and_nickname():
    leg = pl.DataFrame({"last_name": ["Risch", "Smith", "Smith"], "first_name": ["James", "Adam", "Christopher"],
                        "nickname": [None, None, "Chris"], "gender": ["M"] * 3, "bioguide_id": ["R1", "S1", "S2"],
                        "state": ["ID", "WA", "WA"]})  # fmt: skip
    assert cand.match_legislator("Jim Risch", "ID", leg)["bioguide_id"] == "R1"  # unique last name: initial is enough
    assert cand.match_legislator("Chris Smith", "WA", leg)["bioguide_id"] == "S2"  # two Smiths: nickname decides
    assert cand.match_legislator("Jim Risch", "WA", leg) is None  # other state


#### Real candidate file ####
def test_real_candidates_cover_all_35_races():
    path = ROOT / "data" / "01-raw_data" / "candidates_2026.csv"
    if not path.exists():
        pytest.skip("candidates_2026.csv not downloaded")
    d = pl.read_csv(path)
    assert d["st"].n_unique() == 35
    major = d.filter(pl.col("party").is_in(["Democratic", "Republican"]))
    assert (major.filter(pl.col("party") == "Republican").group_by("st").len()["len"] >= 1).all()
    assert set(d["race"]) == {"regular", "special"} and d.filter(pl.col("race") == "special")["st"].n_unique() == 2


#### Cards and map ####
def fixture_races():
    head = pl.DataFrame({"st": ["TX", "CO", "ID"], "p_dem": [0.30, 0.52, 0.28], "p_rep": [0.55, 0.30, 0.60], "p_und": [0.15, 0.18, 0.12]})
    cands = pl.DataFrame({
        "st": ["TX", "TX", "CO", "CO", "ID", "ID"], "party": ["Republican", "Democratic", "Republican", "Democratic", "Republican", "Other"],
        "name": ["Ken Paxton", "James Talarico", "Mark Baisley", "John Hickenlooper", "Jim Risch", "Todd Achilles"],
        "is_incumbent": [False, False, False, True, True, False], "photo_file": [None] * 6})  # fmt: skip
    return views.build_races(head, cands)


def test_winners_follow_the_predicted_party_and_missing_nominees_are_marked():
    r = {x["st"]: x for x in fixture_races()}
    assert (r["TX"]["winner_party"], r["TX"]["winner_name"]) == ("Republican", "Ken Paxton")
    assert (r["CO"]["winner_party"], r["CO"]["winner_name"]) == ("Democratic", "John Hickenlooper")
    assert r["ID"]["dem"] is None and r["ID"]["winner_name"] == "Jim Risch"
    assert r["TX"]["dem2p"] == pytest.approx(0.30 / 0.85)


def test_cards_show_bars_three_per_row_and_bigger_winner():
    html = views.cards_html(fixture_races())
    assert "grid-template-columns:repeat(3" in html  # three cards per line
    assert html.count('class="race-card"') == 3
    for card in html.split('class="race-card"')[1:]:
        widths = [float(w) for w in re.findall(r'style="width:([\d.]+)%"', card)]
        assert sum(widths) == pytest.approx(100, abs=0.02)  # blue + red fill the bar
        assert card.count("person win") == 1  # exactly one larger (winner) profile
        assert card.count("badge big") + card.count("pic big") == 1
    assert "No Democratic nominee" in html  # Idaho
    assert "Hickenlooper" in html and "Incumbent" in html


def test_winner_map_colours_by_party_and_grey_for_no_race():
    if not (ROOT / "data" / "01-raw_data" / "us-states.geojson").exists():
        pytest.skip("us-states.geojson not downloaded")
    fig = views.winner_map(fixture_races())
    ax = fig.axes[0]
    from matplotlib.colors import to_hex
    from matplotlib.patches import Polygon

    fills = [to_hex(p.get_facecolor()) for p in ax.patches if isinstance(p, Polygon)]  # state shapes only
    assert {views.BLUE, views.RED, views.GREY} <= set(fills)
    texts = {t.get_text() for t in ax.texts}
    assert {"Paxton", "Hickenlooper", "Risch"} <= texts  # winner's surname printed in the state
    assert views.last_name("Marquita Bradshaw") == "Bradshaw" and views.last_name("Martin Luther King Jr.") == "King"
