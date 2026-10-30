#### Preamble ####
# Purpose: Collect the 2026 U.S. Senate candidates (Democratic and Republican nominees,
#          incumbent status) and official portraits of those who serve or have served
#          in Congress, for the race cards and map on the website.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: Internet access. No account needed.
#
# Sources:
#   - Candidates: the infobox of each race's Wikipedia article (CC BY-SA), which names the
#     nominees and the incumbent and cites state election offices.
#   - Congress members, current and former (names, Bioguide IDs, gender): the public
#     congress-legislators project, https://unitedstates.github.io/congress-legislators/
#   - Official portraits: the Biographical Directory of the U.S. Congress
#     (bioguide.congress.gov; works of the U.S. government), with congress.gov as fallback.
#     Only people who are or were members of Congress have one: sitting senators and
#     House members running, not newcomers.
#
# Outputs (in data/01-raw_data/):
#   - candidates_2026.csv   state, race, party, name, is_incumbent, bioguide_id, gender, photo_file
#   - photos/<bioguide>.jpg  160-pixel thumbnails
#
# Run with: uv run scripts/01e-download_candidates.py

#### Workspace setup ####
import io
import re
import time
from pathlib import Path

import urllib.error
import urllib.request

import polars as pl
from PIL import Image

RAW_DIR = Path("data/01-raw_data")
PHOTO_DIR = RAW_DIR / "photos"
UA = {"User-Agent": "Mozilla/5.0 (research script; contact bolong.tang@utexas.edu)"}
WIKI = "https://en.wikipedia.org/w/index.php?title=2026_United_States_Senate_elections&action=raw"
LEGISLATORS = [
    "https://unitedstates.github.io/congress-legislators/legislators-current.csv",
    "https://unitedstates.github.io/congress-legislators/legislators-historical.csv",  # former members running again
]
THUMB = 160


#### Parse the Wikipedia race infoboxes ####
def clean(text: str) -> str:
    """Wikitext -> plain text: drop references, templates and wiki links."""
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def party_of(label: str) -> str:
    label = label.lower()
    return "Democratic" if ("democrat" in label or label.strip() == "dfl") else "Republican" if "republican" in label else "Other"


def infobox_field(box: str, key: str) -> str:
    m = re.search(rf"^\|\s*{key}\s*=(.*)$", box, flags=re.M)
    return clean(m.group(1)) if m else ""


def parse_infobox(wikitext: str, state: str, race: str) -> list[dict]:
    """Nominees (up to four) from an article's election infobox, with incumbent status."""
    i = wikitext.find("{{Infobox election")
    if i < 0:
        return []
    box = wikitext[i : wikitext.find("\n}}", i)]
    incumbent = infobox_field(box, "before_election")
    out = []
    for n in range(1, 6):
        name = infobox_field(box, f"nominee{n}") or infobox_field(box, f"candidate{n}")
        name = re.sub(r"\s*\([^)]*\)", "", name).strip()  # "Troy Jackson (replacing ...)" -> "Troy Jackson"
        if not name:
            continue
        out.append({"state": state, "race": race, "party": party_of(infobox_field(box, f"party{n}")), "name": name,
                    "is_incumbent": bool(incumbent) and same_person(name, incumbent)})  # fmt: skip
    return out


def same_person(a: str, b: str) -> bool:
    """Loose name match: same last name and compatible first name (Dan S. Sullivan ~ Dan Sullivan)."""
    pa, pb = norm(a).split(), norm(b).split()
    return bool(pa and pb) and pa[-1] == pb[-1] and pa[0][0] == pb[0][0]


def race_pages() -> list[tuple[str, str, str]]:
    """(state name, race type, article title) for the 35 races: 33 regular and the FL and OH specials."""
    regular = [n for n, a in STATE_ABBR.items() if a in REGULAR_2026]
    pages = [(n, "regular", f"2026 United States Senate election in {n}") for n in regular]
    return pages + [(n, "special", f"2026 United States Senate special election in {n}") for n in ("Florida", "Ohio")]


REGULAR_2026 = {
    "AL", "AK", "AR", "CO", "DE", "GA", "ID", "IL", "IA", "KS", "KY", "LA", "ME", "MA", "MI", "MN", "MS",
    "MT", "NE", "NH", "NJ", "NM", "NC", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
}  # fmt: skip


#### Match to Congress members and portraits ####
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip


def norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).strip()


def match_legislator(name: str, state: str, leg: pl.DataFrame) -> dict | None:
    """A member from the same state whose last name agrees and whose first name is compatible:
    equal, a prefix, or a nickname. If only one member of the state has that last name, a
    matching first initial is enough ('Jim' ~ 'James')."""
    parts = norm(name).split()
    if len(parts) < 2:
        return None
    last = parts[-1] if parts[-1] not in {"jr", "sr", "ii", "iii"} else parts[-2]
    same_last = [r for r in leg.filter(pl.col("state") == state).iter_rows(named=True) if norm(r["last_name"]).split()[-1] == last]
    for r in same_last:
        first = norm(r["first_name"]).split()[0]
        if first == parts[0] or norm(r.get("nickname") or "x") == parts[0] or first.startswith(parts[0]) or parts[0].startswith(first):
            return r
    if len(same_last) == 1 and norm(same_last[0]["first_name"])[:1] == parts[0][:1]:
        return same_last[0]
    return None


def fetch_photo(bioguide: str) -> str | None:
    """Download an official portrait and save a small thumbnail; None if unavailable."""
    dest = PHOTO_DIR / f"{bioguide}.jpg"
    if dest.exists():
        return dest.name
    for url in (f"https://bioguide.congress.gov/bioguide/photo/{bioguide[0]}/{bioguide}.jpg",
                f"https://www.congress.gov/img/member/{bioguide.lower()}_200.jpg"):  # fmt: skip
        try:
            body, ctype = get(url, tries=2)
            if ctype.startswith("image"):
                img = Image.open(io.BytesIO(body)).convert("RGB")
                img.thumbnail((THUMB, THUMB * 5 // 4))
                PHOTO_DIR.mkdir(parents=True, exist_ok=True)
                img.save(dest, quality=88)
                return dest.name
        except Exception:
            pass
        time.sleep(0.5)
    return None


def get(url: str, tries: int = 5) -> tuple[bytes, str]:
    """(body, content-type). urllib rather than httpx: Wikimedia refuses httpx's requests.
    Wikimedia also rate limits shared addresses (HTTP 429), so back off and retry."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read(), r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or i == tries - 1:
                raise
            time.sleep(10 * (i + 1))
    raise RuntimeError(url)


def fetch_text(url: str) -> str:
    return get(url)[0].decode("utf-8")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    leg = pl.concat([pl.read_csv(io.StringIO(fetch_text(u)), infer_schema_length=0).select("last_name", "first_name", "nickname", "gender", "bioguide_id", "state") for u in LEGISLATORS]).unique("bioguide_id")
    cands = []
    for state, race, title in race_pages():
        cands += parse_infobox(fetch_text("https://en.wikipedia.org/w/index.php?title=" + title.replace(" ", "_") + "&action=raw"), state, race)
        time.sleep(3)  # be gentle: Wikimedia rate limits shared addresses
    rows = []
    for r in cands:
        m = match_legislator(r["name"], STATE_ABBR[r["state"]], leg) if r["party"] != "Other" else None
        bio = m["bioguide_id"] if m else None
        rows.append({**r, "st": STATE_ABBR[r["state"]], "bioguide_id": bio, "gender": m["gender"] if m else None,
                     "photo_file": fetch_photo(bio) if bio else None})  # fmt: skip
    out = pl.DataFrame(rows).select("st", "state", "race", "party", "name", "is_incumbent", "bioguide_id", "gender", "photo_file").sort("st", "race", "party", "name")
    out.write_csv(RAW_DIR / "candidates_2026.csv")
    major = out.filter(pl.col("party").is_in(["Democratic", "Republican"]))
    print(f"{out.height} candidates in {out['st'].n_unique()} states; {out['photo_file'].is_not_null().sum()} with official portraits")
    counts = major.group_by("st", "party").len().pivot(on="party", index="st", values="len").fill_null(0)
    print("races without exactly one nominee per major party:", counts.filter((pl.col("Democratic") != 1) | (pl.col("Republican") != 1)).rows())
