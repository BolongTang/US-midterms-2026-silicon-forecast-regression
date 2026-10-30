#### Preamble ####
# Purpose: Build the race-by-race views for the website: (1) a card per race with a
#          red-blue bar, nominee names and portraits, the predicted winner shown larger;
#          (2) a USA map coloured by predicted winner (blue Democratic, red Republican,
#          grey no race) with the winner's name in each state.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: scripts/05-regression.py outputs; data/01-raw_data/candidates_2026.csv
#   (scripts/01e-download_candidates.py); data/01-raw_data/us-states.geojson (downloaded
#   here on first use from the PublicaMundi MappingAPI repository).
#
# Official portraits exist only for people who are or were members of Congress; every
# other nominee gets an initials badge. Colours are the validated blue/red pair; labels
# on the fills use dark ink (white on the red fill is only 3.95:1).
#
# Run with: uv run scripts/07-race_views.py [--source real|simulated]

#### Workspace setup ####
import argparse
import base64
import html
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLUE, RED, GREY, INK, INK2 = "#2a78d6", "#e34948", "#c9c8c4", "#0b0b0b", "#52514e"
GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip
ABBR_NAME = {v: k for k, v in STATE_ABBR.items()}


#### Join forecast and candidates ####
def last_name(name: str) -> str:
    parts = [p for p in name.replace('"', "").split() if p.lower().strip(".") not in SUFFIXES]
    return parts[-1] if parts else name


def initials(name: str) -> str:
    parts = [p for p in name.replace('"', "").split() if p[:1].isalpha()]
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else name[:2].upper()


def load_candidates(path: Path = ROOT / "data" / "01-raw_data" / "candidates_2026.csv") -> pl.DataFrame:
    return pl.read_csv(path)


def build_races(head: pl.DataFrame, cands: pl.DataFrame) -> list[dict]:
    """One dict per race: shares, nominees, portraits and the predicted winner.
    `head` has st, p_dem, p_rep, p_und (the headline model's three-way forecast)."""
    races = []
    for r in head.sort("st").iter_rows(named=True):
        c = cands.filter(pl.col("st") == r["st"])
        pick = lambda party: next(iter(c.filter(pl.col("party") == party).sort("is_incumbent", descending=True).iter_rows(named=True)), None)  # noqa: E731
        dem, rep = pick("Democratic"), pick("Republican")
        others = c.filter(pl.col("party") == "Other")["name"].to_list()
        dem2p = r["p_dem"] / (r["p_dem"] + r["p_rep"])
        winner_party = "Democratic" if dem2p > 0.5 else "Republican"
        win = dem if winner_party == "Democratic" else rep
        races.append({
            "st": r["st"], "state": ABBR_NAME[r["st"]], "dem": dem, "rep": rep, "others": others,
            "dem2p": dem2p, "p_dem": r["p_dem"], "p_rep": r["p_rep"], "p_und": r["p_und"],
            "winner_party": winner_party, "winner": win,
            "winner_name": win["name"] if win else f"No {winner_party} nominee",
        })  # fmt: skip
    return races


#### Race cards ####
def photo_uri(photo_file: str | None, photo_dir: Path) -> str | None:
    if not photo_file or not (photo_dir / photo_file).exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode((photo_dir / photo_file).read_bytes()).decode()


CSS = """
<style>
.race-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0}
@media (max-width:900px){.race-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:560px){.race-grid{grid-template-columns:minmax(0,1fr)}}
.race-card{border:1px solid #d9d8d3;border-radius:10px;padding:10px 12px 12px;background:#fcfcfb;color:#0b0b0b;font-size:13px;line-height:1.25}
.race-card h4{margin:0 0 6px;font-size:15px;font-weight:700;line-height:1.2}
.race-card h4 small{display:block;font-weight:400;color:#52514e;font-size:11px;margin-top:2px}
.people{display:flex;justify-content:space-between;align-items:flex-end;gap:8px;min-height:104px}
.person{width:50%;text-align:center}
.person .nm{margin-top:4px;font-size:12px}
.person.win .nm{font-weight:700;font-size:13px}
.person .tag{display:block;font-size:10px;color:#52514e}
.person.win .tag{color:#0b0b0b;font-weight:700}
.pic{display:inline-block;border-radius:8px;object-fit:cover;background:#ecebe7}
.pic.big{width:78px;height:97px;border:3px solid var(--c)}
.pic.small{width:44px;height:55px;border:2px solid var(--c);opacity:.9}
.badge{display:inline-flex;align-items:center;justify-content:center;border-radius:8px;font-weight:700;color:#0b0b0b;background:var(--c);opacity:.9}
.badge.big{width:78px;height:97px;font-size:26px}
.badge.small{width:44px;height:55px;font-size:15px}
.bar{display:flex;height:24px;border-radius:6px;overflow:hidden;margin-top:8px;font-weight:700;font-size:12px}
.bar span{display:flex;align-items:center;padding:0 7px;color:#0b0b0b;white-space:nowrap}
.bar .d{background:#2a78d6}.bar .r{background:#e34948;justify-content:flex-end}
.bar .gap{width:2px;background:#fcfcfb}
.meta{margin-top:5px;font-size:11px;color:#52514e;display:flex;justify-content:space-between;gap:6px}
</style>
"""


def person_html(c: dict | None, party: str, big: bool, photo_dir: Path, label: str) -> str:
    colour = BLUE if party == "Democratic" else RED
    size = "big" if big else "small"
    if c is None:
        pic = f'<span class="badge {size}" style="--c:{GREY}">?</span>'
        name, tag = f"No {party} nominee", label
    else:
        uri = photo_uri(c.get("photo_file"), photo_dir)
        pic = (f'<img class="pic {size}" style="--c:{colour}" src="{uri}" alt="Official portrait of {html.escape(c["name"])}">' if uri
               else f'<span class="badge {size}" style="--c:{colour}" role="img" aria-label="{html.escape(c["name"])}">{html.escape(initials(c["name"]))}</span>')  # fmt: skip
        name, tag = html.escape(c["name"]), ("Incumbent · " if c.get("is_incumbent") else "") + label
    return f'<div class="person{" win" if big else ""}">{pic}<div class="nm">{name}</div><span class="tag">{html.escape(tag)}</span></div>'


def cards_html(races: list[dict], photo_dir: Path = ROOT / "data" / "01-raw_data" / "photos") -> str:
    """A grid of race cards, three to a line."""
    out = [CSS, '<div class="race-grid">']
    for r in races:
        d_win = r["winner_party"] == "Democratic"
        dem_html = person_html(r["dem"], "Democratic", d_win, photo_dir, "Democratic")
        rep_html = person_html(r["rep"], "Republican", not d_win, photo_dir, "Republican")
        d, rr = 100 * r["dem2p"], 100 * (1 - r["dem2p"])
        other = f" · independent: {html.escape(', '.join(r['others'][:2]))}" if r["others"] and (r["dem"] is None or r["rep"] is None) else ""
        out.append(
            f'<div class="race-card" aria-label="{html.escape(r["state"])} Senate race">'
            f'<h4>{html.escape(r["state"])}<small>Predicted winner: {html.escape(r["winner_name"])} ({"D" if d_win else "R"})</small></h4>'
            f'<div class="people">{dem_html}{rep_html}</div>'
            f'<div class="bar" role="img" aria-label="Democratic {d:.1f} percent, Republican {rr:.1f} percent of the two-party vote">'
            f'<span class="d" style="width:{d:.2f}%">D {d:.1f}%</span><span class="gap"></span><span class="r" style="width:{rr:.2f}%">R {rr:.1f}%</span></div>'
            f'<div class="meta"><span>Including undecided/other: D {100*r["p_dem"]:.0f}% · R {100*r["p_rep"]:.0f}% · U {100*r["p_und"]:.0f}%</span></div>'
            + (f'<div class="meta"><span>{other[3:]}</span></div>' if other else "")
            + "</div>"
        )
    out.append("</div>")
    return "\n".join(out)


#### Map ####
def ensure_geojson(path: Path = ROOT / "data" / "01-raw_data" / "us-states.geojson") -> dict:
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(urllib.request.Request(GEOJSON_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
    return json.loads(path.read_text())


def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5):
    """Spherical Albers equal-area conic projection (unit sphere)."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    l0, f0, f1, f2 = np.radians(lon0), np.radians(lat0), np.radians(p1), np.radians(p2)
    n = (np.sin(f1) + np.sin(f2)) / 2
    c = np.cos(f1) ** 2 + 2 * n * np.sin(f1)
    rho0 = np.sqrt(c - 2 * n * np.sin(f0)) / n
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    th = n * (lon - l0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


REGIONS = {  # state -> (projection parameters, scale, anchor as fractions of the lower-48 box)
    "Alaska": (dict(lon0=-154, lat0=50, p1=55, p2=65), 0.35, (0.10, -0.10)),
    "Hawaii": (dict(lon0=-157, lat0=3, p1=8, p2=18), 1.0, (0.27, -0.12)),
}
# Small states labelled in the Atlantic with a leader line: abbr -> (dx, dy) as fractions of the box
OUTSIDE = {"NH": (0.075, 0.05), "MA": (0.095, 0.0), "RI": (0.085, -0.04), "DE": (0.075, -0.015), "NJ": (0.07, -0.025)}


def rings(geom: dict) -> list[np.ndarray]:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    out = [np.array(p[0]) for p in polys]  # outer ring of each polygon
    return [r for r in out if r[:, 0].max() < 0]  # drop Aleutian rings across the antimeridian (they smear the map)


def project_states(geo: dict) -> dict[str, list[np.ndarray]]:
    """Projected outer rings per state, with Alaska and Hawaii as insets below the lower 48."""
    proj = {}
    for f in geo["features"]:
        name = f["properties"]["name"]
        if name == "Puerto Rico":
            continue
        params, scale, _ = REGIONS.get(name, (dict(), 1.0, None))
        proj[name] = [np.column_stack([a * scale, b * scale]) for a, b in (albers(r[:, 0], r[:, 1], **params) for r in rings(f["geometry"]))]
    lower = np.vstack([r for n, rs in proj.items() if n not in REGIONS for r in rs])
    (x0, y0), (x1, y1) = lower.min(0), lower.max(0)
    w, h = x1 - x0, y1 - y0
    for name, (_, _, (fx, fy)) in REGIONS.items():
        pts = np.vstack(proj[name])
        c = (pts.min(0) + pts.max(0)) / 2
        target = np.array([x0 + fx * w, y0 + fy * h + 0.05 * h])
        proj[name] = [r - c + target for r in proj[name]]
    return proj


def centroid(ring: np.ndarray) -> np.ndarray:
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    a = cross.sum() / 2
    return np.array([((x + x2) * cross).sum(), ((y + y2) * cross).sum()]) / (6 * a)


def winner_map(races: list[dict], geo: dict | None = None):
    """Matplotlib figure: red Republican win, blue Democratic win, grey no race; winner's name in each state."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    geo = geo or ensure_geojson()
    proj = project_states(geo)
    by_name = {r["state"]: r for r in races}
    allpts = np.vstack([r for n, rs in proj.items() for r in rs])
    (x0, y0), (x1, y1) = allpts.min(0), allpts.max(0)
    w, h = x1 - x0, y1 - y0
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=110)
    for name, rs in proj.items():
        race = by_name.get(name)
        fill = GREY if race is None else (BLUE if race["winner_party"] == "Democratic" else RED)
        for ring in rs:
            ax.add_patch(Polygon(ring, closed=True, facecolor=fill, edgecolor="#fcfcfb", linewidth=0.8))
    for name, race in by_name.items():
        if name not in proj:
            continue
        ab = STATE_ABBR[name]
        big = max(proj[name], key=lambda r: abs(np.ptp(r[:, 0]) * np.ptp(r[:, 1])))
        c = centroid(big)
        label = last_name(race["winner_name"]) if race["winner"] else "—"
        if ab in OUTSIDE:
            dx, dy = OUTSIDE[ab]
            tx, ty = c[0] + dx * w, c[1] + dy * h
            ax.plot([c[0], tx - 0.004 * w], [c[1], ty], color=INK2, linewidth=0.7, zorder=3)
            ax.text(tx, ty, label, ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold", zorder=4)
        else:
            ax.text(c[0], c[1], label, ha="center", va="center", fontsize=9.5 if len(label) < 8 else 8 if len(label) < 10 else 7, color=INK, fontweight="bold", zorder=4)
    ax.set_xlim(x0 - 0.01 * w, x1 + 0.10 * w)
    ax.set_ylim(y0 - 0.03 * h, y1 + 0.10 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, (txt, col) in enumerate([("Predicted Democratic win", BLUE), ("Predicted Republican win", RED), ("No Senate race in 2026", GREY)]):
        ax.add_patch(plt.Rectangle((x0 + (0.02 + 0.27 * i) * w, y1 + 0.045 * h), 0.022 * w, 0.03 * h, facecolor=col, edgecolor="none"))
        ax.text(x0 + (0.048 + 0.27 * i) * w, y1 + 0.06 * h, txt, va="center", fontsize=10.5, color=INK)
    ax.set_title("2026 U.S. Senate: predicted winner in each state", loc="left", fontsize=13, fontweight="bold", color=INK, pad=6)
    fig.tight_layout()
    return fig


def figure_html(fig, alt: str, dpi: int = 150) -> str:
    """A matplotlib figure as a self-contained <img> that scales to the page width."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return f'<img src="{uri}" alt="{html.escape(alt)}" style="max-width:100%;height:auto">'


#### Run ####
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "simulated"], default="real")
    args = ap.parse_args()
    figs_spec = __import__("importlib.util").util.spec_from_file_location("figs", HERE / "06-make_figures.py")
    figs = __import__("importlib.util").util.module_from_spec(figs_spec)
    figs_spec.loader.exec_module(figs)
    res = figs.load(args.source, ROOT)
    head = res["preds"].filter(pl.col("train") == res["headline_train"], pl.col("features") == res["oos_features"])
    races = build_races(head, load_candidates())
    out = ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    winner_map(races).savefig(out / f"{args.source}_winner_map.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    (out / f"{args.source}_race_cards.html").write_text(cards_html(races))
    print(f"saved map and cards for {len(races)} races to {out}")
