#### Preamble ####
# Purpose: Build the figures for the regression forecast (used by
#          paper/regression.qmd) and save them as PNGs in paper/figures/.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: scripts/05-regression.py and its inputs (real or simulated)
#
# Colours: Democratic blue and Republican red pass the data-viz palette checks
# (all-pairs CVD and normal-vision separation); Undecided is a neutral grey and
# is labelled directly. Model comparisons use three validated categorical hues
# plus marker shape, so colour is never the only cue.
#
# Run with: uv run scripts/06-make_figures.py [--source real|simulated]

#### Workspace setup ####
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from mizani.bounds import squish
from plotnine import (
    aes, coord_flip, element_blank, element_line, element_text, facet_grid,
    geom_abline, geom_col, geom_point, geom_text, geom_tile, geom_vline, ggplot,
    labs, scale_color_manual, scale_fill_gradient2, scale_fill_manual,
    scale_shape_manual, scale_x_continuous, scale_y_continuous, theme,
    theme_minimal,
)  # fmt: skip

HERE = Path(__file__).resolve().parent
BLUE, RED, GREY = "#2a78d6", "#e34948", "#9a9893"
MODEL_COLOURS = {"2022": "#2a78d6", "2024": "#eb6834", "2022+2024": "#1baf7a"}
MODEL_SHAPES = {"2022": "o", "2024": "s", "2022+2024": "^"}
INK, INK2, GRID, MID = "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"

# Standard US tile-grid layout: state -> (row, column)
TILES = {
    "AK": (0, 0), "ME": (0, 10),
    "VT": (1, 9), "NH": (1, 10),
    "WA": (2, 0), "ID": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "IL": (2, 5),
    "WI": (2, 6), "MI": (2, 7), "NY": (2, 8), "RI": (2, 9), "MA": (2, 10),
    "OR": (3, 0), "NV": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4), "IN": (3, 5),
    "OH": (3, 6), "PA": (3, 7), "NJ": (3, 8), "CT": (3, 9),
    "CA": (4, 0), "UT": (4, 1), "CO": (4, 2), "NE": (4, 3), "MO": (4, 4), "KY": (4, 5),
    "WV": (4, 6), "VA": (4, 7), "MD": (4, 8), "DE": (4, 9),
    "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4), "TN": (5, 5), "NC": (5, 6),
    "SC": (5, 7), "DC": (5, 8),
    "OK": (6, 3), "LA": (6, 4), "MS": (6, 5), "AL": (6, 6), "GA": (6, 7),
    "HI": (7, 0), "TX": (7, 3), "FL": (7, 8),
}  # fmt: skip


def load_regression_module():
    spec = importlib.util.spec_from_file_location("reg", HERE / "05-regression.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#### Data ####
def load(source: str = "simulated", root: Path = Path(".")) -> dict:
    """Run the regression and collect everything the figures need."""
    reg = load_regression_module()
    reg.ROOT = Path(root)
    metrics, preds = reg.run(source)

    # Out-of-sample (leave-one-state-out) predictions for the widest training set
    p = reg.paths(source)
    ces = pl.concat([pl.read_parquet(f) for f in p["ces"] if f.exists()], how="diagonal_relaxed")
    acs, official = reg.load_acs(p["acs"]), reg.load_official(p["results"])
    table = reg.build_training_table(ces, acs, official)
    sets = reg.train_sets(sorted(table["year"].unique().to_list()))
    label = "all" if "all" in sets else "2022+2024"
    fs = "ces+acs" if acs is not None else "ces"
    features = reg.CES_FEATURES + (reg.ACS_FEATURES if acs is not None else []) + ["midterm"]
    train = table.filter(pl.col("year").is_in(sets[label])).drop_nulls(features)
    pred = reg.leave_one_state_out(train, features)
    oos = train.select("st", "year", "dem", "rep", *(["official_dem2p"] if official is not None else [])).with_columns(
        pred_dem=pl.Series(pred[:, 0]),
        pred_rep=pl.Series(pred[:, 1]),
    ).with_columns(
        pred_dem2p=pl.col("pred_dem") / (pl.col("pred_dem") + pl.col("pred_rep")),
        obs_dem2p=pl.col("dem") / (pl.col("dem") + pl.col("rep")),
    )  # fmt: skip
    return {"metrics": metrics, "preds": preds, "oos": oos, "source": source,
            "oos_label": label, "oos_features": fs, "has_official": official is not None}  # fmt: skip


def pick(preds: pl.DataFrame, train: str, features: str | None = None) -> pd.DataFrame:
    """One model's forecast with the two-party Democratic share added."""
    features = features or sorted(preds["features"].unique().to_list())[-1]
    out = preds.filter(pl.col("train") == train, pl.col("features") == features)
    return out.with_columns(
        dem2p=pl.col("p_dem") / (pl.col("p_dem") + pl.col("p_rep"))
    ).to_pandas()


#### Theme ####
def base_theme(**kw):
    args = dict(
        text=element_text(color=INK),
        axis_text=element_text(color=INK2),
        panel_grid_major=element_line(color=GRID, size=0.5),
        panel_grid_minor=element_blank(),
        plot_title=element_text(size=13, weight="bold"),
        plot_subtitle=element_text(color=INK2),
        legend_position="top",
        legend_title=element_blank(),
    )
    args.update(kw)  # callers override defaults
    return theme_minimal() + theme(**args)


#### Figures ####
def fig_forecast(preds: pl.DataFrame, train: str = "2022+2024", features: str | None = None):
    """Stacked bars: predicted Dem / Rep / Undecided share in every 2026 race."""
    d = pick(preds, train, features)
    order = d.sort_values("dem2p")["st"].tolist()
    long = d.melt(id_vars="st", value_vars=["p_rep", "p_und", "p_dem"], var_name="party", value_name="share")
    names = {"p_dem": "Democratic", "p_rep": "Republican", "p_und": "Undecided / other"}
    long["party"] = pd.Categorical(long["party"].map(names), categories=["Republican", "Undecided / other", "Democratic"], ordered=True)
    long["st"] = pd.Categorical(long["st"], categories=order, ordered=True)
    return (
        ggplot(long, aes("st", "share", fill="party"))
        + geom_col(width=0.8, color="#fcfcfb", size=0.6)
        + coord_flip()
        + scale_fill_manual({"Democratic": BLUE, "Republican": RED, "Undecided / other": GREY})
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v], expand=(0, 0))
        + labs(title="Predicted Senate vote share, every 2026 race",
               subtitle=f"Model trained on {train}; states sorted by Democratic two-party share",
               x=None, y="Share of respondents")
        + base_theme(figure_size=(7, 9), axis_title_y=element_blank())
    )  # fmt: skip


def fig_models(preds: pl.DataFrame, features: str | None = None):
    """Dem two-party share by state under the three requested training sets."""
    parts = [pick(preds, t, features).assign(model=t) for t in MODEL_COLOURS if t in preds["train"].unique().to_list()]
    d = pd.concat(parts)
    order = d[d["model"] == parts[-1]["model"].iloc[0]].sort_values("dem2p")["st"].tolist()
    d["st"] = pd.Categorical(d["st"], categories=order, ordered=True)
    d["model"] = pd.Categorical(d["model"], categories=list(MODEL_COLOURS), ordered=True)
    return (
        ggplot(d, aes("dem2p", "st", color="model", shape="model"))
        + geom_vline(xintercept=0.5, color=GREY, linetype="dashed")
        + geom_point(size=2.6, stroke=0.4, alpha=0.9)
        + scale_color_manual(MODEL_COLOURS)
        + scale_shape_manual(MODEL_SHAPES)
        + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="Does the training set matter?",
               subtitle="Predicted Democratic share of the two-party vote, by training years",
               x="Democratic two-party share", y=None)
        + base_theme(figure_size=(7, 9), axis_title_y=element_blank())
    )  # fmt: skip


def fig_tilemap(preds: pl.DataFrame, train: str = "2022+2024", features: str | None = None):
    """Tile map of the predicted Dem two-party share; states without a 2026 race are blank."""
    d = pick(preds, train, features).set_index("st")["dem2p"].to_dict()
    rows = [{"st": s, "x": c, "y": -r, "dem2p": d.get(s, np.nan), "race": s in d} for s, (r, c) in TILES.items()]
    t = pd.DataFrame(rows)
    return (
        ggplot(t, aes("x", "y"))
        + geom_tile(aes(fill="dem2p"), width=0.94, height=0.94, color="#fcfcfb", data=t[t["race"]])
        + geom_tile(width=0.94, height=0.94, fill="#f5f4f1", color="#fcfcfb", data=t[~t["race"]])
        + geom_text(aes(label="st"), size=8, color=INK)
        + scale_fill_gradient2(low=RED, mid=MID, high=BLUE, midpoint=0.5, limits=(0.2, 0.8), oob=squish,
                               breaks=[0.2, 0.5, 0.8], labels=lambda v: [f"{x:.0%}" for x in v], name="Dem two-party share")  # fmt: skip
        + labs(title="Predicted Democratic share of the two-party vote",
               subtitle=f"2026 Senate races (model trained on {train}); grey = no race; shades saturate beyond 20%/80%", x=None, y=None)
        + base_theme(figure_size=(8, 5.2), axis_text=element_blank(), axis_title=element_blank(), panel_grid_major=element_blank())
    )  # fmt: skip


def fig_accuracy(oos: pl.DataFrame, has_official: bool, label: str):
    """Out-of-sample predicted vs observed two-party share, with the 45-degree line."""
    d = oos.to_pandas()
    xcol = "official_dem2p" if has_official else "obs_dem2p"
    d = d.dropna(subset=[xcol])
    what = "certified result" if has_official else "CES respondents"
    return (
        ggplot(d, aes(xcol, "pred_dem2p"))
        + geom_abline(slope=1, intercept=0, color=GREY, linetype="dashed")
        + geom_point(color=BLUE, size=2.2, alpha=0.75, stroke=0)
        + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="Out-of-sample accuracy",
               subtitle=f"Leave-one-state-out, trained on {label}; each point is one state-race",
               x=f"Observed Democratic two-party share ({what})", y="Predicted share")
        + base_theme(figure_size=(6, 5.5))
    )  # fmt: skip


def fig_metrics(metrics: pl.DataFrame):
    """Error of the Dem-share prediction against a mean-share baseline."""
    d = metrics.to_pandas()
    d["train"] = pd.Categorical(d["train"], categories=list(dict.fromkeys(d["train"])), ordered=True)
    return (
        ggplot(d, aes("train"))
        + geom_col(aes(y="mae_dem"), fill=BLUE, width=0.6)
        + geom_point(aes(y="baseline_mae_dem"), color=INK, shape="D", size=2.4)
        + facet_grid("features ~ scheme", scales="free_x")
        + labs(title="Prediction error by training set",
               subtitle="Bars: model error in Democratic share (points); diamonds: mean-share baseline",
               x="Training years", y="Mean absolute error (percentage points)")
        + base_theme(figure_size=(14, 5.5), axis_text_x=element_text(angle=30, ha="right", color=INK2))
    )  # fmt: skip


def all_figures(res: dict) -> dict:
    return {
        "forecast_bars": fig_forecast(res["preds"]),
        "forecast_map": fig_tilemap(res["preds"]),
        "model_comparison": fig_models(res["preds"]),
        "accuracy_scatter": fig_accuracy(res["oos"], res["has_official"], res["oos_label"]),
        "error_by_model": fig_metrics(res["metrics"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "simulated"], default="simulated")
    args = ap.parse_args()
    out = HERE.parent / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in all_figures(load(args.source, HERE.parent)).items():
        fig.save(out / f"{args.source}_{name}.png", dpi=150, verbose=False)
        print(f"saved {out / f'{args.source}_{name}.png'}")
