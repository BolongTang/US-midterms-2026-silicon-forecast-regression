#### Preamble ####
# Purpose: Tables, figures and cards for the silicon-sampling part of the website, built
#          only from the stored LLM results in data/02-analysis_data. Nothing here calls an
#          LLM; the numbers come from scripts/03-query_models.py (responses) and
#          scripts/04-forecast.py (backtest, calibration and forecast).
# Author: Yi Zhi Zhang
# Date: 31 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: the parquet files from scripts/03 and scripts/04; candidates_2026.csv
#   (scripts/01e) for names and portraits; senate_races.csv for who is on each ballot.
#
# Style follows the regression part of the page: same colours (blue Democratic or
# opposition, red Republican), same plot theme, same race cards and map.
#
# Run with: imported by paper/index.qmd

#### Workspace setup ####
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from plotnine import (
    aes, coord_flip, element_blank, element_text, facet_grid, facet_wrap, geom_abline,
    geom_col, geom_point, geom_segment, geom_text, geom_vline, ggplot, labs,
    scale_color_manual, scale_fill_manual, scale_shape_manual, scale_x_continuous,
    scale_x_discrete, scale_y_continuous,
)  # fmt: skip

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "02-analysis_data"
RAW = ROOT / "data" / "01-raw_data"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


figs = _load("figs", "06-make_figures.py")  # colours and plot theme shared with the regression part
views = _load("views", "07-race_views.py")  # race cards and map shared with the regression part
BLUE, RED, GREY, INK, INK2 = figs.BLUE, figs.RED, figs.GREY, figs.INK, figs.INK2

MODEL_NAME = {
    "openai/gpt-4o-mini": "GPT-4o mini",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
}
MODEL_ORDER = ["GPT-4o mini", "Llama 3.3 70B", "DeepSeek V4 Flash"]
MODEL_COLOURS = {"GPT-4o mini": "#eb6834", "Llama 3.3 70B": "#1baf7a", "DeepSeek V4 Flash": "#6250d6"}
MODEL_SHAPES = {"GPT-4o mini": "o", "Llama 3.3 70B": "s", "DeepSeek V4 Flash": "^"}
BACKTESTED = ["GPT-4o mini", "Llama 3.3 70B"]  # models with a 2024 backtest; only these feed the headline
PID7_ORDER = ["Strong Democrat", "Not Very Strong Democrat", "Lean Democrat", "Independent",
              "Lean Republican", "Not Very Strong Republican", "Strong Republican", "Not Sure"]  # fmt: skip
ANSWERS = ["Democrat", "Republican", "Independent", "Other", "Would not vote", "No valid answer"]
ANSWER_COLOURS = {"Democrat": BLUE, "Republican": RED, "Independent": "#1baf7a", "Other": "#9a9893",
                  "Would not vote": "#d5d4cf", "No valid answer": "#efeeea"}  # fmt: skip
SEATS_NOT_UP_REP = 31  # as assumed in scripts/04-forecast.py (Republican seats not up in 2026)
MAJORITY_REP = 50  # the Vice President breaks a 50-50 tie


def short(model: str) -> str:
    return MODEL_NAME.get(model, model)


#### Loading ####
def load_silicon(root: Path = ROOT) -> dict:
    d = root / "data" / "02-analysis_data"
    summary = pl.read_parquet(d / "forecast_2026_summary.parquet")
    model_cols = [c for c in summary.columns if "/" in c]
    summary = summary.rename({c: short(c) for c in model_cols})
    est = pl.read_parquet(d / "estimates_2026_by_model.parquet").with_columns(model=pl.col("model").replace(MODEL_NAME))
    bt = pl.read_parquet(d / "backtest_2024_states.parquet").with_columns(model=pl.col("model").replace(MODEL_NAME))
    calib = pl.read_parquet(d / "calibration_2024.parquet").with_columns(model=pl.col("model").replace(MODEL_NAME))
    ind = pl.read_parquet(d / "backtest_2024_individual.parquet")
    by_party = pl.read_parquet(d / "backtest_2024_by_party.parquet")
    resp = {n: pl.read_parquet(d / f"{n}.parquet") for n in
            ["responses_2024", "responses_2024_forced", "responses_2026", "responses_2026_deepseek", "responses_2026_forced"]}  # fmt: skip
    return {"summary": summary, "est": est, "bt": bt, "calib": calib, "ind": ind, "by_party": by_party,
            "seats": pl.read_parquet(d / "forecast_2026_seats.parquet"), "resp": resp,
            "personas": pl.read_parquet(d / "personas_2026.parquet"),
            "races": pl.read_csv(root / "data" / "01-raw_data" / "senate_races.csv").filter(pl.col("year") == 2026),
            "cands": views.load_candidates(root / "data" / "01-raw_data" / "candidates_2026.csv"),
            "models_26": MODEL_ORDER}  # fmt: skip


#### Headline numbers ####
def control_stats(seats: pl.DataFrame) -> dict:
    s = seats["rep_seats"]
    return {"p_rep_control": float((s >= MAJORITY_REP).mean()), "p_rep_51": float((s >= 51).mean()),
            "median": float(s.median()), "lo": float(s.quantile(0.1)), "hi": float(s.quantile(0.9)),
            "mean": float(s.mean())}  # fmt: skip


def glance_html(summary: pl.DataFrame, seats: pl.DataFrame) -> str:
    c = control_stats(seats)
    rep = int((summary["p_rep_win"] >= 0.5).sum())
    close = summary.with_columns(gap=(pl.col("p_rep_win") - 0.5).abs()).sort("gap").head(3)["state"].to_list()
    tiles = [
        (f"{100 * c['p_rep_control']:.0f}%", "chance Republicans keep control", "50 or more seats, with the Vice President breaking ties"),
        (f"{c['median']:.0f}", "Republican seats, median", f"80% interval {c['lo']:.0f} to {c['hi']:.0f} of 100"),
        (f"{rep} of {summary.height}", "races where a Republican is favoured", f"{summary.height - rep} favour the opposition"),
        (", ".join(close), "closest races", "win probability nearest 50%"),
    ]
    inner = "".join(
        f'<div class="tile"><div class="big">{a}</div><div class="lab">{b}</div><div class="sub">{s}</div></div>' for a, b, s in tiles
    )
    return ("<style>.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0}"
            "@media (max-width:760px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}"
            ".tile{border:1px solid #d9d8d3;border-radius:10px;padding:12px;background:#fcfcfb;color:#0b0b0b}"
            ".tile .big{font-size:26px;font-weight:700;line-height:1.1}.tile .lab{font-size:13px;margin-top:4px}"
            ".tile .sub{font-size:11px;color:#52514e;margin-top:3px}</style>"
            f'<div class="tiles">{inner}</div>')  # fmt: skip


#### Races: cards and map ####
def _find(cands: pl.DataFrame, st: str, name: str | None, party: str) -> dict | None:
    """The candidate-file row for a name from senate_races.csv (names differ by middle initials)."""
    if not name:
        return None
    sub = cands.filter(pl.col("st") == st)
    key = lambda s: (s.replace('"', "").split()[-1].lower(), s[:1].lower())  # noqa: E731
    for r in sub.iter_rows(named=True):
        if key(r["name"]) == key(name):
            return r
    return next(iter(sub.filter(pl.col("party") == party).iter_rows(named=True)), None)


def silicon_races(summary: pl.DataFrame, races: pl.DataFrame, cands: pl.DataFrame) -> list[dict]:
    """One dict per race for views.cards_html and views.winner_map. The bar is Republican
    against the main opposition (a Democrat, or an independent in ID, MT, NE and SD)."""
    out = []
    by_state = {r["state"]: r for r in races.iter_rows(named=True)}
    for s in summary.sort("state").iter_rows(named=True):
        r = by_state[s["state"]]
        rep = _find(cands, s["state"], r["republican"], "Republican")
        indep = r["main_opponent"] == "Independent"
        opp = _find(cands, s["state"], r["independent"] if indep else r["democrat"], "Other" if indep else "Democratic")
        rep_win = s["p_rep_win"] >= 0.5
        win = rep if rep_win else opp
        models = " · ".join(f"{m.split()[0]} {100 * s[m]:.0f}%" for m in MODEL_ORDER if m in s)
        out.append({
            "st": s["state"], "state": views.ABBR_NAME[s["state"]], "dem": opp, "rep": rep, "others": [],
            "dem2p": 1 - s["median"], "p_dem": 1 - s["median"], "p_rep": s["median"], "p_und": 0.0,
            "winner_party": "Republican" if rep_win else "Democratic", "winner": win,
            "winner_name": win["name"] if win else ("Republican" if rep_win else "the opposition"),
            "bar_left": "I" if indep else "D", "tag": "R" if rep_win else ("I" if indep else "D"),
            "opp_label": "Independent" if indep else "Democratic",
            "meta": [f"P(Republican wins) {100 * s['p_rep_win']:.0f}% · 80% interval for R share {100 * s['lower_80']:.0f}%-{100 * s['upper_80']:.0f}%",
                     f"Raw model estimates of R share: {models}"],  # fmt: skip
        })
    return out


#### Figures ####
def _pct_axis(p):
    return p + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v])


def fig_seats(seats: pl.DataFrame):
    d = seats.group_by("rep_seats").len().with_columns(p=pl.col("len") / seats.height, control=pl.col("rep_seats") >= MAJORITY_REP).to_pandas()
    d["control"] = np.where(d["control"], "Republican control (50+)", "Democratic control")
    c = control_stats(seats)
    return (
        ggplot(d, aes("rep_seats", "p", fill="control"))
        + geom_col(width=0.85, color="#fcfcfb", size=0.4)
        + geom_vline(xintercept=MAJORITY_REP - 0.5, color=INK2, linetype="dashed")
        + scale_fill_manual({"Republican control (50+)": RED, "Democratic control": BLUE})
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="Republican seats after the 2026 election",
               subtitle=f"10,000 simulated elections; Republicans keep control in {100 * c['p_rep_control']:.0f}% (median {c['median']:.0f} seats)",
               x="Republican seats out of 100", y="Share of simulations")
        + figs.base_theme(figure_size=(8, 4.2))
    )  # fmt: skip


def fig_intervals(summary: pl.DataFrame):
    d = summary.sort("median").to_pandas()
    d["state"] = pd.Categorical(d["state"], categories=d["state"].tolist(), ordered=True)
    d["call"] = np.where(d["p_rep_win"] >= 0.5, "Republican favoured", "Opposition favoured")
    return (
        ggplot(d, aes(y="state", color="call"))
        + geom_vline(xintercept=0.5, color=GREY, linetype="dashed")
        + geom_segment(aes(x="lower_80", xend="upper_80", yend="state"), size=1.4, alpha=0.85)
        + geom_point(aes(x="median"), size=2.6)
        + scale_color_manual({"Republican favoured": RED, "Opposition favoured": BLUE})
        + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="Forecast Republican share, every race",
               subtitle="Dot: median. Bar: 80% interval. Share of the vote against the main opposition",
               x="Republican share of the two-candidate vote", y=None)
        + figs.base_theme(figure_size=(7, 9), axis_title_y=element_blank())
    )  # fmt: skip


def fig_models26(summary: pl.DataFrame):
    models = [m for m in MODEL_ORDER if m in summary.columns]
    d = summary.select("state", "median", *models).to_pandas().sort_values("median")
    order = d["state"].tolist()
    long = d.melt(id_vars=["state", "median"], value_vars=models, var_name="model", value_name="share")
    long["state"] = pd.Categorical(long["state"], categories=order, ordered=True)
    long["model"] = pd.Categorical(long["model"], categories=models, ordered=True)
    return (
        ggplot(long, aes("share", "state", color="model", shape="model"))
        + geom_vline(xintercept=0.5, color=GREY, linetype="dashed")
        + geom_point(size=2.4, stroke=0.4, alpha=0.9)
        + scale_color_manual(MODEL_COLOURS) + scale_shape_manual(MODEL_SHAPES)
        + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="Do the models agree?", subtitle="Raw Republican share from each model's synthetic voters, before calibration",
               x="Republican share", y=None)
        + figs.base_theme(figure_size=(7, 9), axis_title_y=element_blank())
    )  # fmt: skip


def fig_backtest(bt: pl.DataFrame, calib: pl.DataFrame):
    d = bt.to_pandas()
    d["result"] = np.where(d["winner_right"], "winner called correctly", "winner missed")
    ln = pd.DataFrame([{"model": r["model"], "x": 0.15, "y": r["intercept"] + r["slope"] * 0.15,
                        "xend": 0.7, "yend": r["intercept"] + r["slope"] * 0.7} for r in calib.iter_rows(named=True)])  # fmt: skip
    return (
        ggplot(d, aes("rep_share", "actual"))
        + geom_abline(slope=1, intercept=0, color=GREY, linetype="dashed")
        + geom_segment(aes(x="x", y="y", xend="xend", yend="yend"), data=ln, color=INK, size=0.8, inherit_aes=False)
        + geom_point(aes(color="result"), size=2.2, alpha=0.85, stroke=0)
        + facet_wrap("model")
        + scale_color_manual({"winner called correctly": BLUE, "winner missed": RED})
        + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="2024 backtest: synthetic voters versus the real result",
               subtitle="Each dot is a Senate race. Dashed: perfect. Solid: the calibration line fitted to these 33 races",
               x="Republican share from the models", y="Actual Republican share")
        + figs.base_theme(figure_size=(8, 4.4))
    )  # fmt: skip


def _clean(m: str) -> str:
    return (short(m.split(" (")[0]) + (" (same people as baseline)" if "baseline's" in m else "")) if "baseline" not in m or "(" in m else "Party-line baseline"


def fig_agreement(ind: pl.DataFrame):
    d = ind.to_pandas()
    d["label"] = d["model"].map(_clean)
    d["kind"] = np.where(d["label"].str.startswith("Party-line"), "Party-line baseline", "Synthetic voters")
    d = d.sort_values("agreement")
    d["label"] = pd.Categorical(d["label"], categories=d["label"].tolist(), ordered=True)
    return (
        ggplot(d, aes("label", "agreement", fill="kind"))
        + geom_col(width=0.7, color="#fcfcfb", size=0.6)
        + geom_text(aes(label="agreement"), format_string="{:.1%}", nudge_y=0.02, size=8, color=INK)
        + coord_flip()
        + scale_fill_manual({"Synthetic voters": BLUE, "Party-line baseline": GREY})
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v], limits=(0, 1.08), expand=(0, 0))
        + labs(title="2024: does each synthetic voter pick the same side as the real person?",
               subtitle="Republican versus opposition, real CES respondents, matched to their reported Senate vote",
               x=None, y="Agreement")
        + figs.base_theme(figure_size=(8, 3.6), axis_title_y=element_blank())
    )  # fmt: skip


def fig_by_party(by_party: pl.DataFrame):
    d = by_party.to_pandas()
    d["model"] = d["model"].map(_clean)
    d = d[d["model"].isin(BACKTESTED + ["Party-line baseline"]) & d["pid7"].isin(PID7_ORDER)]
    d["pid7"] = pd.Categorical(d["pid7"], categories=PID7_ORDER[::-1], ordered=True)
    cols = {**{m: MODEL_COLOURS[m] for m in BACKTESTED}, "Party-line baseline": INK}
    shapes = {**{m: MODEL_SHAPES[m] for m in BACKTESTED}, "Party-line baseline": "D"}
    return (
        ggplot(d, aes("agreement", "pid7", color="model", shape="model"))
        + geom_point(size=2.8, alpha=0.9, stroke=0.4)
        + scale_color_manual(cols) + scale_shape_manual(shapes)
        + scale_x_continuous(labels=lambda v: [f"{x:.0%}" for x in v], limits=(0.4, 1.0))
        + labs(title="Where do synthetic voters go wrong?", subtitle="2024 agreement with the real vote, by party identification",
               x="Agreement with the real vote", y=None)
        + figs.base_theme(figure_size=(8, 4.2), axis_title_y=element_blank())
    )  # fmt: skip


#### Responses ####
def free_2026(resp: dict) -> pl.DataFrame:
    """The 2026 free-choice answers used in the forecast: DeepSeek's first run failed (its
    reasoning used up the token limit), so its no-reasoning rerun replaces it."""
    cols = ["persona_id", "state", "model", "repeat", "answer"]
    first = resp["responses_2026"].filter(pl.col("model") != "deepseek/deepseek-v4-flash").select(cols)
    return pl.concat([first, resp["responses_2026_deepseek"].select(cols)]).with_columns(model=pl.col("model").replace(MODEL_NAME))


def answer_table(resp: dict) -> pl.DataFrame:
    parts = []
    for label, df in [("Free choice", free_2026(resp)),
                      ("Forced choice (pure independents)", resp["responses_2026_forced"].select("persona_id", "state", "model", "repeat", "answer").with_columns(model=pl.col("model").replace(MODEL_NAME)))]:  # fmt: skip
        n = df.group_by("model").len().rename({"len": "n"})
        c = df.with_columns(answer=pl.col("answer").fill_null("No valid answer")).group_by("model", "answer").len()
        parts.append(c.join(n, on="model").with_columns(share=pl.col("len") / pl.col("n"), design=pl.lit(label)))
    return pl.concat(parts)


def fig_answers(resp: dict):
    d = answer_table(resp).to_pandas()
    d["answer"] = pd.Categorical(d["answer"], categories=ANSWERS[::-1], ordered=True)
    d["model"] = pd.Categorical(d["model"], categories=MODEL_ORDER[::-1], ordered=True)
    return (
        ggplot(d, aes("model", "share", fill="answer"))
        + geom_col(width=0.7, color="#fcfcfb", size=0.5)
        + coord_flip() + facet_grid("design ~ .")
        + scale_fill_manual(ANSWER_COLOURS, breaks=ANSWERS)
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v], expand=(0, 0))
        + labs(title="What do the models answer?", subtitle="2026 Senate vote question, all personas and five repeats per model",
               x=None, y="Share of replies")
        + figs.base_theme(figure_size=(8, 4.4), axis_title_y=element_blank())
    )  # fmt: skip


def run_quality(resp: dict) -> pl.DataFrame:
    """Valid-answer rate per run, including DeepSeek's failed first run (excluded from the forecast)."""
    rows = []
    for name, df in resp.items():
        for m, g in df.group_by("model"):
            rows.append({"run": name, "model": short(m[0]), "replies": g.height, "valid": float(g["answer"].is_not_null().mean())})
    return pl.DataFrame(rows).sort("run", "model")


#### Cross-tabs ####
def crosstab(resp: dict, personas: pl.DataFrame, by: str) -> pd.DataFrame:
    """Weighted Republican share among Republican, Democrat and independent answers, by a persona trait."""
    p = personas.with_columns(
        age_group=pl.when(pl.col("age") < 30).then(pl.lit("18-29")).when(pl.col("age") < 45).then(pl.lit("30-44"))
        .when(pl.col("age") < 65).then(pl.lit("45-64")).otherwise(pl.lit("65+")),
    ).select("persona_id", by, w=pl.col("weight").fill_null(0))
    d = free_2026(resp).join(p, on="persona_id").filter(pl.col("answer").is_in(["Republican", "Democrat", "Independent"]))
    g = d.group_by(by, "model").agg(rep=((pl.col("answer") == "Republican") * pl.col("w")).sum() / pl.col("w").sum())
    wide = g.pivot(on="model", index=by, values="rep").to_pandas()
    wide = wide[[by] + [m for m in MODEL_ORDER if m in wide.columns]]
    if by == "pid7":
        wide[by] = pd.Categorical(wide[by], categories=PID7_ORDER, ordered=True)
    return wide.sort_values(by).reset_index(drop=True)


def composition(personas: pl.DataFrame) -> pl.DataFrame:
    p = personas.with_columns(
        age_group=pl.when(pl.col("age") < 30).then(pl.lit("18-29")).when(pl.col("age") < 45).then(pl.lit("30-44"))
        .when(pl.col("age") < 65).then(pl.lit("45-64")).otherwise(pl.lit("65+")))
    parts = []
    for var, label in [("pid7", "Party identification"), ("race", "Race"), ("gender", "Gender"), ("age_group", "Age"), ("educ", "Education")]:
        g = p.with_columns(pl.col(var).fill_null("Unknown")).group_by(var).agg(n=pl.len(), share=pl.len() / p.height).rename({var: "category"})
        parts.append(g.with_columns(variable=pl.lit(label)))
    return pl.concat(parts)


def fig_composition(personas: pl.DataFrame):
    d = composition(personas).to_pandas()
    d["category"] = d["category"].astype(str)
    d = d.sort_values(["variable", "share"])
    d["key"] = pd.Categorical(d["variable"] + "|" + d["category"], categories=(d["variable"] + "|" + d["category"]).tolist(), ordered=True)
    return (
        ggplot(d, aes("key", "share"))
        + geom_col(fill=BLUE, width=0.7)
        + coord_flip()
        + facet_wrap("variable", scales="free_y", ncol=2)
        + scale_x_discrete(labels=lambda ks: [k.split("|")[1] for k in ks])
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(title="Who are the synthetic voters?",
               subtitle=f"{personas.height:,} personas built from real 2024 CES respondents in the 35 states with a 2026 race",
               x=None, y="Share of personas")
        + figs.base_theme(figure_size=(8, 6.4), axis_text_y=element_text(size=7, color=INK2), axis_title_y=element_blank())
    )  # fmt: skip


#### Models and costs ####
def cost_table(resp: dict) -> pd.DataFrame:
    labels = {"responses_2024": "2024 backtest, free choice", "responses_2024_forced": "2024 backtest, forced choice (independents)",
              "responses_2026": "2026 forecast, free choice (first run)", "responses_2026_deepseek": "2026 DeepSeek rerun, no reasoning",
              "responses_2026_forced": "2026 forced choice (independents)"}  # fmt: skip
    rows = []
    for name, df in resp.items():
        rows.append({"Run": labels[name], "Models": ", ".join(short(m) for m in sorted(df["model"].unique())), "replies": df.height,
                     "prompt": df["prompt_tokens"].sum() / 1e6, "completion": df["completion_tokens"].sum() / 1e6,
                     "cost": df["cost_usd"].sum(), "Queried": str(df["queried_at"].min())[:10]})  # fmt: skip
    out = pd.DataFrame(rows)
    tot = {"Run": "Total", "Models": "", "replies": out["replies"].sum(), "prompt": out["prompt"].sum(),
           "completion": out["completion"].sum(), "cost": out["cost"].sum(), "Queried": ""}  # fmt: skip
    out = pd.concat([out, pd.DataFrame([tot])], ignore_index=True)
    out["cost_usd_total"] = out["cost"]  # numeric, kept for tests
    out["Replies"] = out["replies"].map("{:,.0f}".format)
    out["Prompt tokens (M)"] = out["prompt"].map("{:.1f}".format)
    out["Completion tokens (M)"] = out["completion"].map("{:.2f}".format)
    out["Cost (USD)"] = out["cost"].map("{:.2f}".format)
    return out[["Run", "Models", "Replies", "Prompt tokens (M)", "Completion tokens (M)", "Cost (USD)", "Queried", "cost_usd_total"]]


def cost_by_model(resp: dict) -> pd.DataFrame:
    allr = pl.concat([df.select("model", "prompt_tokens", "completion_tokens", "cost_usd") for df in resp.values()])
    g = allr.group_by("model").agg(replies=pl.len(), cost=pl.col("cost_usd").sum(), per_1000=1000 * pl.col("cost_usd").mean()).to_pandas()
    g["Model"] = g["model"].map(short)
    g["Replies"] = g["replies"].map("{:,.0f}".format)
    g["Cost (USD)"] = g["cost"].map("{:.2f}".format)
    g["Cost per 1,000 replies (USD)"] = g["per_1000"].map("{:.3f}".format)
    return g[["Model", "Replies", "Cost (USD)", "Cost per 1,000 replies (USD)"]].sort_values("Model")


def backtest_summary(bt: pl.DataFrame) -> pd.DataFrame:
    g = bt.group_by("model").agg(bias=pl.col("error").mean() * 100, mae=pl.col("error").abs().mean() * 100,
                                 right=pl.col("winner_right").sum(), races=pl.len()).to_pandas()
    g["Winners called"] = g["right"].astype(int).astype(str) + " of " + g["races"].astype(str)
    return g.rename(columns={"model": "Model", "bias": "Bias (pp, actual minus model)", "mae": "Mean absolute error (pp)"})[
        ["Model", "Bias (pp, actual minus model)", "Mean absolute error (pp)", "Winners called"]].sort_values("Model")  # fmt: skip


#### Comparison with the regression part ####
def compare_with_regression(summary: pl.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    d = root / "data" / "02-analysis_data"
    sel = pl.read_parquet(d / "model_selection.parquet").filter(pl.col("selected"), pl.col("phase") == "validation").row(0, named=True)
    reg = pl.read_parquet(d / "regression_forecast_2026.parquet").filter(pl.col("features") == sel["features"], pl.col("train") == "all")
    reg = reg.select("st", reg_dem2p=pl.col("p_dem") / (pl.col("p_dem") + pl.col("p_rep")))
    j = summary.select("state", "p_rep_win", "median").join(reg, left_on="state", right_on="st")
    return j.with_columns(silicon_rep=pl.col("p_rep_win") >= 0.5, regression_rep=pl.col("reg_dem2p") < 0.5).to_pandas()
