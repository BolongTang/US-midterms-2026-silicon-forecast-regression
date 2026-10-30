#### Preamble ####
# Purpose: Test that every regression figure builds and saves from simulated data.
# Author: Yi Zhi Zhang
# Date: 30 October 2026
# Contact: yizhiharry.zhang@mail.utoronto.ca
# License: MIT
# Pre-requisites: None (simulates its own inputs in a temp directory)

#### Workspace setup ####
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    load_module("sim", "00b-simulate_regression_inputs.py").simulate(root / "data" / "00-simulated_data")
    figs = load_module("figs", "06-make_figures.py")
    return figs, figs.load("simulated", root)


def test_every_2026_race_has_shares_in_results(results):
    figs, res = results
    head = figs.pick(res["preds"], "2022+2024")
    assert head["st"].nunique() == 35
    assert ((head["p_dem"] + head["p_rep"] + head["p_und"]) - 1).abs().max() < 1e-9
    assert head["dem2p"].between(0, 1).all()


def test_all_figures_build_and_save(results, tmp_path):
    figs, res = results
    plots = figs.all_figures(res)
    assert set(plots) == {"forecast_bars", "forecast_map", "model_comparison",
                          "accuracy_scatter", "error_by_model"}  # fmt: skip
    for name, fig in plots.items():
        out = tmp_path / f"{name}.png"
        fig.save(out, dpi=60, verbose=False)
        assert out.stat().st_size > 5_000


def test_tile_grid_has_unique_cells_for_all_states(results):
    figs, _ = results
    assert len(figs.TILES) == 51
    assert len(set(figs.TILES.values())) == 51
