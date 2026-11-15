"""Metrics and figures."""

from pathlib import Path

import numpy as np

from hybrid.data.dataset import Panel
from hybrid.evaluation.metrics import panel_metrics
from hybrid.evaluation.plots import plot_fits, plot_learning, plot_parity, spread_subjects


def test_metrics(small_panel: Panel) -> None:
    perfect = panel_metrics(small_panel.y_clean, small_panel)
    for name, k in (("C", 0), ("E", 1)):
        assert perfect[name]["rmse_truth"] == 0 and perfect[name]["r2_truth"] == 1
        noise = (small_panel.y - small_panel.y_clean)[..., k][small_panel.mask[..., k]]
        np.testing.assert_allclose(perfect[name]["rmse_data"], np.sqrt(np.mean(noise**2)), rtol=1e-12)
    # predicting the mean of every observable explains nothing
    mean = np.broadcast_to(small_panel.y_clean.mean(axis=(0, 1)), small_panel.y.shape)
    for values in panel_metrics(mean, small_panel).values():
        np.testing.assert_allclose([values["r2_truth"], values["nrmse_truth"]], [0.0, 1.0], atol=1e-12)


def test_spread_subjects(small_panel: Panel) -> None:
    rows = spread_subjects(small_panel, 4)
    assert len(set(rows)) == 4
    doses = small_panel.x[rows, 0]
    assert (
        np.all(np.diff(doses) >= 0)
        and doses[0] == small_panel.x[:, 0].min()
        and doses[-1] == small_panel.x[:, 0].max()
    )
    assert len(spread_subjects(small_panel.subset(train=False), 100)) == 12


def test_figures(tmp_path: Path, small_panel: Panel) -> None:
    rows = spread_subjects(small_panel, 3)
    dense = np.linspace(small_panel.times[0], small_panel.times[-1], 20)
    plot_fits(tmp_path / "fits.png", small_panel, rows, dense, np.zeros((3, 20, 2)), "fits")
    plot_parity(tmp_path / "parity.png", small_panel, small_panel.y, "parity")
    log = [
        {"step": 0, "phase": "adam", "train": 1.0, "valid": 1.1},
        {"step": 10, "phase": "adam", "train": 0.5, "valid": 0.7},
        {"step": 12, "phase": "lbfgs", "train": 0.4, "valid": 0.6},
    ]
    plot_learning(tmp_path / "learning.png", log, "learning")
    for name in ("fits.png", "parity.png", "learning.png"):
        assert (tmp_path / name).stat().st_size > 5000
