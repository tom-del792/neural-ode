"""Figures of a trained model: fits of single patients, predicted against simulated, learning curves."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from hybrid.data.dataset import Panel

NETWORK = "#2a78d6"
MECHANISTIC = "#eb6834"
INK = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e4e3dc"


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=7, length=3)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def _save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor="white")


def spread_subjects(panel: Panel, n: int) -> np.ndarray:
    """Rows of `n` patients spread over the first input (the dose), then the second."""
    order = np.lexsort(panel.x.T[::-1])
    return order[np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int)]


def plot_fits(
    path: Path,
    panel: Panel,
    rows: np.ndarray,
    dense_times: np.ndarray,
    dense_prediction: np.ndarray,
    title: str,
) -> None:
    """Time courses of single patients: measurements, mechanistic model and network.

    Args:
        path: File of the figure.
        panel: The patients, usually the validation patients.
        rows: The patients which are shown, one column each.
        dense_times: Times of the predictions of the network.
        dense_prediction: Predictions `(len(rows), len(dense_times), n_observables)`.
        title: Title of the figure.
    """
    n_rows, n_columns = len(panel.y_names), len(rows)
    figure = Figure(figsize=(2.4 * n_columns + 0.6, 1.7 * n_rows + 1.0), layout="constrained")
    axes = figure.subplots(n_rows, n_columns, sharex=True, sharey="row", squeeze=False)
    for j, row in enumerate(rows):
        inputs = ", ".join(
            f"{name} {value:.3g}" for name, value in zip(panel.x_names, panel.x[row], strict=True)
        )
        axes[0, j].set_title(inputs.replace(", ", "\n"), fontsize=7, color=INK, loc="left")
        for k in range(n_rows):
            axis = axes[k, j]
            _style(axis)
            mask = panel.mask[row, :, k]
            (truth,) = axis.plot(
                panel.times, panel.y_clean[row, :, k], color=MECHANISTIC, linewidth=1.5, linestyle="--"
            )
            (network,) = axis.plot(dense_times, dense_prediction[j, :, k], color=NETWORK, linewidth=1.8)
            data = axis.scatter(
                panel.times[mask],
                panel.y[row, mask, k],
                s=12,
                color=INK,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
    for k, name in enumerate(panel.y_names):
        axes[k, 0].set_ylabel(name, fontsize=8, color=INK)
    for axis in axes[-1]:
        axis.set_xlabel("time", fontsize=8, color=INK)
    figure.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    figure.legend(
        [data, truth, network],
        ["measurements", "mechanistic model", "network"],
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    _save(figure, path)


def plot_parity(path: Path, panel: Panel, prediction: np.ndarray, title: str) -> None:
    """Predictions of the network against the simulated values without noise, one panel per observable."""
    n = len(panel.y_names)
    figure = Figure(figsize=(2.6 * n + 0.4, 3.1), layout="constrained")
    axes = figure.subplots(1, n, squeeze=False)[0]
    for k, (axis, name) in enumerate(zip(axes, panel.y_names, strict=True)):
        _style(axis)
        truth, predicted = panel.y_clean[..., k].ravel(), prediction[..., k].ravel()
        low, high = min(truth.min(), predicted.min()), max(truth.max(), predicted.max())
        axis.plot([low, high], [low, high], color=MUTED, linewidth=1.0)
        axis.scatter(truth, predicted, s=6, color=NETWORK, alpha=0.5, linewidths=0)
        axis.set_title(name, fontsize=8, color=INK, loc="left")
        axis.set_xlabel("mechanistic model", fontsize=8, color=INK)
        axis.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("network", fontsize=8, color=INK)
    figure.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    _save(figure, path)


def plot_learning(path: Path, log: list[dict], title: str) -> None:
    """Training and validation loss over the steps; a line marks the start of L-BFGS."""
    figure = Figure(figsize=(5.2, 3.2), layout="constrained")
    axis = figure.subplots()
    _style(axis)
    steps = np.asarray([entry["step"] for entry in log])
    axis.plot(
        steps, [entry["train"] for entry in log], color=NETWORK, linewidth=1.8, label="training patients"
    )
    axis.plot(
        steps,
        [entry["valid"] for entry in log],
        color=MECHANISTIC,
        linewidth=1.8,
        label="validation patients",
    )
    lbfgs = [entry["step"] for entry in log if entry["phase"] == "lbfgs"]
    if lbfgs:
        start = max(entry["step"] for entry in log if entry["phase"] == "adam")
        axis.axvline(start, color=MUTED, linewidth=1.0, linestyle=":")
        axis.annotate(
            " L-BFGS", (start, 1.0), xycoords=("data", "axes fraction"), va="top", fontsize=7, color=MUTED
        )
    axis.set_xlabel("step", fontsize=8, color=INK)
    axis.set_ylabel("negative log-likelihood per measurement", fontsize=8, color=INK)
    axis.legend(frameon=False, fontsize=8, labelcolor=INK)
    axis.set_title(title, fontsize=10, color=INK, loc="left")
    _save(figure, path)
