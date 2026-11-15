"""Errors of predictions per observable."""

import numpy as np

from hybrid.data.dataset import Panel


def panel_metrics(prediction: np.ndarray, panel: Panel) -> dict[str, dict[str, float]]:
    """Errors of the predictions of a panel, per observable.

    Args:
        prediction: Predictions of the shape of `panel.y`.
        panel: The patients which were predicted.

    Returns:
        For every observable: `rmse_data` against the measurements which exist,
        `rmse_truth` against the simulated values without noise, `nrmse_truth`
        (`rmse_truth` relative to the standard deviation of the simulated
        values) and `r2_truth`.
    """
    metrics = {}
    for k, name in enumerate(panel.y_names):
        mask = panel.mask[..., k]
        truth = panel.y_clean[..., k]
        error_truth = prediction[..., k] - truth
        rmse_truth = float(np.sqrt(np.mean(error_truth**2)))
        variance = float(np.var(truth))
        metrics[name] = {
            "rmse_data": float(np.sqrt(np.mean((prediction[..., k] - panel.y[..., k])[mask] ** 2))),
            "rmse_truth": rmse_truth,
            "nrmse_truth": rmse_truth / np.sqrt(variance) if variance > 0 else float("nan"),
            "r2_truth": 1.0 - float(np.mean(error_truth**2)) / variance if variance > 0 else float("nan"),
        }
    return metrics
