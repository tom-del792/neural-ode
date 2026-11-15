"""Saving and loading a panel as one `.npz` file."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from hybrid.data.dataset import Panel

_TUPLES = ("x_names", "y_names", "log_x")


def save_panel(path: Path, panel: Panel) -> None:
    """Writes the panel to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(file, **{key: np.asarray(value) for key, value in asdict(panel).items()})


def load_panel(path: Path) -> Panel:
    """Reads a panel written by `save_panel`."""
    with np.load(path, allow_pickle=False) as file:
        fields = {key: file[key] for key in file.files}
    for key in _TUPLES:
        fields[key] = tuple(fields[key].tolist())
    return Panel(**fields)
