"""Saving and loading the arrays of a model."""

from pathlib import Path
from typing import TypeVar

import equinox as eqx

Model = TypeVar("Model")


def save_weights(path: Path, model: Model) -> None:
    """Writes all arrays of the model (weights and scaling constants) to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_weights(path: Path, like: Model) -> Model:
    """Reads the arrays written by `save_weights` into a model of the same structure as `like`."""
    return eqx.tree_deserialise_leaves(path, like)
