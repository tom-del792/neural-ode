"""The warm-up: a plain network learns the panel which the mechanistic model simulates.

No hybrid model yet. The steps are the ones of every later variant: generate
the data, train, evaluate; each writes its files below `hybrid.runs.runs_dir`.
"""

import csv
import json
import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrid.config import Config, read_config
from hybrid.data.dataset import STREAM_TRAINING, Panel, generate_panel, rng
from hybrid.data.io import load_panel, save_panel
from hybrid.evaluation.metrics import panel_metrics
from hybrid.evaluation.plots import plot_fits, plot_learning, plot_parity, spread_subjects
from hybrid.nn.panel import PanelNetwork
from hybrid.runs import dataset_path, new_run, runs_dir
from hybrid.training.checkpoint import load_weights, save_weights
from hybrid.training.fit import FitSettings, fit
from hybrid.training.loss import ErrorModel, gaussian_nll

logger = logging.getLogger(__name__)


class WarmupModel(eqx.Module):
    """What is trained in the warm-up: the network and the error model of the measurements."""

    network: PanelNetwork
    error: ErrorModel


def dataset(config: Config, regenerate: bool = False) -> Panel:
    """The panel of a case: loaded from its file, or generated and saved if it does not exist."""
    path = dataset_path(config)
    if path.is_file() and not regenerate:
        logger.info("dataset %s", path)
        return load_panel(path)
    logger.info("generating %d patients", config.n_subjects)
    panel = generate_panel(config)
    save_panel(path, panel)
    logger.info("dataset written to %s", path)
    return panel


def build(config: Config, panel: Panel) -> WarmupModel:
    """The untrained model of a case, with the initial weights of its seed."""
    network = PanelNetwork.create(panel, config.network, jax.random.PRNGKey(config.seed))
    return WarmupModel(network=network, error=ErrorModel.create(network.scale_out.scale))


def loss(model: WarmupModel, times: jax.Array, x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    """Negative log-likelihood of the measurements `y` of the patients `x`."""
    prediction = model.network.predict(times, x)
    return gaussian_nll(prediction, y, model.error.sigma(prediction), mask)


def train(config: Config, regenerate: bool = False) -> Path:
    """Trains the network of a case and evaluates it.

    Returns:
        The directory of the run with `config.toml`, `run.json`, `weights.eqx`,
        `log.csv`, `metrics.json` and the figures.
    """
    panel = dataset(config, regenerate)
    directory = new_run(config)
    logger.info("run %s", directory)
    times = jnp.asarray(panel.times)
    parts = [panel.subset(train=True), panel.subset(train=False)]
    settings = FitSettings(
        batch_size=config.training.batch_size,
        adam_steps=config.training.adam_steps,
        learning_rate=config.training.learning_rate,
        lbfgs_steps=config.training.lbfgs_steps,
        eval_every=config.training.eval_every,
    )
    model, log = fit(
        build(config, panel),
        lambda m, x, y, mask: loss(m, times, x, y, mask),
        *((part.x, part.y, part.mask) for part in parts),
        settings,
        rng(config, STREAM_TRAINING),
    )
    save_weights(directory / "weights.eqx", model)
    with (directory / "log.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "phase", "train", "valid"])
        writer.writeheader()
        writer.writerows(log)
    evaluate(directory)
    return directory


def load(directory: Path) -> tuple[Config, Panel, WarmupModel]:
    """The configuration, the panel and the trained model of a run."""
    config = read_config(directory / "config.toml")
    record = json.loads((directory / "run.json").read_text())
    panel = load_panel(runs_dir() / record["dataset"])
    return config, panel, load_weights(directory / "weights.eqx", build(config, panel))


def evaluate(directory: Path) -> dict:
    """Writes `metrics.json` and the figures of a run from its saved weights.

    Returns:
        The metrics: errors per observable on the training and the validation
        patients, and the trained error model next to the one of the data.
    """
    config, panel, model = load(directory)
    predict = jax.jit(lambda times, x: model.network.predict(times, x))
    times = jnp.asarray(panel.times)
    metrics: dict = {"observables": {}, "error_model": {}}
    for name, part in (("train", panel.subset(train=True)), ("valid", panel.subset(train=False))):
        prediction = np.asarray(predict(times, jnp.asarray(part.x)))
        metrics["observables"][name] = panel_metrics(prediction, part)
        metrics[f"n_{name}"] = len(part.x)
    for k, (name, observable) in enumerate(config.observables.items()):
        metrics["error_model"][name] = {
            "proportional": float(model.error.proportional[k]),
            "proportional_data": observable.proportional,
            "additive": float(model.error.additive[k]),
            "additive_data": observable.additive,
        }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    valid = panel.subset(train=False)
    rows = spread_subjects(valid, 4)
    dense = np.linspace(panel.times[0], panel.times[-1], 201)
    plot_fits(
        directory / "fits.png",
        valid,
        rows,
        dense,
        np.asarray(predict(jnp.asarray(dense), jnp.asarray(valid.x[rows]))),
        f"{config.name}: validation patients",
    )
    plot_parity(
        directory / "parity.png",
        valid,
        np.asarray(predict(times, jnp.asarray(valid.x))),
        f"{config.name}: validation patients, all sampling times",
    )
    log_path = directory / "log.csv"
    if log_path.is_file():
        with log_path.open(newline="") as file:
            log = [
                {
                    "step": int(row["step"]),
                    "phase": row["phase"],
                    "train": float(row["train"]),
                    "valid": float(row["valid"]),
                }
                for row in csv.DictReader(file)
            ]
        plot_learning(directory / "learning.png", log, f"{config.name}: training")
    return metrics
