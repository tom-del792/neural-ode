"""The configuration of a case: one TOML file with everything specific to it.

The code contains no model ids, doses, units or covariates. A case names its
SBML model, how the patients differ (dose and covariates), what is measured
(observables with their noise), the network and the training.
"""

import hashlib
import itertools
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sbml2jax.pinned import PinnedModel, fetch

DISTRIBUTIONS = ("uniform", "loguniform", "choice")
ACTIVATIONS = ("tanh", "relu", "gelu", "softplus")


class ConfigError(ValueError):
    """A configuration file which cannot be used, with the reason."""


@dataclass(frozen=True)
class ModelConfig:
    """The SBML model: a local file or a pinned download.

    Attributes:
        name: Name of the model (`name` of the table, else the name of the case), the
            directory of a pinned model in the cache.
        path: Local SBML file, or `None` for a pinned model.
        url: URL of a pinned model with the placeholder `{ref}`.
        ref: Commit of a pinned model.
        sha256: Checksum of a pinned model.
    """

    name: str
    path: Path | None = None
    url: str | None = None
    ref: str | None = None
    sha256: str | None = None

    def sbml_path(self) -> Path:
        """The SBML file; a pinned model is downloaded into the cache and verified.

        Raises:
            ConfigError: The local file does not exist.
        """
        if self.path is not None:
            if not self.path.is_file():
                raise ConfigError(f"[model] path {self.path} does not exist")
            return self.path
        assert self.url is not None and self.ref is not None
        return fetch(PinnedModel(name=self.name, url=self.url, ref=self.ref, sha256=self.sha256))


@dataclass(frozen=True)
class Distribution:
    """How a quantity varies between the patients.

    Attributes:
        kind: `uniform` and `loguniform` between `low` and `high`, or `choice` of `values`.
        low: Lower bound.
        high: Upper bound.
        values: The values of a `choice`.
    """

    kind: str
    low: float = 0.0
    high: float = 0.0
    values: tuple[float, ...] = ()

    @property
    def log(self) -> bool:
        """Whether the network sees the logarithm of the quantity (positive, over a ratio scale)."""
        if self.kind == "loguniform":
            return True
        return self.kind == "choice" and min(self.values) > 0


@dataclass(frozen=True)
class Observable:
    """A measured output of the model with the noise `sigma = additive + proportional * value`."""

    proportional: float
    additive: float


@dataclass(frozen=True)
class NetworkConfig:
    """Hidden layers of the network."""

    width: int
    depth: int
    activation: str


@dataclass(frozen=True)
class TrainingConfig:
    """Settings of the training: Adam on batches of patients, then L-BFGS on all."""

    batch_size: int
    adam_steps: int
    learning_rate: float
    lbfgs_steps: int
    eval_every: int


@dataclass(frozen=True)
class Config:
    """A case, read from a TOML file with `read_config`.

    Attributes:
        name: Name of the case, the directory of its runs.
        seed: Seed of the population, the noise, the split and the training.
        model: The SBML model.
        times: Sampling times of every patient, in the time unit of the model.
        simulation_batch_size: Number of patients simulated at once.
        rtol: Relative tolerance of the simulation.
        atol: Absolute tolerance of the simulation.
        dose_target: State or constant which is set to the dose at the first time.
        dose: Distribution of the dose.
        covariates: Inputs of the model which differ between the patients.
        observables: Measured outputs of the model with their noise.
        n_subjects: Number of patients.
        validation_fraction: Fraction of the patients held out of the training.
        missing_fraction: Fraction of the measurements dropped at random.
        network: Hidden layers of the network.
        training: Settings of the training.
        text: The text of the file, copied into every run.
    """

    name: str
    seed: int
    model: ModelConfig
    times: tuple[float, ...]
    simulation_batch_size: int
    rtol: float
    atol: float
    dose_target: str
    dose: Distribution
    covariates: dict[str, Distribution]
    observables: dict[str, Observable]
    n_subjects: int
    validation_fraction: float
    missing_fraction: float
    network: NetworkConfig
    training: TrainingConfig
    text: str

    @property
    def subject_inputs(self) -> tuple[str, ...]:
        """Names of the quantities which differ between the patients: the dose target, then the covariates."""
        return (self.dose_target, *self.covariates)

    @property
    def subject_distributions(self) -> tuple[Distribution, ...]:
        """Distributions of `subject_inputs`."""
        return (self.dose, *self.covariates.values())

    def data_hash(self) -> str:
        """Identifies the dataset: a hash of everything the generated data depends on."""
        raw = tomllib.loads(self.text)
        keys = ("seed", "model", "simulation", "dose", "covariates", "observables", "population")
        content = json.dumps({key: raw.get(key) for key in keys}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:10]


def _distribution(table: dict[str, Any], where: str) -> Distribution:
    kind = table.get("distribution")
    if kind not in DISTRIBUTIONS:
        raise ConfigError(f"[{where}] distribution must be one of {DISTRIBUTIONS}, not {kind!r}")
    if kind == "choice":
        values = tuple(float(value) for value in table.get("values", ()))
        if not values:
            raise ConfigError(f"[{where}] a choice needs `values`")
        return Distribution(kind=kind, values=values)
    try:
        low, high = float(table["low"]), float(table["high"])
    except KeyError as error:
        raise ConfigError(f"[{where}] {kind} needs `low` and `high`") from error
    if not low < high:
        raise ConfigError(f"[{where}] low must be below high")
    if kind == "loguniform" and low <= 0:
        raise ConfigError(f"[{where}] loguniform needs a positive `low`")
    return Distribution(kind=kind, low=low, high=high)


def _model(table: dict[str, Any], name: str, directory: Path) -> ModelConfig:
    name = str(table.get("name", name))
    if "path" in table:
        return ModelConfig(name=name, path=(directory / table["path"]).resolve())
    missing = [key for key in ("url", "ref", "sha256") if key not in table]
    if missing:
        raise ConfigError(
            f"[model] needs `path`, or `url`, `ref` and `sha256` (missing: {', '.join(missing)})"
        )
    return ModelConfig(name=name, url=table["url"], ref=table["ref"], sha256=table["sha256"])


def _fraction(value: float, where: str) -> float:
    if not 0.0 <= value < 1.0:
        raise ConfigError(f"{where} must be in [0, 1), not {value}")
    return float(value)


def read_config(path: Path) -> Config:
    """Reads and checks the TOML file of a case.

    Ids of the model (dose target, covariates, observables) are checked against
    the model when the data is generated, see `hybrid.data.simulate.panel_model`.

    Raises:
        ConfigError: A table or key is missing or has a value which cannot be used.
    """
    path = Path(path)
    text = path.read_text()
    raw = tomllib.loads(text)
    try:
        simulation, population = raw["simulation"], raw["population"]
        network, training = raw["network"], raw["training"]
        times = tuple(float(time) for time in simulation["times"])
        if len(times) < 2 or any(b <= a for a, b in itertools.pairwise(times)):
            raise ConfigError("[simulation] times must be at least two increasing values")
        observables = {
            name: Observable(proportional=float(table["proportional"]), additive=float(table["additive"]))
            for name, table in raw["observables"].items()
        }
        if not observables:
            raise ConfigError("[observables] needs at least one observable")
        if any(
            o.proportional < 0 or o.additive < 0 or o.proportional + o.additive == 0
            for o in observables.values()
        ):
            raise ConfigError("[observables] noise must be non-negative and not zero")
        if network["activation"] not in ACTIVATIONS:
            raise ConfigError(f"[network] activation must be one of {ACTIVATIONS}")
        config = Config(
            name=str(raw["name"]),
            seed=int(raw["seed"]),
            model=_model(raw["model"], str(raw["name"]), path.parent),
            times=times,
            simulation_batch_size=int(simulation.get("batch_size", 64)),
            rtol=float(simulation.get("rtol", 1e-8)),
            atol=float(simulation.get("atol", 1e-10)),
            dose_target=str(raw["dose"]["target"]),
            dose=_distribution(raw["dose"], "dose"),
            covariates={
                name: _distribution(table, f"covariates.{name}")
                for name, table in raw.get("covariates", {}).items()
            },
            observables=observables,
            n_subjects=int(population["n_subjects"]),
            validation_fraction=_fraction(
                population["validation_fraction"], "[population] validation_fraction"
            ),
            missing_fraction=_fraction(
                population.get("missing_fraction", 0.0), "[population] missing_fraction"
            ),
            network=NetworkConfig(
                width=int(network["width"]),
                depth=int(network["depth"]),
                activation=str(network["activation"]),
            ),
            training=TrainingConfig(
                batch_size=int(training["batch_size"]),
                adam_steps=int(training["adam_steps"]),
                learning_rate=float(training["learning_rate"]),
                lbfgs_steps=int(training["lbfgs_steps"]),
                eval_every=int(training["eval_every"]),
            ),
            text=text,
        )
    except KeyError as error:
        raise ConfigError(f"{path.name}: missing {error}") from error
    if config.n_subjects < 2 or config.simulation_batch_size < 1:
        raise ConfigError("[population] n_subjects must be at least 2 and [simulation] batch_size at least 1")
    if config.dose_target in config.covariates:
        raise ConfigError(f"{config.dose_target!r} is the dose target and cannot be a covariate")
    return config
