"""Simulation of the panel with the mechanistic model, and the measurement noise."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from sbml2jax.model import JaxModel, Segment, SolverSettings

from hybrid.config import Config, ConfigError


def panel_model(config: Config) -> JaxModel:
    """The mechanistic model of a case, with its ids checked against the configuration.

    Raises:
        ConfigError: A covariate is not an input, an observable is not an output,
            or the dose target is not a state or constant of the model.
    """
    model = JaxModel.from_sbml(config.model.sbml_path())
    module = model.module
    unknown = [name for name in config.covariates if name not in module.INPUTS]
    if unknown:
        raise ConfigError(f"covariates {unknown} are not inputs of the model")
    unknown = [name for name in config.observables if name not in module.OUTPUTS]
    if unknown:
        raise ConfigError(f"observables {unknown} are not outputs of the model")
    if config.dose_target not in (*module.STATES, *module.CONSTANTS):
        raise ConfigError(f"dose target {config.dose_target!r} is not a state or constant of the model")
    return model


def subject_simulator(model: JaxModel, config: Config) -> Callable[[jax.Array], jax.Array]:
    """The simulation of one patient as a function of its row of the population.

    The function maps the dose and the covariates `(len(config.subject_inputs),)`
    to the observables at the sampling times `(n_times, n_observables)`. It can be
    traced, so it is batched with `jax.vmap` and compiled with `jax.jit`.
    """
    module = model.module
    ts = np.asarray(config.times)
    covariates = jnp.asarray([module.INPUTS.index(name) for name in config.covariates], dtype=int)
    observables = np.asarray([module.OUTPUTS.index(name) for name in config.observables])
    settings = SolverSettings(rtol=config.rtol, atol=config.atol)

    def simulate(x: jax.Array) -> jax.Array:
        p = model.p.at[covariates].set(x[1:]) if covariates.size else model.p
        subject = JaxModel(p=p, module=module)
        segments = [Segment(end=float(ts[-1]), changes={config.dose_target: x[0]})]
        return subject.simulate(ts, segments, settings)[:, observables]

    return simulate


def simulate_panel(model: JaxModel, config: Config, x: np.ndarray) -> np.ndarray:
    """The observables of every patient without noise.

    The patients are simulated in batches of `config.simulation_batch_size` with
    one compiled function; the last batch is filled up, so it is compiled once.

    Returns:
        Array of shape `(n_subjects, n_times, n_observables)`.
    """
    batch = jax.jit(jax.vmap(subject_simulator(model, config)))
    size = min(config.simulation_batch_size, len(x))
    ys = []
    for start in range(0, len(x), size):
        rows = x[start : start + size]
        filled = np.concatenate([rows, np.repeat(rows[-1:], size - len(rows), axis=0)])
        ys.append(np.asarray(batch(jnp.asarray(filled)))[: len(rows)])
    y = np.concatenate(ys)
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("the simulation of the panel gave values which are not finite")
    return y


def add_noise(y_clean: np.ndarray, config: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Measurements of the simulated observables.

    Normal noise with `sigma = additive + proportional * value` per observable;
    `config.missing_fraction` of the measurements are dropped at random.

    Returns:
        The measurements and the mask of the measurements which exist, both of
        the shape of `y_clean`.
    """
    proportional = np.asarray([o.proportional for o in config.observables.values()])
    additive = np.asarray([o.additive for o in config.observables.values()])
    sigma = additive + proportional * np.abs(y_clean)
    y = y_clean + sigma * rng.standard_normal(y_clean.shape)
    mask = rng.uniform(size=y_clean.shape) >= config.missing_fraction
    return y, mask
