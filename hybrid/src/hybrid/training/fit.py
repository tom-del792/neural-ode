"""The optimization: Adam on batches of patients, then L-BFGS on all training patients.

`fit` knows nothing about the model: it trains the JAX arrays of any pytree
(an equinox module, or a tuple of modules) on a loss of the model and the data
of a batch of patients.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

logger = logging.getLogger(__name__)

Model = TypeVar("Model")
#: loss of a model on the data of some patients: `loss(model, *arrays) -> scalar`
Loss = Callable[..., jax.Array]


@dataclass(frozen=True)
class FitSettings:
    """Settings of `fit`.

    Attributes:
        batch_size: Patients per Adam step.
        adam_steps: Number of Adam steps; the learning rate decays to 1 % on a cosine.
        learning_rate: Initial learning rate of Adam.
        lbfgs_steps: Number of L-BFGS steps on all training patients, after Adam.
        eval_every: Adam steps between two evaluations on all patients.
        clip: Maximal global norm of an Adam gradient.
    """

    batch_size: int
    adam_steps: int
    learning_rate: float
    lbfgs_steps: int
    eval_every: int
    clip: float = 1.0


def is_trainable(leaf: Any) -> bool:
    """Whether a leaf of a model is trained: floating point JAX arrays, not numpy constants."""
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)


def fit(
    model: Model,
    loss: Loss,
    train: tuple[np.ndarray, ...],
    valid: tuple[np.ndarray, ...],
    settings: FitSettings,
    rng: np.random.Generator,
) -> tuple[Model, list[dict[str, float | int | str]]]:
    """Trains a model and returns it at its lowest validation loss.

    Args:
        model: Pytree whose floating point JAX arrays are trained.
        loss: Loss of the model on the arrays of some patients.
        train: Arrays of the training patients, the patients on the first axis.
        valid: Arrays of the validation patients, only used to choose the returned model.
        settings: Settings of the optimization.
        rng: Random numbers of the batches.

    Returns:
        The model with the lowest validation loss of all evaluations, and the
        log: one entry per evaluation with `step`, `phase` (`adam` or `lbfgs`),
        `train` and `valid` loss.
    """
    params, static = eqx.partition(model, is_trainable)
    train_arrays = tuple(jnp.asarray(array) for array in train)
    valid_arrays = tuple(jnp.asarray(array) for array in valid)

    def objective(p: Model, *arrays: jax.Array) -> jax.Array:
        return loss(eqx.combine(p, static), *arrays)

    evaluate = jax.jit(objective)
    log: list[dict[str, float | int | str]] = []
    best = (np.inf, params)

    def record(step: int, phase: str, p: Model) -> None:
        nonlocal best
        entry = {
            "step": step,
            "phase": phase,
            "train": float(evaluate(p, *train_arrays)),
            "valid": float(evaluate(p, *valid_arrays)),
        }
        if not np.isfinite(entry["train"]):
            raise FloatingPointError(f"the training loss is not finite at step {step} ({phase})")
        log.append(entry)
        logger.info("%-5s step %5d  train %.4f  valid %.4f", phase, step, entry["train"], entry["valid"])
        if entry["valid"] < best[0]:
            best = (entry["valid"], p)

    # Adam on batches of patients
    schedule = optax.cosine_decay_schedule(settings.learning_rate, max(settings.adam_steps, 1), alpha=0.01)
    adam = optax.chain(optax.clip_by_global_norm(settings.clip), optax.adam(schedule))

    @jax.jit
    def adam_step(p: Model, state: optax.OptState, *arrays: jax.Array) -> tuple[Model, optax.OptState]:
        updates, state = adam.update(jax.grad(objective)(p, *arrays), state, p)
        return optax.apply_updates(p, updates), state

    n = len(train_arrays[0])
    size = min(settings.batch_size, n)
    state = adam.init(params)
    record(0, "adam", params)
    order = rng.permutation(n)
    for step in range(1, settings.adam_steps + 1):
        if len(order) < size:
            order = rng.permutation(n)
        rows, order = order[:size], order[size:]
        params, state = adam_step(params, state, *(array[rows] for array in train_arrays))
        if step % settings.eval_every == 0 or step == settings.adam_steps:
            record(step, "adam", params)

    # L-BFGS on all training patients, from the best model of Adam
    if settings.lbfgs_steps > 0:
        params = best[1]
        lbfgs = optax.lbfgs()

        def full(p: Model) -> jax.Array:
            return objective(p, *train_arrays)

        value_and_grad = optax.value_and_grad_from_state(full)

        @jax.jit
        def lbfgs_step(p: Model, state: optax.OptState) -> tuple[Model, optax.OptState]:
            value, grad = value_and_grad(p, state=state)
            updates, state = lbfgs.update(grad, state, p, value=value, grad=grad, value_fn=full)
            return optax.apply_updates(p, updates), state

        state = lbfgs.init(params)
        for step in range(1, settings.lbfgs_steps + 1):
            params, state = lbfgs_step(params, state)
            if step % max(settings.eval_every // 10, 1) == 0 or step == settings.lbfgs_steps:
                record(settings.adam_steps + step, "lbfgs", params)

    return eqx.combine(best[1], static), log
