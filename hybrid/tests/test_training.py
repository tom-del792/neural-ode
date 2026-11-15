"""The loss, the optimization and the saved weights."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid.nn.mlp import mlp
from hybrid.nn.scaling import Scaler
from hybrid.training.checkpoint import load_weights, save_weights
from hybrid.training.fit import FitSettings, fit
from hybrid.training.loss import ErrorModel, gaussian_nll

SETTINGS = FitSettings(batch_size=16, adam_steps=300, learning_rate=1e-2, lbfgs_steps=30, eval_every=50)


def test_gaussian_nll() -> None:
    prediction, observation = jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([1.5, 2.0, 100.0])
    sigma, mask = jnp.asarray([0.5, 2.0, 1.0]), jnp.asarray([True, True, False])
    expected = np.mean([np.log(0.5) + 0.5 * 1.0**2, np.log(2.0)]) + 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(gaussian_nll(prediction, observation, sigma, mask), expected, rtol=1e-12)
    # values without a measurement do not change the loss or its gradient
    grad = jax.grad(gaussian_nll)(prediction, observation, sigma, mask)
    assert grad[2] == 0 and jnp.all(jnp.isfinite(grad))


def test_error_model() -> None:
    error = ErrorModel.create(np.asarray([2.0, 10.0]), additive=1.0, proportional=0.05)
    np.testing.assert_allclose(error.additive, [2.0, 10.0], rtol=1e-12)
    np.testing.assert_allclose(error.proportional, [0.05, 0.05], rtol=1e-12)
    np.testing.assert_allclose(error.sigma(jnp.asarray([[-4.0, 20.0]])), [[2.2, 11.0]], rtol=1e-12)


class Regression(eqx.Module):
    """A network with a scaler and an error model, as in the warm-up."""

    scaler: Scaler
    network: eqx.nn.MLP
    error: ErrorModel


def _regression() -> tuple[Regression, tuple, tuple, callable]:
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, (160, 1))
    clean = np.sin(2.0 * x) + 3.0
    y = clean + (0.05 + 0.02 * clean) * rng.standard_normal(clean.shape)
    mask = np.ones_like(y, dtype=bool)
    model = Regression(
        scaler=Scaler.fit(x),
        network=mlp(1, 1, 16, 2, "tanh", jax.random.PRNGKey(0)),
        error=ErrorModel.create(np.asarray([y.std()])),
    )

    def loss(m: Regression, x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
        prediction = jax.vmap(lambda row: m.network(m.scaler(row)))(x) + 3.0
        return gaussian_nll(prediction, y, m.error.sigma(prediction), mask)

    return model, (x[:120], y[:120], mask[:120]), (x[120:], y[120:], mask[120:]), loss


def test_fit() -> None:
    model, train, valid, loss = _regression()
    fitted, log = fit(model, loss, train, valid, SETTINGS, np.random.default_rng(0))
    assert [entry["phase"] for entry in log[:2]] == ["adam", "adam"] and log[-1]["phase"] == "lbfgs"
    assert log[0]["step"] == 0 and log[-1]["step"] == SETTINGS.adam_steps + SETTINGS.lbfgs_steps
    # the returned model is the one with the lowest validation loss
    best = min(entry["valid"] for entry in log)
    np.testing.assert_allclose(loss(fitted, *valid), best, rtol=1e-9)
    assert best < log[0]["valid"] - 1.0
    # the function is learned, and the noise is found: sigma at y = 3 is 0.05 + 0.02 * 3
    x = np.linspace(-1.8, 1.8, 50)[:, None]
    prediction = np.asarray(jax.vmap(lambda row: fitted.network(fitted.scaler(row)))(jnp.asarray(x))) + 3.0
    np.testing.assert_allclose(prediction, np.sin(2.0 * x) + 3.0, atol=0.1)
    np.testing.assert_allclose(fitted.error.sigma(jnp.asarray([3.0])), 0.11, rtol=0.3)
    # constants are not trained
    np.testing.assert_array_equal(fitted.scaler.shift, model.scaler.shift)
    np.testing.assert_array_equal(fitted.error.scale, model.error.scale)


def test_fit_is_reproducible() -> None:
    model, train, valid, loss = _regression()
    settings = FitSettings(batch_size=16, adam_steps=40, learning_rate=1e-2, lbfgs_steps=3, eval_every=20)
    _, first = fit(model, loss, train, valid, settings, np.random.default_rng(1))
    _, second = fit(model, loss, train, valid, settings, np.random.default_rng(1))
    assert first == second


def test_loss_which_is_not_finite() -> None:
    model, train, valid, _ = _regression()

    def loss(m: Regression, *_: jax.Array) -> jax.Array:
        return jnp.sum(m.network.layers[0].weight) * jnp.nan

    with pytest.raises(FloatingPointError, match="not finite"):
        fit(model, loss, train, valid, SETTINGS, np.random.default_rng(0))


def test_weights(tmp_path: Path) -> None:
    model, _, valid, loss = _regression()
    path = tmp_path / "run" / "weights.eqx"
    save_weights(path, model)
    other = Regression(
        scaler=Scaler.fit(np.asarray([[0.0], [1.0]])),
        network=mlp(1, 1, 16, 2, "tanh", jax.random.PRNGKey(5)),
        error=ErrorModel.create(np.asarray([1.0])),
    )
    loaded = load_weights(path, other)
    np.testing.assert_array_equal(loaded.scaler.scale, model.scaler.scale)
    np.testing.assert_array_equal(loaded.error.scale, model.error.scale)
    np.testing.assert_allclose(loss(loaded, *valid), loss(model, *valid), rtol=1e-12)
