"""The networks and their scaling."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid.config import NetworkConfig
from hybrid.data.dataset import Panel
from hybrid.nn.mlp import ACTIVATIONS, mlp
from hybrid.nn.panel import PanelNetwork
from hybrid.nn.scaling import Scaler
from hybrid.training.fit import is_trainable


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_mlp(activation: str) -> None:
    network = mlp(3, 2, width=8, depth=2, activation=activation, key=jax.random.PRNGKey(0))
    y = network(jnp.asarray([0.1, -0.2, 0.3]))
    assert y.shape == (2,) and y.dtype == jnp.float64
    assert len(network.layers) == 3


def test_zero_last_layer() -> None:
    """A network with a zero last layer gives zero for every input, but can be trained."""
    network = mlp(3, 2, width=8, depth=2, activation="tanh", key=jax.random.PRNGKey(0), zero_last=True)
    x = jnp.asarray([0.5, -1.0, 2.0])
    np.testing.assert_array_equal(network(x), np.zeros(2))
    grad = eqx.filter_grad(lambda n: jnp.sum(n(x)))(network)
    assert jnp.any(grad.layers[-1].weight != 0)


def test_scaler() -> None:
    rng = np.random.default_rng(0)
    values = np.stack(
        [rng.normal(5.0, 2.0, 500), np.exp(rng.normal(1.0, 0.5, 500)), np.full(500, 3.0)], axis=1
    )
    scaler = Scaler.fit(values, (False, True, False))
    scaled = np.asarray(jax.vmap(scaler)(jnp.asarray(values)))
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), [1.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(jax.vmap(scaler.inverse)(jnp.asarray(scaled)), values, rtol=1e-12)
    with pytest.raises(ValueError, match="not positive"):
        Scaler.fit(np.asarray([[1.0], [-1.0]]), (True,))


def test_scaler_has_finite_gradients() -> None:
    scaler = Scaler.fit(np.asarray([[1.0, -2.0], [3.0, 5.0]]), (True, False))
    grad = jax.jacobian(scaler)(jnp.asarray([2.0, -1.0]))
    assert jnp.all(jnp.isfinite(grad))


def test_panel_network(small_panel: Panel) -> None:
    network = PanelNetwork.create(
        small_panel, NetworkConfig(width=8, depth=2, activation="tanh"), jax.random.PRNGKey(0)
    )
    times, x = jnp.asarray(small_panel.times), jnp.asarray(small_panel.x[:5])
    prediction = network.predict(times, x)
    assert prediction.shape == (5, len(times), 2)
    np.testing.assert_allclose(prediction[3, 4], network(times[4], x[3]), rtol=1e-12)
    # scaled to the training patients: the scaled inputs are standardized
    train = small_panel.subset(train=True)
    assert network.time_step == 5.0
    np.testing.assert_allclose(network.scale_in.shift[2:], np.log(train.x).mean(axis=0), rtol=1e-12)
    measured = np.where(train.mask, train.y, np.nan)
    np.testing.assert_allclose(network.scale_out.shift, np.nanmean(measured, axis=(0, 1)), rtol=1e-12)
    np.testing.assert_allclose(network.scale_out.scale, np.nanstd(measured, axis=(0, 1)), rtol=1e-12)


def test_scaling_is_not_trained(small_panel: Panel) -> None:
    network = PanelNetwork.create(
        small_panel, NetworkConfig(width=8, depth=2, activation="tanh"), jax.random.PRNGKey(0)
    )
    params, static = eqx.partition(network, is_trainable)
    assert params.scale_in.shift is None and params.scale_out.scale is None and params.time_step is None
    assert static.network.layers[0].weight is None
    assert params.network.layers[0].weight is not None
