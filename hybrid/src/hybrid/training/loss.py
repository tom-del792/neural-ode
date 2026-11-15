"""The loss: negative log-likelihood of the measurements under a normal error model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def gaussian_nll(
    prediction: jax.Array, observation: jax.Array, sigma: jax.Array, mask: jax.Array
) -> jax.Array:
    """Mean negative log-likelihood of the measurements which exist.

    Args:
        prediction: Predicted values.
        observation: Measurements, of the shape of `prediction`; values without
            a measurement are ignored and may be anything finite.
        sigma: Standard deviation of every measurement, of the same shape.
        mask: Which measurements exist, of the same shape.
    """
    residual = jnp.where(mask, (observation - prediction) / sigma, 0.0)
    terms = jnp.log(sigma) + 0.5 * residual**2 + 0.5 * jnp.log(2 * jnp.pi)
    return jnp.sum(jnp.where(mask, terms, 0.0)) / jnp.sum(mask)


def _inverse_softplus(value: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(value))


class ErrorModel(eqx.Module):
    """The combined error model `sigma = additive + proportional * |prediction|` per observable.

    Both parts are trained with the network, as positive numbers through a
    softplus. The additive part is in units of `scale`, the scale of each
    observable, so the same initial value fits every observable.

    Attributes:
        raw_additive: Additive part before the softplus, in units of `scale`.
        raw_proportional: Proportional part before the softplus.
        scale: Scale of every observable (not trained).
    """

    raw_additive: jax.Array
    raw_proportional: jax.Array
    scale: np.ndarray

    @classmethod
    def create(cls, scale: np.ndarray, additive: float = 1.0, proportional: float = 0.05) -> "ErrorModel":
        """An error model which starts as a constant sigma of about the scale of the observable.

        The loss is then close to a mean squared error of the scaled values at
        first, which keeps the start of the training stable.
        """
        ones = np.ones_like(np.asarray(scale, dtype=float))
        return cls(
            raw_additive=jnp.asarray(_inverse_softplus(additive * ones)),
            raw_proportional=jnp.asarray(_inverse_softplus(proportional * ones)),
            scale=np.asarray(scale, dtype=float),
        )

    @property
    def additive(self) -> jax.Array:
        """Additive part in the units of the observables."""
        return self.scale * jax.nn.softplus(self.raw_additive)

    @property
    def proportional(self) -> jax.Array:
        """Proportional part."""
        return jax.nn.softplus(self.raw_proportional)

    def sigma(self, prediction: jax.Array) -> jax.Array:
        """Standard deviation of the measurements of predictions `(..., n_observables)`."""
        return self.additive + self.proportional * jnp.abs(prediction)
