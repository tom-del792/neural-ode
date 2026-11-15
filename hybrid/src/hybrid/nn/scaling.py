"""Scaling of the inputs and outputs of a network, as part of the network.

The constants are fitted to the training data once and saved with the weights,
so a loaded network needs no data. They are numpy arrays, which the training
leaves unchanged (it only updates JAX arrays).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class Scaler(eqx.Module):
    """Standardizes a vector, some columns after a logarithm.

    Attributes:
        log: Which columns are transformed with the logarithm first.
        shift: Mean of the (transformed) training values.
        scale: Standard deviation of the (transformed) training values.
    """

    log: np.ndarray
    shift: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, log: tuple[bool, ...] | None = None) -> "Scaler":
        """The scaler of training values `(n, n_columns)`; a constant column keeps scale 1."""
        log_columns = np.zeros(values.shape[1], dtype=bool) if log is None else np.asarray(log, dtype=bool)
        if np.any(values[:, log_columns] <= 0):
            raise ValueError("a column with logarithm has values which are not positive")
        transformed = np.where(log_columns, np.log(np.where(log_columns, values, 1.0)), values)
        scale = transformed.std(axis=0)
        return cls(log=log_columns, shift=transformed.mean(axis=0), scale=np.where(scale > 0, scale, 1.0))

    def __call__(self, values: jax.Array) -> jax.Array:
        """Scales a vector `(n_columns,)`."""
        transformed = jnp.where(self.log, jnp.log(jnp.where(self.log, values, 1.0)), values)
        return (transformed - self.shift) / self.scale

    def inverse(self, scaled: jax.Array) -> jax.Array:
        """The values of a scaled vector."""
        transformed = scaled * self.scale + self.shift
        return jnp.where(self.log, jnp.exp(jnp.where(self.log, transformed, 0.0)), transformed)
