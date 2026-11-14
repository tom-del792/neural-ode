"""The package imports and builds on sbml2jax."""

import jax
import jax.numpy as jnp

import hybrid


def test_version() -> None:
    assert hybrid.__version__ == "0.1.0"


def test_double_precision() -> None:
    assert jax.config.jax_enable_x64
    assert jnp.asarray(1.0).dtype == jnp.float64
