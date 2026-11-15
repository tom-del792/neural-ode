"""A multilayer perceptron with several inputs and outputs."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
}


def mlp(
    n_in: int, n_out: int, width: int, depth: int, activation: str, key: jax.Array, zero_last: bool = False
) -> eqx.nn.MLP:
    """A network of `depth` hidden layers of `width` units, vector to vector.

    Args:
        n_in: Number of inputs.
        n_out: Number of outputs.
        width: Units of a hidden layer.
        depth: Number of hidden layers.
        activation: Name of the activation, a key of `ACTIVATIONS`.
        key: Random key of the initial weights.
        zero_last: Sets the weights and biases of the last layer to zero, so
            the network gives zero for every input at initialization. A hybrid
            model with such a network starts as the mechanistic model.
    """
    network = eqx.nn.MLP(n_in, n_out, width, depth, activation=ACTIVATIONS[activation], key=key)
    if zero_last:
        last = network.layers[-1]
        network = eqx.tree_at(
            lambda n: (n.layers[-1].weight, n.layers[-1].bias),
            network,
            (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
        )
    return network
