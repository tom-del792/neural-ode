"""The network of the warm-up: time and patient to the observables."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrid.config import NetworkConfig
from hybrid.data.dataset import Panel
from hybrid.nn.mlp import mlp
from hybrid.nn.scaling import Scaler


class PanelNetwork(eqx.Module):
    """Maps a time and the inputs of a patient to all observables.

    The time enters twice, linear and as `log1p(time / time_step)`, because a
    time course changes fast after the dose and slowly later.

    Attributes:
        time_step: The smallest sampling interval, the unit of the logarithmic time.
        scale_in: Scaling of the two time features and the inputs of the patient.
        network: The multilayer perceptron on the scaled features.
        scale_out: Scaling of the observables; the network predicts scaled values.
    """

    time_step: float
    scale_in: Scaler
    network: eqx.nn.MLP
    scale_out: Scaler

    @classmethod
    def create(cls, panel: Panel, config: NetworkConfig, key: jax.Array) -> "PanelNetwork":
        """A network with random weights, scaled to the training patients of the panel."""
        train = panel.subset(train=True)
        time_step = float(np.min(np.diff(panel.times)))
        features = np.concatenate(
            [
                np.tile(_time_features(panel.times, time_step), (len(train.x), 1)),
                np.repeat(train.x, len(panel.times), axis=0),
            ],
            axis=1,
        )
        n_out = len(panel.y_names)
        return cls(
            time_step=time_step,
            scale_in=Scaler.fit(features, (False, False, *panel.log_x)),
            network=mlp(features.shape[1], n_out, config.width, config.depth, config.activation, key),
            scale_out=_output_scaler(train),
        )

    def __call__(self, time: jax.Array, x: jax.Array) -> jax.Array:
        """The observables `(n_observables,)` of a patient with the inputs `x` at one time."""
        features = jnp.concatenate([jnp.stack([time, jnp.log1p(time / self.time_step)]), x])
        return self.scale_out.inverse(self.network(self.scale_in(features)))

    def predict(self, times: jax.Array, x: jax.Array) -> jax.Array:
        """The time courses `(n_subjects, n_times, n_observables)` of patients `(n_subjects, n_inputs)`."""
        return jax.vmap(lambda row: jax.vmap(lambda time: self(time, row))(times))(x)


def _time_features(times: np.ndarray, time_step: float) -> np.ndarray:
    return np.stack([times, np.log1p(times / time_step)], axis=1)


def _output_scaler(train: Panel) -> Scaler:
    """Mean and standard deviation of the measurements which exist, per observable."""
    values = np.where(train.mask, train.y, np.nan).reshape(-1, len(train.y_names))
    scale = np.nanstd(values, axis=0)
    return Scaler(
        log=np.zeros(values.shape[1], dtype=bool),
        shift=np.nanmean(values, axis=0),
        scale=np.where(scale > 0, scale, 1.0),
    )
