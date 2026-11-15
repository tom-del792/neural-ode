"""The panel: many patients with one time course each, split by patient."""

from dataclasses import dataclass, replace

import numpy as np

from hybrid.config import Config
from hybrid.data.population import sample_population
from hybrid.data.simulate import add_noise, panel_model, simulate_panel

#: streams of the random numbers of a case, so that one does not shift another
STREAM_POPULATION, STREAM_NOISE, STREAM_SPLIT, STREAM_TRAINING = range(4)


def rng(config: Config, stream: int) -> np.random.Generator:
    """The random number generator of one stream of a case."""
    return np.random.default_rng([config.seed, stream])


@dataclass(frozen=True)
class Panel:
    """Measurements of a panel of patients at common sampling times.

    Attributes:
        times: Sampling times `(n_times,)`.
        x: Dose and covariates of the patients `(n_subjects, n_inputs)`.
        y: Measurements `(n_subjects, n_times, n_observables)`.
        y_clean: The simulated values without noise, of the shape of `y`.
        mask: Which measurements exist, of the shape of `y`.
        train: Which patients are used for training `(n_subjects,)`; the others validate.
        x_names: Names of the columns of `x`.
        y_names: Names of the observables.
        log_x: Which columns of `x` the network sees as logarithm.
    """

    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    mask: np.ndarray
    train: np.ndarray
    x_names: tuple[str, ...]
    y_names: tuple[str, ...]
    log_x: tuple[bool, ...]

    def subset(self, train: bool) -> "Panel":
        """The training patients, or the validation patients."""
        rows = self.train if train else ~self.train
        return replace(
            self,
            x=self.x[rows],
            y=self.y[rows],
            y_clean=self.y_clean[rows],
            mask=self.mask[rows],
            train=self.train[rows],
        )


def split_subjects(n: int, validation_fraction: float, generator: np.random.Generator) -> np.ndarray:
    """A random split by patient: `True` for training.

    At least one patient is in each part, so no patient contributes
    measurements to both.
    """
    n_validation = min(max(round(n * validation_fraction), 1), n - 1)
    train = np.ones(n, dtype=bool)
    train[generator.permutation(n)[:n_validation]] = False
    return train


def generate_panel(config: Config) -> Panel:
    """Samples the patients of a case, simulates them and adds the noise."""
    x = sample_population(config, rng(config, STREAM_POPULATION))
    y_clean = simulate_panel(panel_model(config), config, x)
    y, mask = add_noise(y_clean, config, rng(config, STREAM_NOISE))
    return Panel(
        times=np.asarray(config.times),
        x=x,
        y=y,
        y_clean=y_clean,
        mask=mask,
        train=split_subjects(config.n_subjects, config.validation_fraction, rng(config, STREAM_SPLIT)),
        x_names=config.subject_inputs,
        y_names=tuple(config.observables),
        log_x=tuple(distribution.log for distribution in config.subject_distributions),
    )
