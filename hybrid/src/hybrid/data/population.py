"""Virtual patients: the dose and the covariates of every patient, sampled from the configuration."""

import numpy as np

from hybrid.config import Config, Distribution


def sample(distribution: Distribution, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` values of a distribution."""
    if distribution.kind == "choice":
        return rng.choice(np.asarray(distribution.values), size=n)
    if distribution.kind == "loguniform":
        return np.exp(rng.uniform(np.log(distribution.low), np.log(distribution.high), size=n))
    return rng.uniform(distribution.low, distribution.high, size=n)


def sample_population(config: Config, rng: np.random.Generator) -> np.ndarray:
    """The patients of a case.

    Returns:
        Array of shape `(n_subjects, len(config.subject_inputs))`: the dose and
        the covariates of every patient, in the order of `config.subject_inputs`.
    """
    columns = [sample(distribution, config.n_subjects, rng) for distribution in config.subject_distributions]
    return np.stack(columns, axis=1)
