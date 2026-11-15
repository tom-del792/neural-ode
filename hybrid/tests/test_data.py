"""Data generation: population, simulation of the panel, noise, split and files."""

from pathlib import Path

import numpy as np
import pytest
from sbml2jax.model import JaxModel, Segment, SolverSettings

from hybrid.config import Config, ConfigError, Distribution, read_config
from hybrid.data.dataset import Panel, generate_panel, split_subjects
from hybrid.data.io import load_panel, save_panel
from hybrid.data.population import sample, sample_population
from hybrid.data.simulate import add_noise, panel_model, simulate_panel

from .conftest import write_case


def test_sample_within_bounds() -> None:
    rng = np.random.default_rng(0)
    uniform = sample(Distribution("uniform", low=2.0, high=3.0), 1000, rng)
    loguniform = sample(Distribution("loguniform", low=0.01, high=1.0), 1000, rng)
    choice = sample(Distribution("choice", values=(1.0, 5.0)), 1000, rng)
    assert 2.0 <= uniform.min() and uniform.max() <= 3.0
    assert 0.01 <= loguniform.min() and loguniform.max() <= 1.0
    # loguniform: half of the values below the geometric mean
    assert abs(np.mean(loguniform < 0.1) - 0.5) < 0.05
    assert set(choice) == {1.0, 5.0}


def test_population_is_reproducible(small_config: Config) -> None:
    first = sample_population(small_config, np.random.default_rng(3))
    second = sample_population(small_config, np.random.default_rng(3))
    assert first.shape == (small_config.n_subjects, 3)
    np.testing.assert_array_equal(first, second)


def test_panel_equals_single_simulations(small_config: Config, small_panel: Panel) -> None:
    """The batched panel is the simulation of each patient on its own."""
    model = JaxModel.from_sbml(small_config.model.sbml_path())
    outputs = [model.module.OUTPUTS.index(name) for name in small_panel.y_names]
    for row in (0, 17, 47):  # one patient of each batch
        dose, ka, cl = small_panel.x[row]
        ys = model.with_inputs({"ka": ka, "CL": cl}).simulate(
            small_panel.times,
            [Segment(end=float(small_panel.times[-1]), changes={"Agut": dose})],
            SolverSettings(rtol=small_config.rtol, atol=small_config.atol),
        )
        np.testing.assert_allclose(small_panel.y_clean[row], np.asarray(ys)[:, outputs], rtol=1e-6, atol=1e-9)


def test_batches_do_not_change_the_panel(tmp_path: Path, small_panel: Panel) -> None:
    config = read_config(write_case(tmp_path, n_subjects=48, batch_size=7))
    y = simulate_panel(panel_model(config), config, small_panel.x)
    np.testing.assert_allclose(y, small_panel.y_clean, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[covariates.ka]", "[covariates.kb]", "covariates"),
        ("[observables.E]", "[observables.F]", "observables"),
    ],
)
def test_ids_are_checked_against_the_model(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = write_case(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=message):
        panel_model(read_config(path))


def test_dose_target_is_checked(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dose target"):
        panel_model(read_config(write_case(tmp_path, target='"E"')))


def test_noise(small_config: Config) -> None:
    """Standard deviation of the noise is additive + proportional * value; values are dropped."""
    clean = np.full((4000, 1, 2), 2.0)
    y, mask = add_noise(clean, small_config, np.random.default_rng(0))
    sigma = [o.additive + o.proportional * 2.0 for o in small_config.observables.values()]
    np.testing.assert_allclose((y - clean).std(axis=(0, 1)), sigma, rtol=0.05)
    assert abs((~mask).mean() - small_config.missing_fraction) < 0.02


def test_split_by_subject() -> None:
    train = split_subjects(20, 0.25, np.random.default_rng(0))
    assert train.sum() == 15
    # never empty
    assert split_subjects(2, 0.0, np.random.default_rng(0)).sum() == 1
    assert split_subjects(2, 0.9, np.random.default_rng(0)).sum() == 1


def test_panel(small_config: Config, small_panel: Panel) -> None:
    n, n_times = small_config.n_subjects, len(small_config.times)
    assert small_panel.y.shape == small_panel.y_clean.shape == small_panel.mask.shape == (n, n_times, 2)
    assert small_panel.x_names == ("Agut", "ka", "CL") and small_panel.y_names == ("C", "E")
    train, valid = small_panel.subset(train=True), small_panel.subset(train=False)
    assert len(train.x) + len(valid.x) == n and len(valid.x) == 12
    assert train.train.all() and not valid.train.any()
    # the concentration starts at zero and the dose reaches the plasma
    assert np.all(small_panel.y_clean[:, 0, 0] == 0) and np.all(small_panel.y_clean[:, 3, 0] > 0)


def test_generation_is_reproducible(small_config: Config, small_panel: Panel) -> None:
    again = generate_panel(small_config)
    np.testing.assert_array_equal(again.x, small_panel.x)
    np.testing.assert_array_equal(again.y, small_panel.y)
    np.testing.assert_array_equal(again.train, small_panel.train)


def test_save_and_load(tmp_path: Path, small_panel: Panel) -> None:
    path = tmp_path / "data" / "dataset.npz"
    save_panel(path, small_panel)
    loaded = load_panel(path)
    assert loaded.x_names == small_panel.x_names and loaded.y_names == small_panel.y_names
    assert loaded.log_x == small_panel.log_x and all(isinstance(value, bool) for value in loaded.log_x)
    for name in ("times", "x", "y", "y_clean", "mask", "train"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(small_panel, name))
