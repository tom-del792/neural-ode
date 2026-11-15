"""The warm-up from the configuration to the evaluated run, and where its files go."""

import json
from pathlib import Path

import numpy as np
import pytest

from hybrid import warmup
from hybrid.cli import main
from hybrid.config import Config
from hybrid.runs import commit, dataset_path, runs_dir


@pytest.fixture
def run(small_config: Config) -> Path:
    return warmup.train(small_config)


def test_runs_directory(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert runs_dir() == runs
    monkeypatch.delenv("HYBRID_RUNS")
    assert runs_dir() == Path.home() / ".cache" / "hybrid" / "runs"


def test_commit() -> None:
    head, dirty = commit()
    assert len(head) == 40 and isinstance(dirty, bool)


def test_dataset_is_generated_once(small_config: Config, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    first = warmup.dataset(small_config)
    assert dataset_path(small_config).is_file() and "generating" in caplog.text
    caplog.clear()
    second = warmup.dataset(small_config)
    assert "generating" not in caplog.text
    np.testing.assert_array_equal(first.y, second.y)


def test_run(run: Path, runs: Path, small_config: Config) -> None:
    """A run holds everything which produced it and what it produced."""
    assert run.parent == runs / small_config.name
    files = {path.name for path in run.iterdir()}
    assert files == {
        "config.toml",
        "run.json",
        "weights.eqx",
        "log.csv",
        "metrics.json",
        "fits.png",
        "parity.png",
        "learning.png",
    }
    assert (run / "config.toml").read_text() == small_config.text
    record = json.loads((run / "run.json").read_text())
    assert run.name.endswith(record["commit"][:7]) and record["seed"] == small_config.seed
    assert (runs / record["dataset"]).is_file()
    assert set(record["packages"]) >= {"hybrid", "sbml2jax", "jax", "equinox", "optax"}


def test_network_learns_the_panel(run: Path) -> None:
    """Validation patients, which the training has not seen, are predicted."""
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["n_train"] == 36 and metrics["n_valid"] == 12
    for part in ("train", "valid"):
        for name in ("C", "E"):
            assert metrics["observables"][part][name]["r2_truth"] > 0.9, (part, name)
    assert set(metrics["error_model"]["C"]) == {
        "proportional",
        "proportional_data",
        "additive",
        "additive_data",
    }


def test_evaluate_from_saved_weights(run: Path) -> None:
    """The weights file is the trained model: the evaluation of a run can be repeated."""
    first = json.loads((run / "metrics.json").read_text())
    for name in ("metrics.json", "fits.png"):
        (run / name).unlink()
    assert warmup.evaluate(run) == first
    assert (run / "fits.png").is_file()


def test_command_line(small_case: Path, runs: Path, capsys: pytest.CaptureFixture) -> None:
    main(["generate", str(small_case)])
    assert len(list((runs / "datasets").glob("*/dataset.npz"))) == 1
    main(["train", str(small_case)])
    assert "r2_truth" in capsys.readouterr().out
    (run,) = (runs / "warmup_absorption").iterdir()
    main(["evaluate", str(run)])
    assert "r2_truth" in capsys.readouterr().out
