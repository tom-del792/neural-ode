"""Shared fixtures: the cases of `configs/`, and a small case which trains in seconds."""

import re
from pathlib import Path

import pytest

from hybrid.config import Config, read_config
from hybrid.data.dataset import Panel, generate_panel
from hybrid.runs import RUNS_ENV

CONFIGS = Path(__file__).parents[1] / "configs"
ABSORPTION_SBML = Path(__file__).parents[2] / "docs" / "examples" / "absorption.xml"


def write_case(directory: Path, source: str = "warmup_absorption.toml", **values: object) -> Path:
    """Copies a case of `configs/` with other values of keys (every `key = value` line of that key)."""
    text = (CONFIGS / source).read_text()
    text = re.sub(r"^path = .*$", f'path = "{ABSORPTION_SBML}"', text, flags=re.MULTILINE)
    for key, value in values.items():
        text, count = re.subn(rf"^{key} = [^#\n]*", f"{key} = {value} ", text, flags=re.MULTILINE)
        assert count >= 1, key
    path = directory / source
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test writes its generated files into its own directory."""
    directory = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ENV, str(directory))
    return directory


@pytest.fixture(scope="session")
def small_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The absorption case with few patients and steps."""
    return write_case(
        tmp_path_factory.mktemp("case"),
        n_subjects=48,
        adam_steps=300,
        lbfgs_steps=20,
        eval_every=50,
        batch_size=16,
    )


@pytest.fixture(scope="session")
def small_config(small_case: Path) -> Config:
    return read_config(small_case)


@pytest.fixture(scope="session")
def small_panel(small_config: Config) -> Panel:
    return generate_panel(small_config)
