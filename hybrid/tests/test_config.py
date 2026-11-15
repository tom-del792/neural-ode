"""The configuration of a case."""

from pathlib import Path

import pytest

from hybrid.config import ConfigError, read_config

from .conftest import CONFIGS, write_case


@pytest.mark.parametrize("name", ["warmup_apixaban.toml", "warmup_absorption.toml"])
def test_cases_of_the_repository(name: str) -> None:
    config = read_config(CONFIGS / name)
    assert config.name == Path(name).stem
    assert config.subject_inputs == (config.dose_target, *config.covariates)
    assert len(config.subject_distributions) == len(config.subject_inputs)
    assert config.observables


def test_absorption_case() -> None:
    config = read_config(CONFIGS / "warmup_absorption.toml")
    assert config.subject_inputs == ("Agut", "ka", "CL")
    assert tuple(config.observables) == ("C", "E")
    assert config.model.sbml_path().name == "absorption.xml"
    # doses and loguniform covariates are positive and seen as logarithm by the network
    assert [d.log for d in config.subject_distributions] == [True, True, True]


def test_apixaban_case_is_pinned() -> None:
    model = read_config(CONFIGS / "warmup_apixaban.toml").model
    assert model.path is None and model.name == "apixaban"
    assert len(model.ref) == 40 and len(model.sha256) == 64


def test_data_hash(tmp_path: Path) -> None:
    base = read_config(write_case(tmp_path)).data_hash()
    # the training does not change the data, the seed and the population do
    assert read_config(write_case(tmp_path, adam_steps=7, width=8)).data_hash() == base
    assert read_config(write_case(tmp_path, seed=1)).data_hash() != base
    assert read_config(write_case(tmp_path, n_subjects=5)).data_hash() != base


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"times": "[0, 10, 10]"}, "increasing"),
        ({"validation_fraction": 1.0}, "validation_fraction"),
        ({"activation": '"sine"'}, "activation"),
        ({"n_subjects": 1}, "n_subjects"),
        ({"target": '"ka"'}, "dose target"),
        ({"values": "[]"}, "values"),
    ],
)
def test_errors(tmp_path: Path, values: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        read_config(write_case(tmp_path, **values))


def test_missing_key(tmp_path: Path) -> None:
    path = write_case(tmp_path)
    path.write_text(path.read_text().replace("adam_steps", "steps"))
    with pytest.raises(ConfigError, match="adam_steps"):
        read_config(path)


def test_missing_model_file(tmp_path: Path) -> None:
    path = write_case(tmp_path)
    path.write_text(path.read_text().replace("absorption.xml", "nothing.xml"))
    with pytest.raises(ConfigError, match="does not exist"):
        read_config(path).model.sbml_path()
