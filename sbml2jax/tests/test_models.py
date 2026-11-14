"""Work package 1 milestone: published models match roadrunner, gradients are correct.

The models are the pinned models of `examples/models/models.toml`. Every model
is downloaded at its commit (cached, checksum verified) and runs through the
same two checks; a new model is a new table in that file.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import roadrunner

from sbml2jax.model import JaxModel, Segment, SolverSettings
from sbml2jax.pinned import PinnedModel, fetch, read_pinned

from .conftest import MODELS_TOML

PINNED = read_pinned(MODELS_TOML)
SETTINGS = SolverSettings(rtol=1e-10, atol=1e-12)
#: |jax - roadrunner| <= ATOL + RTOL * |roadrunner|
RTOL, ATOL = 1e-5, 1e-9
N_TIMES = 97
#: step of the central finite difference, relative to the value of the parameter
FD_STEP = 1e-3


@dataclass(frozen=True)
class Case:
    """A pinned model with the settings of its verification."""

    pinned: PinnedModel
    sbml: Path
    model: JaxModel
    dose_id: str
    dose: float
    output: str
    parameter: str
    ts: np.ndarray

    @property
    def segments(self) -> list[Segment]:
        return [Segment(end=float(self.ts[-1]), changes={self.dose_id: self.dose})]


@pytest.fixture(scope="module", params=list(PINNED))
def case(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Case:
    name = request.param
    with MODELS_TOML.open("rb") as f_in:
        settings = tomllib.load(f_in)[name]["verification"]
    sbml = fetch(PINNED[name])
    return Case(
        pinned=PINNED[name],
        sbml=sbml,
        model=JaxModel.from_sbml(sbml, tmp_path_factory.mktemp(name) / f"{name}.py"),
        dose_id=settings["dose_id"],
        dose=settings["dose"],
        output=settings["output"],
        parameter=settings["parameter"],
        ts=np.linspace(0.0, settings["end"], N_TIMES),
    )


def test_matches_roadrunner(case: Case) -> None:
    m = case.model.module
    rr = roadrunner.RoadRunner(str(case.sbml))
    rr.integrator.setValue("relative_tolerance", SETTINGS.rtol)
    rr.integrator.setValue("absolute_tolerance", SETTINGS.atol)
    rr.timeCourseSelections = [f"[{o}]" if o in m.SPECIES and not m.SPECIES[o][1] else o for o in m.OUTPUTS]
    rr.setValue(case.dose_id, case.dose)
    expected = np.asarray(rr.simulate(times=case.ts.tolist()))
    ys = np.asarray(case.model.simulate(jnp.asarray(case.ts), case.segments, SETTINGS))
    assert ys.shape == expected.shape
    # a quantity without a value in the SBML file is NaN in both simulators
    undefined = np.isnan(expected)
    excess = np.abs(ys - expected) - (ATOL + RTOL * np.abs(expected))
    differs = (np.isnan(ys) != undefined) | (~undefined & ~(excess <= 0))
    failed = [m.OUTPUTS[k] for k in range(len(m.OUTPUTS)) if differs[:, k].any()]
    assert not failed
    target = m.OUTPUTS.index(case.output)
    assert expected[:, target].max() > 0, "the dose reaches the plasma"


def test_gradient_matches_finite_differences(case: Case) -> None:
    target = case.model.module.OUTPUTS.index(case.output)

    def loss(m: JaxModel) -> jax.Array:
        return jnp.mean(m.simulate(jnp.asarray(case.ts), case.segments, SETTINGS)[:, target])

    name = case.parameter
    k = case.model.module.INPUTS.index(name)
    value = float(case.model.p[k])
    assert value != 0, f"{name} is zero, the relative step needs a value"
    # a relative step: large enough that the solver tolerances do not show in
    # the difference, small enough that the truncation error is below the tolerance
    h = FD_STEP * abs(value)
    up = loss(case.model.with_inputs({name: value + h}))
    down = loss(case.model.with_inputs({name: value - h}))
    fd = float((up - down) / (2 * h))
    assert fd != 0, f"{name} has no effect on {case.output}"
    assert float(jax.grad(loss)(case.model).p[k]) == pytest.approx(fd, rel=1e-5)
