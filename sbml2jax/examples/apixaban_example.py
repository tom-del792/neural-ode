"""Verification of the apixaban PBPK/PD model against roadrunner (work package 1).

The flattened whole-body model is downloaded at its pinned commit (see
`examples/models/models.toml`), converted to JAX and a 10 mg oral dose is
simulated over 24 h with roadrunner and with diffrax. Every output of the
model (species, parameters, compartments, reaction rates; 300+ quantities)
is compared at 241 time points, then the gradient of the plasma
concentration with respect to the inputs is checked against finite
differences.

    uv run python examples/apixaban_example.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import roadrunner

from sbml2jax.model import JaxModel, Segment, SolverSettings
from sbml2jax.pinned import fetch, read_pinned

MODELS = Path(__file__).parent / "models"
RESULTS = Path(__file__).parent / "results"
#: the pinned model: url, commit and checksum
PINNED = read_pinned(MODELS / "models.toml")["apixaban"]

DOSE = 10.0  # mg, oral
END = 24 * 60.0  # min
TS = np.linspace(0.0, END, 241)
#: |jax - roadrunner| <= ATOL + RTOL * |roadrunner|
RTOL, ATOL = 1e-5, 1e-9


def roadrunner_outputs(sbml: Path, outputs: tuple[str, ...], species: dict) -> np.ndarray:
    """Simulate the dose with roadrunner, selections in the quantities of the JAX outputs."""
    rr = roadrunner.RoadRunner(str(sbml))
    rr.integrator.setValue("relative_tolerance", 1e-10)
    rr.integrator.setValue("absolute_tolerance", 1e-12)
    selections = [f"[{o}]" if o in species and not species[o][1] else o for o in outputs]
    rr.timeCourseSelections = selections
    rr.setValue("PODOSE_api", DOSE)
    return np.asarray(rr.simulate(times=TS.tolist()))


def main() -> None:
    """Run the comparison and the gradient check."""
    sbml = fetch(PINNED)  # downloaded on first use, then from the cache
    print(f"{sbml.name} at commit {PINNED.ref[:7]}, sha256 {PINNED.sha256}")
    model = JaxModel.from_sbml(sbml, module_path=RESULTS / "apixaban_body_flat.py")
    m = model.module
    print(
        f"{len(m.STATES)} states, {len(m.CONSTANTS)} constants, "
        f"{len(m.INPUTS)} inputs, {len(m.OUTPUTS)} outputs"
    )
    segments = [Segment(end=END, changes={"PODOSE_api": DOSE})]
    settings = SolverSettings(rtol=1e-10, atol=1e-12)

    t0 = time.perf_counter()
    ys = np.asarray(model.simulate(jnp.asarray(TS), segments, settings))
    print(f"jax simulation (incl. compilation): {time.perf_counter() - t0:.1f} s")
    rr = roadrunner_outputs(sbml, m.OUTPUTS, m.SPECIES)

    excess = np.abs(ys - rr) - (ATOL + RTOL * np.abs(rr))
    worst = np.argsort(-excess.max(axis=0))[:5]
    failed = [m.OUTPUTS[k] for k in range(len(m.OUTPUTS)) if excess[:, k].max() > 0]
    print(f"outputs within tolerance: {len(m.OUTPUTS) - len(failed)} of {len(m.OUTPUTS)}")
    for k in worst:
        rel = np.max(np.abs(ys[:, k] - rr[:, k]) / (np.abs(rr[:, k]) + ATOL))
        print(f"  {m.OUTPUTS[k]:<30} max relative difference {rel:.2e}")

    # gradient of the mean plasma concentration w.r.t. all inputs
    target = m.OUTPUTS.index("Cve_api")

    def loss(model: JaxModel) -> jax.Array:
        return jnp.mean(model.simulate(jnp.asarray(TS), segments, settings)[:, target])

    grad = jax.grad(loss)(model).p
    name = "LI__API2M_k"
    k = m.INPUTS.index(name)
    value = float(model.p[k])
    h = 1e-3 * abs(value)
    up = loss(model.with_inputs({name: value + h}))
    down = loss(model.with_inputs({name: value - h}))
    fd = (up - down) / (2 * h)
    print(f"d mean(Cve_api) / d {name}: autodiff {float(grad[k]):.6e}, finite differences {float(fd):.6e}")


if __name__ == "__main__":
    main()
