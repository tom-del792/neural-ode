"""`simulate` under `jax.jit` and `jax.vmap` equals the eager simulation.

Work package 2 simulates many subjects and doses per training step, so the
simulation is compiled once and batched over inputs and doses.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbml2jax.model import JaxModel, Segment, SolverSettings

TS = np.linspace(0.0, 24 * 60.0, 25)
SETTINGS = SolverSettings(rtol=1e-8, atol=1e-10)
RTOL, ATOL = 1e-6, 1e-12
COVARIATE = "BW"
DOSE = "PODOSE_api"


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> JaxModel:
    sbml = Path(__file__).parent.parent / "examples" / "models" / "apixaban_body_flat.xml"
    return JaxModel.from_sbml(sbml, tmp_path_factory.mktemp("apixaban") / "apixaban.py")


def simulate(model: JaxModel, weight: jax.Array, dose: jax.Array) -> jax.Array:
    """Outputs of one subject: body weight and oral dose may be traced."""
    k = model.module.INPUTS.index(COVARIATE)
    subject = JaxModel(p=model.p.at[k].set(weight), module=model.module)
    return subject.simulate(TS, [Segment(end=float(TS[-1]), changes={DOSE: dose})], SETTINGS)


def test_jit_equals_eager(model: JaxModel) -> None:
    eager = simulate(model, 75.0, 10.0)
    compiled = jax.jit(simulate)(model, 75.0, 10.0)
    np.testing.assert_allclose(compiled, eager, rtol=RTOL, atol=ATOL)


def test_vmap_equals_loop(model: JaxModel) -> None:
    weights = jnp.asarray([55.0, 75.0, 100.0])
    doses = jnp.asarray([2.5, 10.0, 25.0])
    batched = jax.jit(jax.vmap(simulate, in_axes=(None, 0, 0)))(model, weights, doses)
    assert batched.shape == (3, len(TS), len(model.module.OUTPUTS))
    for k in range(3):
        eager = simulate(model, float(weights[k]), float(doses[k]))
        np.testing.assert_allclose(batched[k], eager, rtol=RTOL, atol=ATOL)


def test_compiled_once(model: JaxModel) -> None:
    traces = 0

    def counted(weight: jax.Array, dose: jax.Array) -> jax.Array:
        nonlocal traces
        traces += 1
        return simulate(model, weight, dose)

    f = jax.jit(counted)
    f(70.0, 5.0).block_until_ready()
    f(80.0, 20.0).block_until_ready()
    assert traces == 1


def test_gradient_under_jit(model: JaxModel) -> None:
    target = model.module.OUTPUTS.index("Cve_api")

    def loss(m: JaxModel, dose: jax.Array) -> jax.Array:
        return jnp.mean(simulate(m, 75.0, dose)[:, target])

    eager = jax.grad(loss, argnums=(0, 1))(model, 10.0)
    compiled = jax.jit(jax.grad(loss, argnums=(0, 1)))(model, 10.0)
    np.testing.assert_allclose(compiled[0].p, eager[0].p, rtol=1e-5, atol=1e-14)
    np.testing.assert_allclose(compiled[1], eager[1], rtol=1e-5)
    assert float(compiled[1]) > 0, "a larger dose raises the mean concentration"


def test_traced_times_raise(model: JaxModel) -> None:
    with pytest.raises(TypeError, match="must be concrete"):
        jax.jit(lambda ts: model.simulate(ts))(jnp.asarray(TS))
