# sbml2jax

`sbml2jax` converts the ODE system of an [SBML](https://sbml.org) model into a
readable, differentiable [JAX](https://github.com/jax-ml/jax) module and
simulates it with [diffrax](https://github.com/patrick-kidger/diffrax). The
model is an [equinox](https://github.com/patrick-kidger/equinox) module, so
`jax.grad` gives gradients with respect to every number of the SBML model, and
neural networks can be added to its right-hand side (hybrid models,
universal differential equations).

It is the first module of [neural-ode](../README.md) and work package 1 of the M.Sc. thesis *Hybrid neural-network ODE models for
PK/PD* (supervisor Matthias König), with the structure of
[sbml2cellml](https://github.com/matthiaskoenig/sbml2cellml).
[libRoadRunner](https://github.com/sys-bio/roadrunner) is the reference
simulator.

## Status

- **Apixaban PBPK/PD model** (Myshkina et al. 2026, flattened whole-body
  model, 51 states): all 257 outputs match roadrunner over a 24 h oral dose
  (relative 1e-5, absolute 1e-9), and the gradient of the plasma
  concentration agrees with finite differences
  (`tests/test_models.py`, `examples/apixaban_example.py`). The glimepiride and
  dapagliflozin models pass the same checks. The models are downloaded at a
  pinned commit, see `examples/models/README.md`.
- **SBML test suite** (3.5.0, all 1823 cases, `docs/testsuite.md`): 1001 pass,
  2 fail, 655 unsupported, 165 skip. Every model the converter accepts is
  simulated correctly except two cases with a parameter named `time`, which
  roadrunner fails as well. The unsupported cases use events, algebraic rules,
  delays, rateOf or rate rules of stoichiometries, see
  [conversion issues](docs/conversion-issues.md).

## Installation

```bash
uv sync --extra dev           # development, from uv.lock
pip install "sbml2jax[testsuite] @ git+https://github.com/matthiaskoenig/neural-ode.git#subdirectory=sbml2jax"   # e.g. in Colab
```

## Usage

```python
from pathlib import Path

import jax
import jax.numpy as jnp

from sbml2jax.model import JaxModel, Segment
from sbml2jax.pinned import fetch, read_pinned

# the apixaban model at a pinned commit, downloaded into the cache on first use
sbml_path = fetch(read_pinned(Path("examples/models/models.toml"))["apixaban"])
model = JaxModel.from_sbml(sbml_path)
ts = jnp.linspace(0.0, 24 * 60.0, 241)  # min
dose = [Segment(end=24 * 60.0, changes={"PODOSE_api": 10.0})]  # 10 mg oral
ys = model.simulate(ts, dose)  # (time, OUTPUTS)
cve = ys[:, model.module.OUTPUTS.index("Cve_api")]

# gradient of the mean plasma concentration w.r.t. all inputs of the model
grad = jax.grad(lambda m: m.simulate(ts, dose)[:, model.module.OUTPUTS.index("Cve_api")].mean())(model)
```

The generated module can also be written and read as a file:

```bash
sbml2jax model.xml -o model.py
```

It contains `INPUTS`, `STATES`, `CONSTANTS`, `OUTPUTS` and the functions
`initial(p, t)`, `rhs(t, y, c)` and `outputs(t, y, c)` as plain `jax.numpy`
code.

## Verification

```bash
uv run pytest                                # unit tests and the pinned models against roadrunner
uv run sbml2jax-testsuite run --first 10     # SBML test suite cases (all: ~40 min on 2 CPUs)
uv run python examples/apixaban_example.py
```

Every case of the [SBML test suite](https://github.com/sbmlteam/sbml-test-suite)
is simulated with roadrunner, converted, simulated with diffrax, and both
simulations are compared with the expected results. The results of the full
run are committed (`testsuite/results.json`); `tests/test_testsuite_full.py`
reruns an evenly spread selection of 60 passing cases and fails on regressions
(`SBML2JAX_TESTSUITE=1`, in CI). After converter changes, rerun the full suite
and commit the regenerated results.

## License

MIT. `src/sbml2jax/testsuite/cases.py` and `compare.py` are adapted from
sbml2cellml (MIT, Matthias König). The models of the examples are not part of
the package: they are downloaded from their source repositories and keep their
license (CC BY-SA 4.0), see `examples/models/README.md`.
