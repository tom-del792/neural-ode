# hybrid

Hybrid neural-network ODE models for PK/PD in JAX, work package 2 of the
M.Sc. thesis "Hybrid neural-network ODE models for PK/PD". Neural networks
replace poorly known terms of a mechanistic ODE model which
[`sbml2jax`](../sbml2jax) generates from SBML. The plan is described in the
[documentation](../docs/hybrid/index.md).

Everything specific to a substance (SBML model, inputs, outputs, dosing,
covariates) is configuration, so the same code applies to another model
without changes.

## Installation

```bash
cd hybrid
uv sync --extra dev
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE).
