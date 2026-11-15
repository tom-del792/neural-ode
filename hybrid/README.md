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

## Warm-up: a plain network on a simulated panel

```bash
uv run hybrid train configs/warmup_apixaban.toml
```

simulates a panel of virtual patients with the apixaban model, trains a network
on it and evaluates it on patients which were held out, see the
[documentation](../docs/hybrid/warmup.md). A case is one TOML file in
`configs/`. Generated files (datasets, weights, logs, figures) are not part of
the repository: they are written below `HYBRID_RUNS` (default
`~/.cache/hybrid/runs`). `notebooks/warmup.ipynb` runs the same in Google
Colab and writes to Google Drive.

```
configs/         one TOML file per case
notebooks/       entry point for Google Colab
examples/        scripts which regenerate the figures of the documentation
src/hybrid/
  config.py      reads and checks a case
  data/          population, simulation of the panel, noise, split, files
  nn/            networks and their scaling
  training/      loss, optimization, saved weights
  evaluation/    metrics and figures
  variants/      the hybrid models (next steps)
  runs.py        where generated files go, and what produced them
  warmup.py      the warm-up: generate, train, evaluate
  cli.py         the command `hybrid`
```

## License

MIT, see [LICENSE](LICENSE).
