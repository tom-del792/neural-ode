# Contributing

The repository is [matthiaskoenig/neural-ode](https://github.com/matthiaskoenig/neural-ode). Changes go through pull requests against `main`.

## Layout

The repository is a monorepo. Every top-level directory is a module: a self-contained Python package with its own `pyproject.toml`, `uv.lock`, `tests/`, `CLAUDE.md` and a CI workflow `.github/workflows/<module>.yml`, which runs with `working-directory: <module>` and only on changes under it. Run the commands of a module from its directory. New modules (hybrid models, PEtab SciML export) go into new top-level directories and depend on `sbml2jax` as a package, not by importing its files.

```text
.
├── sbml2jax/            # work package 1, see sbml2jax/CLAUDE.md
├── hybrid/              # work package 2, see hybrid/CLAUDE.md
├── docs/                # this documentation
├── zensical.toml        # configuration of the documentation
├── pyproject.toml       # the tools which build the documentation (not a package)
└── .pre-commit-config.yaml
```

## Checks of sbml2jax

```bash
cd sbml2jax
uv sync --extra dev
uv run pre-commit install

uv run pytest                                  # unit tests and the pinned models against roadrunner
tox r -e ruff                                  # lint and format check
SBML2JAX_TESTSUITE=1 tox r -e testsuite        # no regressions against testsuite/results.json
```

Code style: [ruff](https://docs.astral.sh/ruff/) with the `.ruff.toml` of the module, Google style docstrings on modules, classes and functions; they are the [API reference](api/index.md).

### Tests of sbml2jax

| file | checks |
| --- | --- |
| `tests/test_jaxmath.py` | translation of SBML formulas to `jnp` expressions: values of the operators and functions, real powers of a negative base and their finite gradients, collected dependencies, local names, errors for `delay` and `rateOf` |
| `tests/test_variables.py` | Python names of SBML ids: ids kept where possible, reserved words and keywords renamed, derived names unique |
| `tests/test_codegen.py` | assignments ordered by their dependencies; cyclic and missing definitions raise |
| `tests/test_pinned.py` | pinned models, without network (a local directory as remote, the absorption model): download into the cache, use of the cache, a wrong checksum raises and caches nothing, a damaged cache is downloaded again, a branch warns and follows the remote, invalid definitions and models files raise |
| `tests/test_models.py` | milestone of work package 1, for every model of `examples/models/models.toml` (apixaban, glimepiride, dapagliflozin; needs network on the first run, then the cache): all outputs match roadrunner (oral dose, 24 h, 97 time points), the gradient of the plasma concentration matches finite differences |
| `tests/test_jit.py` | on the absorption model, without network: `simulate` under `jax.jit` and `jax.vmap` equals the eager simulation (subjects with different clearance and dose), the gradient under `jax.jit` equals the eager gradient, the function is traced once for different values, traced output times raise a `TypeError` |
| `tests/test_testsuite_full.py` | gate, only with `SBML2JAX_TESTSUITE=1`: no regression against the committed results of the SBML test suite in `testsuite/results.json` |

The tests which need no network are `test_jaxmath.py`, `test_variables.py`, `test_codegen.py`, `test_pinned.py` and `test_jit.py`; their model is `docs/examples/absorption.xml`. `test_models.py` downloads the pinned models once into `~/.cache/sbml2jax/models/` (or `SBML2JAX_CACHE`), the test suite gate downloads the SBML test suite into the same cache. CI stores the cache, with the hash of `models.toml` in the key.

After changes of the converter, rerun the full SBML test suite and commit the regenerated `testsuite/results.json` and `docs/testsuite.md`, see [Command line](sbml2jax/cli.md#sbml2jax-testsuite). Known gaps are documented in `sbml2jax/docs/conversion-issues.md`; fixing one updates the document and the results.

## Checks of hybrid

```bash
cd hybrid
uv sync --extra dev
uv run pytest                                  # unit tests
tox r -e ruff                                  # lint and format check
```

`sbml2jax` is a path dependency (`../sbml2jax`), so the CI workflow of `hybrid` also runs on changes of `sbml2jax`.

### Tests of hybrid

| file | checks |
| --- | --- |
| `tests/test_package.py` | the package imports, double precision is enabled through `sbml2jax` |

## Documentation

The documentation is built with [zensical](https://zensical.org) from `docs/` and `zensical.toml` at the root of the repository. The API reference is rendered from the docstrings by [mkdocstrings](https://mkdocstrings.github.io/python/), which reads the sources statically. From the root of the repository:

```bash
uv sync                              # zensical and mkdocstrings
uv run zensical serve                # live preview on http://localhost:8000
uv run zensical build --strict       # the site in site/, a broken link fails
```

Some pages show files of the repository, included with `--8<--` (pymdownx.snippets), so they cannot go out of date:

| page | included file | regenerated by |
| --- | --- | --- |
| [SBML test suite](sbml2jax/testsuite.md) | `sbml2jax/docs/testsuite.md` | `sbml2jax-testsuite run` or `report` |
| [Conversion issues](sbml2jax/conversion-issues.md) | `sbml2jax/docs/conversion-issues.md` | edited by hand |
| [Home](index.md#a-first-simulation) | `sbml2jax/examples/first_simulation.py`, figure `docs/images/first-simulation.png` | `uv run python examples/first_simulation.py ../docs/images/first-simulation.png` in `sbml2jax/` |
| [Generated module](sbml2jax/generated-module.md) | `docs/examples/absorption.py` | `uv run sbml2jax ../docs/examples/absorption.xml -o ../docs/examples/absorption.py` in `sbml2jax/` |
| [Pinned models](sbml2jax/models.md) | `sbml2jax/examples/models/models.toml`, `sbml2jax/examples/apixaban_example.py` | edited by hand |

The workflow `.github/workflows/docs.yml` regenerates `docs/examples/absorption.py` and fails if it differs from the committed file, builds the site with `--strict` on every pull request, and publishes it to GitHub Pages on every push to `main`. The site is https://matthiaskoenig.github.io/neural-ode/; GitHub Pages is enabled in the settings of the repository (Pages, source "GitHub Actions").
