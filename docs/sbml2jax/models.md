# Pinned models

Published models are not copied into this repository. A model is **pinned**: named by the URL of its SBML file at a fixed commit of the source repository and by the sha256 checksum of the file. [`pinned.fetch`](../api/pinned.md) downloads the file into a local cache on first use and verifies the checksum on every use, so a result always states exactly which model produced it.

```python
from sbml2jax.model import JaxModel
from sbml2jax.pinned import PinnedModel, fetch

apixaban = PinnedModel(
    name="apixaban",
    url="https://raw.githubusercontent.com/matthiaskoenig/apixaban-model/{ref}/models/apixaban_body_flat.xml",
    ref="d0e3277f3ebf389fe8ed8c9bc63be567ba929d3d",
    sha256="eeba5a98213ee4b32c2c2320b61c8409a5f427d59761c406738c6e21e741c6bb",
)
model = JaxModel.from_sbml(fetch(apixaban))
```

- **`url`** contains the placeholder `{ref}`, **`ref`** is the full commit hash (40 digits). A commit never changes, so the file is always the same.
- **`sha256`** is required for a commit. A file with another checksum is not used: `fetch` raises a `ChecksumError` with the URL, the expected and the actual checksum, and nothing is written to the cache. A cached file which no longer has the checksum is downloaded again.
- **The cache** is `~/.cache/sbml2jax/models/<name>/<ref>/`, or below the directory in the environment variable `SBML2JAX_CACHE`. It is the cache of the [SBML test suite](testsuite.md) as well. After the first download no network is needed.
- **A branch or tag** such as `ref="main"` is allowed to try out the current state of a model. It has no checksum, is downloaded on every call and logs a warning. Results of the thesis always use a commit.

## The models file

Several models are kept in a TOML file and read with `read_pinned`, one table per model:

```toml
--8<-- "sbml2jax/examples/models/models.toml"
```

```python
from pathlib import Path

from sbml2jax.pinned import fetch, read_pinned

models = read_pinned(Path("examples/models/models.toml"))
sbml_path = fetch(models["glimepiride"])
```

`read_pinned` reads `url`, `ref` and `sha256`. Further keys and sub-tables belong to the user of the file; here the table `verification` holds the settings of the verification below. Nothing about a model is in the code: a new model is a new table.

To update a model to a newer commit, change `ref` and `sha256` together. The checksum of a file is `sbml2jax.pinned.sha256sum(path)`; a wrong one is also reported with the actual value by the `ChecksumError`.

## Verified models

`tests/test_models.py` runs every model of `sbml2jax/examples/models/models.toml` through the same two checks, in CI:

1. **All outputs match roadrunner.** The dose of the `verification` table is simulated up to `end` with roadrunner and with diffrax (both with relative tolerance 1e-10 and absolute tolerance 1e-12). Every output (species, parameters, compartments, reaction rates) is compared at 97 time points with $|y_\text{jax} - y_\text{rr}| \le 10^{-9} + 10^{-5} \, |y_\text{rr}|$. A quantity without a value in the SBML file must be NaN in both.
2. **The gradient matches finite differences.** The gradient of the mean of `output` with respect to `parameter` by automatic differentiation is compared with a central finite difference (step 1e-3 of the value, relative tolerance 1e-5).

| model | states | inputs | outputs | dose | reference |
| --- | --- | --- | --- | --- | --- |
| apixaban | 51 | 123 | 257 | 10 mg oral | [Myshkina *et al.* 2026](../references.md#models) |
| glimepiride | 51 | 109 | 220 | 4 mg oral | [Elias & König 2026](../references.md#models) |
| dapagliflozin | 38 | 95 | 200 | 10 mg oral | [Nemitz *et al.* 2026](../references.md#models) |

All three models pass both checks. The dapagliflozin model first failed the gradient check, which led to the handling of real powers described in [Conversion issues](conversion-issues.md). [Apixaban in detail](#apixaban-in-detail) shows the comparison for one model.

## Apixaban in detail

The milestone of work package 1: the apixaban PBPK/PD model ([Myshkina *et al.* 2026](../references.md#models)) simulated with `sbml2jax` and diffrax matches libRoadRunner, and its gradients are correct.

`apixaban_body_flat.xml` is the flattened whole-body model: arterial and venous blood, lung, liver, kidney, intestine and the rest of the body, with the absorption of the tablet, the hepatic metabolism to the metabolites M1, M2 and M7, renal and fecal excretion, and the effect of the plasma concentration on the coagulation markers INR, PT, aPTT, mPT and the anti-Xa activity. It has 51 states, 72 constants, 123 inputs and 257 outputs. The comp model and its submodels are in the same directory of the [source repository](https://github.com/matthiaskoenig/apixaban-model).

`examples/apixaban_example.py` runs the two checks above for this model alone and prints the result: a 10 mg oral dose over 24 hours, every output compared at 241 time points, and the gradient of the mean plasma concentration with respect to the rate constant of the hepatic metabolism `LI__API2M_k`.

```bash
cd sbml2jax
uv run python examples/apixaban_example.py
```

```text
apixaban_body_flat.xml at commit d0e3277, sha256 eeba5a98213ee4b32c2c2320b61c8409a5f427d59761c406738c6e21e741c6bb
51 states, 72 constants, 123 inputs, 257 outputs
jax simulation (incl. compilation): 9.2 s
outputs within tolerance: 257 of 257
  PODOSE_api                     max relative difference 5.26e-03
  GU__APITAB                     max relative difference 4.89e-06
  LI__M7EX                       max relative difference 8.03e-06
  GU__m2_colon                   max relative difference 5.68e-06
  GU__APIUNDIS                   max relative difference 1.92e-06
d mean(Cve_api) / d LI__API2M_k: autodiff -5.595828e-04, finite differences -5.595828e-04
```

All 257 outputs are within the tolerance. The largest relative differences are in quantities which decay towards zero, such as the remaining dose in the tablet `PODOSE_api`, where the absolute difference is below the absolute tolerance. The derivative by automatic differentiation agrees with finite differences to all printed digits.

??? example "The script `examples/apixaban_example.py`"

    ```python
    --8<-- "sbml2jax/examples/apixaban_example.py"
    ```
