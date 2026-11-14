# Models

The models of the examples and of the verification are not copied into this
repository. `models.toml` pins each one to a commit of its source repository
and to the sha256 checksum of the SBML file; `sbml2jax.pinned.fetch` downloads
the file into the cache (`~/.cache/sbml2jax/models/`, or `SBML2JAX_CACHE`) on
first use and verifies the checksum on every use.

| model | source | reference | license |
| --- | --- | --- | --- |
| apixaban | https://github.com/matthiaskoenig/apixaban-model | Myshkina M. et al. (2026), doi:10.5281/zenodo.18877904 | CC BY-SA 4.0 |
| glimepiride | https://github.com/matthiaskoenig/glimepiride-model | Elias M. & König M. (2026), doi:10.5281/zenodo.14628700 | CC BY-SA 4.0 |
| dapagliflozin | https://github.com/matthiaskoenig/dapagliflozin-model | Nemitz N., Elias M. & König M. (2026), doi:10.5281/zenodo.13987865 | CC BY-SA 4.0 |

The pinned file of each model is the flattened whole-body model
`models/<name>_body_flat.xml`; the comp model and its submodels (liver,
kidney, intestine) are in the same directory of the source repository.

## Adding or updating a model

Add a table to `models.toml`, or change `ref` and `sha256` of an existing one.
The checksum of a file is printed by

```bash
uv run python -c "from pathlib import Path; from sbml2jax.pinned import sha256sum; print(sha256sum(Path('model.xml')))"
```

`tests/test_models.py` then verifies the model against roadrunner; no code
changes are needed.
