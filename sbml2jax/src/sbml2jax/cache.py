"""The cache of downloaded files: pinned SBML models and the SBML test suite."""

import os
from pathlib import Path

#: environment variable overriding the cache root
CACHE_ENV = "SBML2JAX_CACHE"


def cache_dir() -> Path:
    """Root of the cache, `SBML2JAX_CACHE` or `~/.cache/sbml2jax`."""
    return Path(os.environ.get(CACHE_ENV, Path.home() / ".cache" / "sbml2jax"))
