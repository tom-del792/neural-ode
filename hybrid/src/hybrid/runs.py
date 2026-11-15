"""Where generated files go: datasets, weights, logs and figures are not part of the repository.

All of them are written below one directory, `HYBRID_RUNS` or
`~/.cache/hybrid/runs`. In Colab the variable points to a folder of Google
Drive. Every run records the commit, the configuration and the seed which
produced it.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from hybrid.config import Config

#: environment variable with the directory of the generated files
RUNS_ENV = "HYBRID_RUNS"


def runs_dir() -> Path:
    """Root of the generated files, `HYBRID_RUNS` or `~/.cache/hybrid/runs`."""
    return Path(os.environ.get(RUNS_ENV, Path.home() / ".cache" / "hybrid" / "runs"))


def dataset_path(config: Config) -> Path:
    """File of the dataset of a case; a change of the data settings gives another file."""
    return runs_dir() / "datasets" / f"{config.name}-{config.data_hash()}" / "dataset.npz"


def commit() -> tuple[str, bool]:
    """The commit of the code and whether it has uncommitted changes.

    Read from the git repository of an editable installation, else from the
    installation record of pip (`pip install git+...`); `("unknown", False)`
    if neither exists.
    """
    directory = Path(__file__).parent
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(directory), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(directory), "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head, bool(status)
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        direct_url = metadata.distribution("hybrid").read_text("direct_url.json")
        return json.loads(direct_url or "{}")["vcs_info"]["commit_id"], False
    except (metadata.PackageNotFoundError, KeyError, json.JSONDecodeError):
        return "unknown", False


def new_run(config: Config) -> Path:
    """Creates the directory of a new run with `config.toml` and `run.json`.

    The directory is `<runs>/<name>/<time>-<commit>`. `run.json` holds what
    produced the run: commit, seed, dataset, versions of the packages and the
    device.
    """
    import jax

    head, dirty = commit()
    started = datetime.now().astimezone()
    directory = runs_dir() / config.name / f"{started:%Y%m%d-%H%M%S}-{head[:7]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.toml").write_text(config.text)
    record = {
        "name": config.name,
        "started": started.isoformat(timespec="seconds"),
        "commit": head,
        "uncommitted_changes": dirty,
        "seed": config.seed,
        "dataset": str(dataset_path(config).relative_to(runs_dir())),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": jax.default_backend(),
        "packages": {
            name: metadata.version(name)
            for name in ("hybrid", "sbml2jax", "jax", "equinox", "optax", "diffrax", "numpy")
        },
    }
    (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return directory
