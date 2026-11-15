"""Builds `walkthrough.ipynb`: the code of the package in the cells of one notebook.

    python notebooks/build_walkthrough.py          # writes the notebook
    python notebooks/build_walkthrough.py --check  # fails if the notebook is not current

Every source file of `hybrid` is a cell which starts with `%%source <file>`.
Running the cell writes the file into the clone of the repository, so the
code in the notebook is the code which runs, and it is edited in the cell.
Cells between the files run the step and show its result.

The notebook is generated from the files and never edited by hand: after a
change of a source file, run this script and commit the notebook. A test
checks that it is current.
"""

import hashlib
import json
import sys
from pathlib import Path

HYBRID = Path(__file__).parents[1]
NOTEBOOK = HYBRID / "notebooks" / "walkthrough.ipynb"

SETTINGS = """
REPO = "https://github.com/matthiaskoenig/neural-ode.git"  # the repository, or your fork to push changes
REF = "main"  # branch, tag or commit
RUNS = "/content/drive/MyDrive/hybrid_runs"  # generated files; "MyDrive" is "Meine Ablage"
"""

COLAB = """
import os

from google.colab import drive

drive.mount("/content/drive")

REPO_DIR = "/content/neural-ode"
if not os.path.isdir(REPO_DIR):  # a second run of this cell keeps your changes of the files
    !git clone -q {REPO} {REPO_DIR} && git -C {REPO_DIR} checkout -q {REF}
!pip install -q -e {REPO_DIR}/sbml2jax -e {REPO_DIR}/hybrid
"""

SESSION = '''
import hashlib
import logging
import os
import sys
import warnings
from pathlib import Path

from IPython.core.magic import register_cell_magic

os.makedirs(RUNS, exist_ok=True)
os.environ["HYBRID_RUNS"] = RUNS
sys.path[:0] = [f"{REPO_DIR}/hybrid/src", f"{REPO_DIR}/sbml2jax/src"]
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
# shell commands (lines with !) after JAX was started
warnings.filterwarnings("ignore", message=r"os\\.fork\\(\\) was called")

# a changed file is imported again by the next cell which uses it
%load_ext autoreload
%autoreload 2

# sha256 of every file when this notebook was built
ORIGINAL = __ORIGINAL__


@register_cell_magic
def source(line, cell):
    """Writes the cell to a file of hybrid/ in the clone, if the cell differs from the file."""
    name = line.strip()
    path = Path(REPO_DIR) / "hybrid" / name
    text = cell.rstrip("\\n") + "\\n"
    on_disk = path.read_text() if path.exists() else None
    cell_is_original = hashlib.sha256(text.encode()).hexdigest() == ORIGINAL.get(name)
    if text == on_disk:
        print(f"{name}: unchanged")
    elif cell_is_original and on_disk is not None:
        # the cell was not edited, so the file is from another commit than this notebook
        print(f"{name}: NOT written. The file of the repository differs from this notebook;")
        print("open the notebook of the same branch or commit as REF.")
    else:
        path.write_text(text)
        print(f"{name}: written")


!git -C {REPO_DIR} log -1 --format="commit %h: %s"
'''

CHECKS = """
!pip install -q ruff pytest pytest-xdist
!cd {REPO_DIR}/hybrid && ruff check && ruff format --check --diff && python -m pytest -q -x -n 2
"""

PUSH = """
BRANCH = "my-change"  # a new branch of your fork, not main
MESSAGE = "hybrid: describe the change"
GIT_NAME = ""  # author of the commit
GIT_EMAIL = ""

import subprocess

from google.colab import userdata

assert BRANCH not in ("main", "develop"), "push to a branch of its own and open a pull request"
assert GIT_NAME and GIT_EMAIL, "set the author of the commit"
token = userdata.get("GITHUB_TOKEN_CLASSIC")  # Colab secret with a token which may push to REPO


def git(*args):
    result = subprocess.run(["git", "-C", REPO_DIR, *args], capture_output=True, text=True)
    print((result.stdout + result.stderr).replace(token, "***").strip())
    result.check_returncode()


# the notebook is generated from the files, so it is built again from your changes
!cd {REPO_DIR}/hybrid && python notebooks/build_walkthrough.py
git("checkout", "-B", BRANCH)
git("add", "-A")
git("status", "--short")
git("-c", f"user.name={GIT_NAME}", "-c", f"user.email={GIT_EMAIL}", "commit", "-m", MESSAGE)
git("push", REPO.replace("https://", f"https://{token}@"), f"HEAD:refs/heads/{BRANCH}")
print(f"open the pull request: {REPO.removesuffix('.git')}/compare/{BRANCH}?expand=1")
"""

#: the notebook in order: ("md", text), ("file", path below hybrid/) or ("code", text)
CELLS: list[tuple[str, str]] = [
    (
        "md",
        """
# Walk-through of `hybrid`: the warm-up, with all of its code

Work package 2, first step: a neural network (no hybrid model yet) learns the pharmacokinetics and pharmacodynamics of a panel of virtual patients, which the mechanistic model simulates.

This notebook contains the code of the package `hybrid`, file by file, in the order in which a run uses it:

- A cell which starts with `%%source <file>` **is** that file of the repository. Running it writes the file into the clone of this session, if the cell differs from the file.
- The cells between the files run the step and show its result.
- To change the code, edit the cell and run it; the cells after it use the new code.
- The last section pushes your changes to a branch of your fork, for a pull request.

Run the cells from top to bottom. Generated files (dataset, weights, logs, figures) are written to Google Drive, not to the repository. The tests are not in this notebook; they are in `hybrid/tests/` of the clone (file browser on the left).
""",
    ),
    (
        "md",
        "## Setup\n\nThe repository, the branch or commit, and the folder of Google Drive for the generated files.",
    ),
    ("code", SETTINGS),
    ("code", COLAB),
    ("code", SESSION),
    (
        "md",
        """
## 1. A case is one file

Everything specific to a case is in a TOML file: the SBML model, how the patients differ (dose and covariates), what is measured (observables with their noise), the network and the training. The code below contains no model ids, doses or covariates; another substance is another file.
""",
    ),
    ("file", "configs/warmup_apixaban.toml"),
    ("md", "`config.py` reads and checks the file."),
    ("file", "src/hybrid/config.py"),
    (
        "code",
        """
from hybrid.config import read_config

CASE = f"{REPO_DIR}/hybrid/configs/warmup_apixaban.toml"
config = read_config(CASE)
print("patients differ in:", config.subject_inputs)
print("observables:       ", tuple(config.observables))
print("sampling times:    ", config.times)
print("dataset id:        ", config.data_hash())
""",
    ),
    (
        "md",
        """
## 2. Where generated files go

Datasets, weights, logs and figures are not part of the repository. `runs.py` puts them below one directory, here the folder of Google Drive, and records for every run which commit, configuration and seed produced it.
""",
    ),
    ("file", "src/hybrid/runs.py"),
    (
        "code",
        """
from hybrid.runs import commit, dataset_path, runs_dir

print("generated files:", runs_dir())
print("dataset of the case:", dataset_path(config).relative_to(runs_dir()))
print("commit of the code, uncommitted changes:", commit())
""",
    ),
    (
        "md",
        """
## 3. Data generation

### Virtual patients

The dose and the covariates of every patient are sampled from the distributions of the case.
""",
    ),
    ("file", "src/hybrid/data/population.py"),
    (
        "code",
        """
import numpy as np
import pandas as pd

from hybrid.data.population import sample_population

x = sample_population(config, np.random.default_rng(0))
patients = pd.DataFrame(x, columns=config.subject_inputs)
display(patients.head())
patients.describe().loc[["min", "mean", "max"]]
""",
    ),
    (
        "md",
        """
### Simulation and noise

All patients are simulated with the mechanistic model: one function for one patient, batched over the patients with `jax.vmap` and compiled with `jax.jit`. Then normal noise with `sigma = additive + proportional * value` is added per observable.
""",
    ),
    ("file", "src/hybrid/data/simulate.py"),
    (
        "code",
        """
import matplotlib.pyplot as plt

from hybrid.data.simulate import add_noise, panel_model, simulate_panel

model = panel_model(config)  # downloads the pinned model once, checks the ids of the case
y_clean = simulate_panel(model, config, x[:8])  # the first 8 patients; compiles first
y, mask = add_noise(y_clean, config, np.random.default_rng(1))

figure, axes = plt.subplots(1, len(config.observables), figsize=(4 * len(config.observables), 3))
for k, (axis, name) in enumerate(zip(np.atleast_1d(axes), config.observables)):
    axis.plot(config.times, y_clean[:, :, k].T, linewidth=1)
    axis.scatter(config.times, y[0, :, k], s=12, color="black", zorder=3)
    axis.set_title(name)
    axis.set_xlabel("time")
figure.suptitle("8 patients without noise (lines), measurements of the first patient (dots)")
figure.tight_layout()
""",
    ),
    (
        "md",
        """
### The panel

`Panel` holds the measurements of all patients and the split by patient: a patient is either used for training or for validation, never both. `generate_panel` runs the three steps above for all patients; `io.py` saves the panel as one file.
""",
    ),
    ("file", "src/hybrid/data/dataset.py"),
    ("file", "src/hybrid/data/io.py"),
    (
        "code",
        """
from hybrid.data.dataset import generate_panel
from hybrid.data.io import load_panel, save_panel

path = dataset_path(config)
if path.exists():
    panel = load_panel(path)
else:
    panel = generate_panel(config)  # all patients, about half a minute
    save_panel(path, panel)
print(path)
print("measurements (patients, times, observables):", panel.y.shape)
print("training patients:", panel.train.sum(), " validation patients:", (~panel.train).sum())
""",
    ),
    (
        "md",
        """
## 4. The network

`mlp.py` builds a plain multilayer perceptron. `zero_last` is not used in the warm-up; the hybrid models need it to start as the mechanistic model.
""",
    ),
    ("file", "src/hybrid/nn/mlp.py"),
    (
        "md",
        "`scaling.py` standardizes the inputs and outputs. The constants are fitted to the training patients, saved with the weights and not trained.",
    ),
    ("file", "src/hybrid/nn/scaling.py"),
    (
        "md",
        "`panel.py` is the network of the warm-up: it maps a time and the inputs of a patient (dose and covariates) to all observables.",
    ),
    ("file", "src/hybrid/nn/panel.py"),
    (
        "code",
        """
import equinox as eqx
import jax
import jax.numpy as jnp

from hybrid.nn.panel import PanelNetwork
from hybrid.training.fit import is_trainable

network = PanelNetwork.create(panel, config.network, jax.random.PRNGKey(config.seed))
weights = jax.tree_util.tree_leaves(eqx.filter(network, is_trainable))
print("trained numbers:", sum(leaf.size for leaf in weights))

times = jnp.asarray(panel.times)
prediction = network.predict(times, jnp.asarray(panel.x))
print("prediction (patients, times, observables):", prediction.shape)
print("one patient at one time, untrained:", dict(zip(panel.y_names, np.asarray(prediction[0, 5]).round(4).tolist())))
""",
    ),
    (
        "md",
        """
## 5. Training

### The loss

The negative log-likelihood of the measurements under a normal error model. The size of the noise is not given to the training: the additive and the proportional part of every observable are trained with the network.
""",
    ),
    ("file", "src/hybrid/training/loss.py"),
    (
        "code",
        """
from hybrid.training.loss import ErrorModel, gaussian_nll

error = ErrorModel.create(network.scale_out.scale)
print("initial sigma of every observable at its mean:", np.asarray(error.sigma(jnp.asarray(network.scale_out.shift))))
initial = gaussian_nll(prediction, jnp.asarray(panel.y), error.sigma(prediction), jnp.asarray(panel.mask))
print("loss of the untrained network:", float(initial))
""",
    ),
    (
        "md",
        """
### The optimization

Adam on batches of patients, then L-BFGS on all training patients. The model with the lowest loss on the validation patients is returned. `fit` knows nothing about the warm-up; the hybrid models train with the same function.
""",
    ),
    ("file", "src/hybrid/training/fit.py"),
    ("file", "src/hybrid/training/checkpoint.py"),
    ("md", "## 6. Evaluation\n\nErrors per observable, and the figures of a run."),
    ("file", "src/hybrid/evaluation/metrics.py"),
    ("file", "src/hybrid/evaluation/plots.py"),
    (
        "md",
        """
## 7. The warm-up: generate, train, evaluate

`warmup.py` puts the steps together, and `cli.py` is the command `hybrid` which calls it (`hybrid train <case>` in a terminal).
""",
    ),
    ("file", "src/hybrid/warmup.py"),
    ("file", "src/hybrid/cli.py"),
    (
        "code",
        """
from hybrid import warmup

config = read_config(CASE)  # read again, in case the case was changed above
run = warmup.train(config)
""",
    ),
    (
        "md",
        """
### Results

The run on Google Drive: `config.toml`, `run.json` (commit, seed, versions), `weights.eqx`, `log.csv`, `metrics.json` and the figures. The errors are against the mechanistic model without noise, on the validation patients, which the training has not seen.
""",
    ),
    (
        "code",
        """
import json

from IPython.display import Image

metrics = json.loads((run / "metrics.json").read_text())
print(run)
display(pd.DataFrame(metrics["observables"]["valid"]).T)
display(pd.DataFrame(metrics["error_model"]).T)
for name in ("fits.png", "parity.png", "learning.png"):
    display(Image(filename=str(run / name), width=900))
""",
    ),
    (
        "md",
        """
## 8. Changing the code

Edit a cell above and run it, then run the cells which use it. Examples:

- another network: `width`, `depth` or `activation` in the cell of `configs/warmup_apixaban.toml`, then the training cell of section 7
- more patients or other covariates: the same cell; a new dataset is generated because the data settings changed
- another loss or optimizer: the cells of `training/loss.py` or `training/fit.py`

`git diff` shows what differs from the commit of this session.
""",
    ),
    ("code", "!git -C {REPO_DIR} status --short && git -C {REPO_DIR} diff --stat"),
    (
        "md",
        "### Checks\n\nThe checks of the repository: lint, format and the tests (a few minutes). They must pass before a pull request is merged.",
    ),
    ("code", CHECKS),
    (
        "md",
        """
### Push the changes

Commits the changed files to a new branch and pushes it. This needs `REPO` to be a repository you may push to (your fork) and a Colab secret `GITHUB_TOKEN_CLASSIC` with a token for it. Nothing is pushed to `main`; open a pull request from the branch.
""",
    ),
    ("code", PUSH),
]


def files() -> list[str]:
    """The files of the package which the notebook must contain: all code and the apixaban case."""
    sources = sorted(
        str(path.relative_to(HYBRID)) for path in (HYBRID / "src").rglob("*.py") if path.name != "__init__.py"
    )
    return ["configs/warmup_apixaban.toml", *sources]


def build() -> str:
    """The text of the notebook, from the files as they are now."""
    shown = [content for kind, content in CELLS if kind == "file"]
    missing = sorted(set(files()) - set(shown))
    if missing or len(shown) != len(set(shown)):
        raise SystemExit(f"files which are not exactly once in the notebook: {missing or 'duplicates'}")
    original = {name: hashlib.sha256((HYBRID / name).read_bytes()).hexdigest() for name in shown}
    cells = []
    for number, (kind, content) in enumerate(CELLS):
        if kind == "file":
            text = f"%%source {content}\n{(HYBRID / content).read_text()}"
        else:
            text = content.replace("__ORIGINAL__", json.dumps(original, indent=4))
        cell = {
            "cell_type": "markdown" if kind == "md" else "code",
            "id": f"cell-{number:02d}",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
        if kind != "md":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    """Writes the notebook, or with `--check` fails if it is not current."""
    text = build()
    if "--check" in sys.argv:
        if not NOTEBOOK.is_file() or NOTEBOOK.read_text() != text:
            raise SystemExit(f"{NOTEBOOK.name} is not current: run python notebooks/build_walkthrough.py")
        return
    NOTEBOOK.write_text(text)


if __name__ == "__main__":
    main()
