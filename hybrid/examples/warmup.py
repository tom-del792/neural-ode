"""The warm-up of a case: generates the panel, trains the network, evaluates the run.

    uv run python examples/warmup.py                      # the apixaban case
    uv run python examples/warmup.py configs/warmup_absorption.toml
    uv run python examples/warmup.py configs/warmup_apixaban.toml ../docs/images/warmup-fits.png

The run is written below `HYBRID_RUNS` (default `~/.cache/hybrid/runs`). With a
second argument, the figure of the fits is copied to that file; this is how
the figure of the documentation is regenerated.
"""

import json
import logging
import shutil
import sys
from pathlib import Path

from hybrid import warmup
from hybrid.config import read_config

DEFAULT = Path(__file__).parents[1] / "configs" / "warmup_apixaban.toml"


def main() -> None:
    """Runs the warm-up and prints the errors on the validation patients."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = read_config(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
    run = warmup.train(config)
    metrics = json.loads((run / "metrics.json").read_text())
    print(f"\n{config.name}: {metrics['n_train']} training and {metrics['n_valid']} validation patients")
    print(f"{'observable':<18}{'R2':>8}{'NRMSE':>8}{'prop.':>8}{'(data)':>8}")
    for name, values in metrics["observables"]["valid"].items():
        error = metrics["error_model"][name]
        print(
            f"{name:<18}{values['r2_truth']:>8.3f}{values['nrmse_truth']:>8.3f}"
            f"{error['proportional']:>8.3f}{error['proportional_data']:>8.3f}"
        )
    print(f"run: {run}")
    if len(sys.argv) > 2:
        shutil.copyfile(run / "fits.png", sys.argv[2])


if __name__ == "__main__":
    main()
