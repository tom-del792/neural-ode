"""Command line: `hybrid generate`, `hybrid train` and `hybrid evaluate`."""

import argparse
import json
import logging
from pathlib import Path

from hybrid import warmup
from hybrid.config import read_config
from hybrid.runs import RUNS_ENV, runs_dir


def main(argv: list[str] | None = None) -> None:
    """Entry point of the command `hybrid`."""
    parser = argparse.ArgumentParser(
        prog="hybrid",
        description=f"Neural networks on PK/PD data. Generated files go to ${RUNS_ENV} (now: {runs_dir()}).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="simulate the panel of a case and save it")
    generate.add_argument("config", type=Path, help="TOML file of the case")
    train = commands.add_parser("train", help="train the network of a case and evaluate it")
    train.add_argument("config", type=Path, help="TOML file of the case")
    train.add_argument(
        "--regenerate", action="store_true", help="simulate the panel again even if its file exists"
    )
    evaluate = commands.add_parser("evaluate", help="write the metrics and figures of a run again")
    evaluate.add_argument("run", type=Path, help="directory of the run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "generate":
        warmup.dataset(read_config(args.config), regenerate=True)
    elif args.command == "train":
        directory = warmup.train(read_config(args.config), regenerate=args.regenerate)
        print((directory / "metrics.json").read_text())
    else:
        print(json.dumps(warmup.evaluate(args.run), indent=2))


if __name__ == "__main__":
    main()
