from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import prepare
from .release_benchmark import benchmark_release
from .training import train_release

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training.toml"


def _release_from_config(config: Path) -> Path:
    import tomllib

    with config.open("rb") as file:
        values = tomllib.load(file)
    return config.resolve().parent / str(values.get("output_dir") or "data/release")


def cmd_prepare(args: argparse.Namespace) -> None:
    output = prepare(args.config)
    print(f"Prepared datasets in {output}")


def cmd_train(args: argparse.Namespace) -> None:
    print(json.dumps(train_release(args.release), indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    print(json.dumps(benchmark_release(args.release), indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    release = prepare(args.config)
    train_release(release)
    report = benchmark_release(release)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-models",
        description="Prepare leakage-safe datasets, train both verdict models, and benchmark once.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.set_defaults(func=cmd_prepare)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--release", type=Path, default=_release_from_config(DEFAULT_CONFIG))
    train_parser.set_defaults(func=cmd_train)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument(
        "--release", type=Path, default=_release_from_config(DEFAULT_CONFIG)
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
