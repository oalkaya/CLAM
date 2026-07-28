#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from custom_utils.pipeline_config import load_config


REPO_DIR = Path(__file__).resolve().parent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Configuration-driven CLAM patching, feature, and LOPO workflow."
    )
    parser.add_argument(
        "operation",
        choices=["validate", "splits", "patch", "features", "train", "evaluate"],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["single", "bulk"], default="bulk")
    parser.add_argument("--target", default="all")
    parser.add_argument("--patching-run-id", default=None)
    return parser.parse_known_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_DIR, check=True)


def validate_precomputed_features(config: dict[str, object]) -> None:
    dataset = config["dataset"]
    metadata_path = Path(dataset["metadata_csv"]).expanduser()
    bags_dir = Path(config["training"]["feature_bags_dir"]).expanduser()
    slide_column = dataset["slide_id_column"]

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")
    if not bags_dir.is_dir():
        raise FileNotFoundError(f"Missing feature bag directory: {bags_dir}")

    metadata = pd.read_csv(metadata_path, dtype={slide_column: str})
    if slide_column not in metadata:
        raise ValueError(f"Metadata is missing slide column {slide_column!r}.")

    missing = [
        str(bags_dir / f"{slide_id}.pt")
        for slide_id in metadata[slide_column].astype(str).str.strip()
        if not (bags_dir / f"{slide_id}.pt").is_file()
    ]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} feature bags. First missing files:\n{preview}"
        )
    print(f"Validated {len(metadata)} feature bags under {bags_dir}.")


def main() -> None:
    args, extra = parse_args()
    config, config_path = load_config(args.config)

    if args.operation == "validate":
        if extra:
            raise ValueError(f"Unexpected arguments: {extra}")
        print(f"Valid pipeline config: {config_path}")
        return

    if args.operation == "splits":
        run(
            [
                sys.executable,
                str(REPO_DIR / "create_splits_strat.py"),
                "--config",
                str(config_path),
                *extra,
            ]
        )
        return

    if args.operation == "patch":
        run(
            [
                "bash",
                str(REPO_DIR / "job_scripts" / "run_create_patches_fp.sh"),
                args.mode,
                args.target,
                "--config",
                str(config_path),
                *extra,
            ]
        )
        return

    if args.operation == "features":
        if args.patching_run_id is None:
            if extra:
                raise ValueError(f"Unexpected arguments: {extra}")
            validate_precomputed_features(config)
        else:
            run(
                [
                    "bash",
                    str(REPO_DIR / "job_scripts" / "run_extract_features_fp.sh"),
                    args.patching_run_id,
                    "--config",
                    str(config_path),
                    *extra,
                ]
            )
        return

    if args.operation == "train":
        run(
            [
                "bash",
                str(REPO_DIR / "job_scripts" / "run_train_grade.sh"),
                "--config",
                str(config_path),
                *extra,
            ]
        )
        return

    if args.operation == "evaluate":
        run(
            [
                sys.executable,
                str(REPO_DIR / "custom_utils" / "evaluate_oof_ips_from_training.py"),
                "--config",
                str(config_path),
                *extra,
            ]
        )


if __name__ == "__main__":
    main()
