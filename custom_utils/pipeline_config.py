#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_SPLIT_STRATEGIES = {"leave_one_patient_out"}
SUPPORTED_AGGREGATIONS = {"ordered_representative_three"}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing pipeline config: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    validate_config(config)
    return config, config_path


def require(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Config missing required field: {dotted_key}")
        value = value[key]
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Config field schema_version must be 1.")

    required_fields = [
        "dataset.name",
        "dataset.metadata_csv",
        "dataset.patient_id_column",
        "dataset.slide_id_column",
        "dataset.slide_label.column",
        "dataset.slide_label.values",
        "splits.strategy",
        "splits.directory",
        "splits.minimum_test_slides",
        "training.feature_bags_dir",
        "training.run_directory",
        "evaluation.output_directory",
    ]
    for field in required_fields:
        require(config, field)

    label_values = require(config, "dataset.slide_label.values")
    if not isinstance(label_values, list) or len(label_values) < 2:
        raise ValueError("dataset.slide_label.values must contain at least two values.")
    if len({str(value) for value in label_values}) != len(label_values):
        raise ValueError("dataset.slide_label.values contains duplicate values.")

    split_strategy = require(config, "splits.strategy")
    if split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported splits.strategy {split_strategy!r}; "
            f"expected one of {sorted(SUPPORTED_SPLIT_STRATEGIES)}."
        )

    minimum_slides = int(require(config, "splits.minimum_test_slides"))
    if minimum_slides < 1:
        raise ValueError("splits.minimum_test_slides must be positive.")

    validation = config["splits"].get("validation", {"strategy": "none"})
    validation_strategy = validation.get("strategy", "none")
    if validation_strategy not in {"none", "stratified_patient_fraction"}:
        raise ValueError(
            "splits.validation.strategy must be 'none' or "
            "'stratified_patient_fraction'."
        )
    if validation_strategy == "stratified_patient_fraction":
        fraction = float(validation.get("fraction", 0))
        if not 0 < fraction < 1:
            raise ValueError(
                "splits.validation.fraction must be between zero and one."
            )
        int(validation.get("seed", 0))
        require(config, "dataset.patient_label.column")
        patient_labels = require(config, "dataset.patient_label.values")
        if not isinstance(patient_labels, list) or len(patient_labels) < 2:
            raise ValueError(
                "dataset.patient_label.values must contain at least two values "
                "for stratified validation."
            )

    slurm_jobs = int(config.get("training", {}).get("slurm_jobs", 1))
    if slurm_jobs < 1:
        raise ValueError("training.slurm_jobs must be a positive integer.")

    aggregation = config.get("evaluation", {}).get("aggregation")
    if aggregation is not None:
        aggregation_type = require(config, "evaluation.aggregation.type")
        if aggregation_type not in SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported evaluation aggregation {aggregation_type!r}; "
                f"expected one of {sorted(SUPPORTED_AGGREGATIONS)}."
            )
        thresholds = require(config, "evaluation.aggregation.thresholds")
        if not isinstance(thresholds, list) or not thresholds:
            raise ValueError("evaluation.aggregation.thresholds must be a non-empty list.")
        for item in thresholds:
            if "label" not in item:
                raise ValueError("Every aggregation threshold requires a label.")


def get_path(config: dict[str, Any], dotted_key: str) -> Path:
    return Path(str(require(config, dotted_key))).expanduser()


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_label_mapping(values: list[Any]) -> dict[str, int]:
    return {normalize_scalar(value): index for index, value in enumerate(values)}


def normalize_with_aliases(value: Any, aliases: dict[str, str] | None = None) -> str:
    normalized = normalize_scalar(value)
    if aliases is None:
        return normalized

    alias_map = {normalize_scalar(key).upper(): str(target) for key, target in aliases.items()}
    return alias_map.get(normalized.upper(), normalized)
