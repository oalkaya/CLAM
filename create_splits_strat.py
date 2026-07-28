#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from custom_utils.pipeline_config import (
    build_label_mapping,
    load_config,
    normalize_scalar,
    normalize_with_aliases,
)


KNOWN_SLIDE_SUFFIXES = (".tiff", ".tif", ".svs", ".ndpi", ".mrxs", ".h5", ".pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-exclusive leave-one-patient-out splits in CLAM format."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override splits.directory from the config.",
    )
    return parser.parse_args()


def normalize_slide_id(value: object) -> str:
    slide_id = normalize_scalar(value)
    for suffix in KNOWN_SLIDE_SUFFIXES:
        if slide_id.lower().endswith(suffix):
            return slide_id[: -len(suffix)]
    return slide_id


def natural_key(value: object) -> tuple[tuple[int, object], ...]:
    text = normalize_scalar(value)
    pieces: list[tuple[int, object]] = []
    current = ""
    digit_mode: bool | None = None

    for character in text:
        is_digit = character.isdigit()
        if digit_mode is None or is_digit == digit_mode:
            current += character
        else:
            pieces.append((0, int(current)) if digit_mode else (1, current.lower()))
            current = character
        digit_mode = is_digit

    if current:
        pieces.append((0, int(current)) if digit_mode else (1, current.lower()))
    return tuple(pieces)


def pad(values: list[str], size: int) -> list[str]:
    return values + [""] * (size - len(values))


def write_split_csv(
    path: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
) -> None:
    size = max(len(train_slides), len(val_slides), len(test_slides))
    pd.DataFrame(
        {
            "train": pad(train_slides, size),
            "val": pad(val_slides, size),
            "test": pad(test_slides, size),
        }
    ).to_csv(path, index=False)


def write_bool_csv(
    path: Path,
    train_slides: list[str],
    val_slides: list[str],
    test_slides: list[str],
) -> None:
    all_slides = train_slides + val_slides + test_slides
    train_set = set(train_slides)
    val_set = set(val_slides)
    test_set = set(test_slides)
    pd.DataFrame(
        {
            "train": [slide in train_set for slide in all_slides],
            "val": [slide in val_set for slide in all_slides],
            "test": [slide in test_set for slide in all_slides],
        },
        index=all_slides,
    ).to_csv(path)


def validate_metadata(
    frame: pd.DataFrame,
    patient_column: str,
    slide_column: str,
    label_column: str,
    label_values: list[object],
) -> pd.DataFrame:
    missing = {patient_column, slide_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset CSV missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[patient_column] = frame[patient_column].map(normalize_scalar)
    frame[slide_column] = frame[slide_column].map(normalize_slide_id)

    if (frame[patient_column] == "").any() or (frame[slide_column] == "").any():
        raise ValueError("Patient and slide identifiers must be non-empty.")

    duplicates = frame[frame[slide_column].duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_ids = sorted(duplicates[slide_column].unique(), key=natural_key)
        raise ValueError(f"Duplicate slide identifiers: {duplicate_ids[:20]}")

    label_mapping = build_label_mapping(label_values)
    unknown = sorted(
        {
            normalize_scalar(value)
            for value in frame[label_column]
            if normalize_scalar(value) not in label_mapping
        }
    )
    if unknown:
        raise ValueError(
            f"Unknown values in slide label column {label_column!r}: {unknown}"
        )

    return frame


def allocate_validation_counts(
    labels: list[str],
    label_order: list[str],
    validation_size: int,
) -> dict[str, int]:
    counts = Counter(labels)
    targets = {
        label: counts[label] * validation_size / len(labels)
        for label in label_order
    }
    allocated = {
        label: int(np.floor(targets[label]))
        for label in label_order
    }
    remainder_order = sorted(
        label_order,
        key=lambda label: (
            targets[label] - allocated[label],
            label,
        ),
        reverse=True,
    )
    for label in remainder_order:
        if sum(allocated.values()) == validation_size:
            break
        allocated[label] += 1

    if (
        sum(allocated.values()) != validation_size
        or any(allocated[label] > counts[label] for label in label_order)
    ):
        raise ValueError(
            "Cannot allocate stratified validation patients: "
            f"counts={dict(counts)}, allocated={allocated}"
        )
    return allocated


def select_validation_patients(
    eligible_candidates: pd.DataFrame,
    label_order: list[str],
    fraction: float,
    seed: int,
    fold: int,
) -> set[str]:
    validation_size = int(round(len(eligible_candidates) * fraction))
    if validation_size <= 0 or validation_size >= len(eligible_candidates):
        raise ValueError(
            f"Fold {fold}: validation fraction {fraction} produces invalid "
            f"validation size {validation_size} from "
            f"{len(eligible_candidates)} candidates."
        )

    allocation = allocate_validation_counts(
        eligible_candidates["stratify_label"].tolist(),
        label_order,
        validation_size,
    )
    rng = np.random.default_rng(seed + fold)
    validation: set[str] = set()
    for label in label_order:
        patient_ids = np.asarray(
            sorted(
                eligible_candidates.loc[
                    eligible_candidates["stratify_label"] == label,
                    "patient_id",
                ].tolist(),
                key=natural_key,
            ),
            dtype=object,
        )
        rng.shuffle(patient_ids)
        validation.update(
            str(patient_id)
            for patient_id in patient_ids[: allocation[label]]
        )
    return validation


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)

    dataset = config["dataset"]
    split_config = config["splits"]
    metadata_path = Path(dataset["metadata_csv"]).expanduser()
    output_dir = args.output_dir or Path(split_config["directory"]).expanduser()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")

    patient_column = dataset["patient_id_column"]
    slide_column = dataset["slide_id_column"]
    label_column = dataset["slide_label"]["column"]
    label_values = dataset["slide_label"]["values"]
    minimum_test_slides = int(split_config["minimum_test_slides"])
    validation_config = split_config.get("validation", {"strategy": "none"})
    validation_strategy = validation_config.get("strategy", "none")

    frame = validate_metadata(
        pd.read_csv(
            metadata_path,
            dtype={patient_column: str, slide_column: str},
        ),
        patient_column,
        slide_column,
        label_column,
        label_values,
    )

    patient_label_config = dataset.get("patient_label")
    if validation_strategy == "stratified_patient_fraction" and not patient_label_config:
        raise ValueError(
            "Stratified validation requires dataset.patient_label configuration."
        )

    groups = []
    for patient_id, group in frame.groupby(patient_column, sort=False):
        row: dict[str, object] = {
            "patient_id": patient_id,
            "n_slides": len(group),
            "eligible_for_test": len(group) >= minimum_test_slides,
        }
        if patient_label_config:
            patient_label_column = patient_label_config["column"]
            if patient_label_column not in frame:
                raise ValueError(
                    f"Dataset CSV missing patient label column "
                    f"{patient_label_column!r}."
                )
            normalized_labels = {
                normalize_with_aliases(
                    value,
                    patient_label_config.get("aliases", {}),
                )
                for value in group[patient_label_column].dropna()
            }
            if len(normalized_labels) != 1:
                raise ValueError(
                    f"Patient {patient_id} has inconsistent or missing "
                    "patient-level labels."
                )
            row["stratify_label"] = normalized_labels.pop()
        groups.append(row)

    patient_table = pd.DataFrame(groups)
    eligible = patient_table[patient_table["eligible_for_test"]].copy()
    ineligible = patient_table[~patient_table["eligible_for_test"]].copy()

    if eligible.empty:
        raise ValueError(
            f"No patients have at least {minimum_test_slides} slides."
        )
    if split_config.get("ineligible_patient_policy", "train_only") != "train_only":
        raise ValueError("LOPO supports only ineligible_patient_policy='train_only'.")

    eligible_ids = sorted(eligible["patient_id"].tolist(), key=natural_key)
    validation_fraction = float(validation_config.get("fraction", 0))
    validation_seed = int(validation_config.get("seed", 0))
    validation_label_order = (
        [str(value) for value in patient_label_config["values"]]
        if patient_label_config
        else []
    )
    if validation_strategy == "stratified_patient_fraction":
        unknown_labels = sorted(
            set(eligible["stratify_label"]) - set(validation_label_order)
        )
        if unknown_labels:
            raise ValueError(
                f"Unknown eligible patient labels: {unknown_labels}"
            )

    all_patient_ids = set(patient_table["patient_id"])
    all_slide_ids = set(frame[slide_column])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    for fold, test_patient in enumerate(eligible_ids):
        eligible_candidates = eligible[
            eligible["patient_id"] != test_patient
        ].copy()
        if validation_strategy == "stratified_patient_fraction":
            val_patients = select_validation_patients(
                eligible_candidates,
                validation_label_order,
                validation_fraction,
                validation_seed,
                fold,
            )
        else:
            val_patients = set()

        test_patients = {test_patient}
        train_patients = all_patient_ids - test_patients - val_patients

        if (
            train_patients & val_patients
            or train_patients & test_patients
            or val_patients & test_patients
        ):
            raise AssertionError(f"Fold {fold}: patient leakage detected.")

        train_slides = frame.loc[
            frame[patient_column].isin(train_patients), slide_column
        ].tolist()
        val_slides = frame.loc[
            frame[patient_column].isin(val_patients), slide_column
        ].tolist()
        test_slides = frame.loc[
            frame[patient_column].isin(test_patients), slide_column
        ].tolist()

        if (
            set(train_slides) & set(val_slides)
            or set(train_slides) & set(test_slides)
            or set(val_slides) & set(test_slides)
        ):
            raise AssertionError(f"Fold {fold}: slide leakage detected.")
        if set(train_slides) | set(val_slides) | set(test_slides) != all_slide_ids:
            raise AssertionError(f"Fold {fold}: split does not cover every slide.")

        write_split_csv(
            output_dir / f"splits_{fold}.csv",
            train_slides,
            val_slides,
            test_slides,
        )
        write_bool_csv(
            output_dir / f"splits_{fold}_bool.csv",
            train_slides,
            val_slides,
            test_slides,
        )

        descriptor = {
            "fold": fold,
            "test_patient": test_patient,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "test_patients": 1,
            "train_slides": len(train_slides),
            "val_slides": len(val_slides),
            "test_slides": len(test_slides),
        }
        for label in validation_label_order:
            descriptor[f"val_{label}_patients"] = int(
                (
                    eligible_candidates[
                        eligible_candidates["patient_id"].isin(val_patients)
                    ]["stratify_label"]
                    == label
                ).sum()
            )
        pd.DataFrame([descriptor]).to_csv(
            output_dir / f"splits_{fold}_descriptor.csv", index=False
        )
        manifest_rows.append(descriptor)

        for patient_id in sorted(train_patients, key=natural_key):
            assignment_rows.append(
                {"fold": fold, "patient_id": patient_id, "role": "train"}
            )
        for patient_id in sorted(val_patients, key=natural_key):
            assignment_rows.append(
                {"fold": fold, "patient_id": patient_id, "role": "val"}
            )
        assignment_rows.append(
            {"fold": fold, "patient_id": test_patient, "role": "test"}
        )

    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "fold_manifest.csv", index=False
    )
    pd.DataFrame(assignment_rows).to_csv(
        output_dir / "patient_assignments.csv", index=False
    )
    patient_table.sort_values(
        "patient_id", key=lambda series: series.map(natural_key)
    ).to_csv(output_dir / "patient_eligibility.csv", index=False)

    snapshot = {
        "source_config": str(config_path),
        "strategy": "leave_one_patient_out",
        "metadata_csv": str(metadata_path),
        "patient_id_column": patient_column,
        "slide_id_column": slide_column,
        "slide_label_column": label_column,
        "minimum_test_slides": minimum_test_slides,
        "eligible_test_patients": len(eligible_ids),
        "train_only_patients": len(ineligible),
        "folds": len(eligible_ids),
        "validation_strategy": validation_strategy,
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
        "validation_candidate_pool": "eligible_non_test_patients",
    }
    with (output_dir / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
        handle.write("\n")

    print(f"Dataset: {dataset['name']}")
    print(f"Patients: {len(patient_table)}")
    print(f"Eligible LOPO test patients: {len(eligible_ids)}")
    print(f"Train-only patients: {len(ineligible)}")
    print(f"LOPO folds written: {len(eligible_ids)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
