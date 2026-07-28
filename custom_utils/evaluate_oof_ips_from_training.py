#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from custom_utils.pipeline_config import (
    load_config,
    normalize_scalar,
    normalize_with_aliases,
)


KNOWN_SLIDE_SUFFIXES = (".tiff", ".tif", ".svs", ".ndpi", ".mrxs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool LOPO slide predictions, aggregate one prediction per patient, "
            "and compute patient-level metrics."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def normalize_slide_id(value: object) -> str:
    slide_id = normalize_scalar(value)
    for suffix in KNOWN_SLIDE_SUFFIXES:
        if slide_id.lower().endswith(suffix):
            return slide_id[: -len(suffix)]
    return slide_id


def representative_three(values: pd.Series) -> tuple[list[float], float]:
    ordered = sorted(float(value) for value in values)
    if len(ordered) < 3:
        raise ValueError("At least three observed slide grades are required.")

    selected = [ordered[0], ordered[(len(ordered) - 1) // 2], ordered[-1]]
    return selected, float(sum(selected))


def aggregate_label(total: float, thresholds: list[dict[str, object]]) -> str:
    for threshold in thresholds:
        if "max" not in threshold or total <= float(threshold["max"]):
            return str(threshold["label"])
    raise ValueError(f"No aggregation threshold accepts total {total}.")


def load_fold_results(
    pkl_path: Path,
    fold: int,
    label_values: list[object],
) -> pd.DataFrame:
    with pkl_path.open("rb") as handle:
        results = pickle.load(handle)

    rows: list[dict[str, object]] = []
    for key, item in results.items():
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported result entry in {pkl_path}: {type(item)}")

        raw_slide_id = item.get("slide_id", key)
        slide_id = str(np.asarray(raw_slide_id).reshape(-1)[0])
        probabilities = np.asarray(item["prob"], dtype=float).reshape(-1)
        true_index = int(np.asarray(item["label"]).reshape(-1)[0])

        if len(probabilities) != len(label_values):
            raise ValueError(
                f"{pkl_path}: expected {len(label_values)} probabilities, "
                f"found {len(probabilities)} for {slide_id}."
            )
        if not 0 <= true_index < len(label_values):
            raise ValueError(f"{pkl_path}: invalid label index {true_index}.")

        numeric_labels = np.asarray(label_values, dtype=float)
        predicted_index = int(probabilities.argmax())
        row: dict[str, object] = {
            "fold": fold,
            "slide_id": normalize_slide_id(slide_id),
            "true_slide_grade_from_pkl": float(numeric_labels[true_index]),
            "predicted_slide_grade": float(numeric_labels[predicted_index]),
            "expected_slide_grade": float(probabilities @ numeric_labels),
        }
        for index, probability in enumerate(probabilities):
            row[f"probability_{normalize_scalar(label_values[index])}"] = float(
                probability
            )
        rows.append(row)

    return pd.DataFrame(rows)


def find_fold_result(run_dir: Path, fold: int) -> Path:
    matches = sorted(run_dir.glob(f"results/**/split_{fold}_results.pkl"))
    if not matches:
        matches = sorted(run_dir.glob(f"**/split_{fold}_results.pkl"))
    if len(matches) != 1:
        listing = "\n".join(str(path) for path in matches) or "(none)"
        raise RuntimeError(
            f"Expected exactly one result PKL for fold {fold}; found "
            f"{len(matches)}:\n{listing}"
        )
    return matches[0]


def prepare_metadata(config: dict[str, object]) -> pd.DataFrame:
    dataset = config["dataset"]
    metadata_path = Path(dataset["metadata_csv"]).expanduser()
    patient_column = dataset["patient_id_column"]
    slide_column = dataset["slide_id_column"]
    grade_column = dataset["slide_label"]["column"]
    frame = pd.read_csv(
        metadata_path,
        dtype={patient_column: str, slide_column: str},
    )
    required = {patient_column, slide_column, grade_column}

    patient_label = dataset.get("patient_label")
    if patient_label:
        required.add(patient_label["column"])

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset CSV missing required columns: {sorted(missing)}")

    output = pd.DataFrame(
        {
            "patient_id": frame[patient_column].map(normalize_scalar),
            "slide_id": frame[slide_column].map(normalize_slide_id),
            "true_slide_grade": pd.to_numeric(frame[grade_column], errors="raise"),
        }
    )

    if patient_label:
        aliases = patient_label.get("aliases", {})
        output["true_patient_label"] = frame[patient_label["column"]].map(
            lambda value: normalize_with_aliases(value, aliases)
        )

    if output["slide_id"].duplicated().any():
        duplicates = output.loc[output["slide_id"].duplicated(), "slide_id"].tolist()
        raise ValueError(f"Duplicate metadata slide IDs: {duplicates[:20]}")
    return output


def build_patient_predictions(
    slide_predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    thresholds: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = slide_predictions.merge(
        metadata,
        on="slide_id",
        how="left",
        validate="one_to_one",
    )
    if merged["patient_id"].isna().any():
        missing = merged.loc[merged["patient_id"].isna(), "slide_id"].tolist()
        raise ValueError(f"Predicted slides missing from metadata: {missing[:20]}")

    mismatch = merged[
        ~np.isclose(
            merged["true_slide_grade_from_pkl"].astype(float),
            merged["true_slide_grade"].astype(float),
        )
    ].copy()
    if not mismatch.empty:
        raise ValueError(
            f"{len(mismatch)} slide labels in result PKLs disagree with metadata."
        )

    patient_rows: list[dict[str, object]] = []
    for patient_id, group in merged.groupby("patient_id", sort=True):
        fold_values = group["fold"].unique()
        if len(fold_values) != 1:
            raise ValueError(f"Patient {patient_id} occurs in multiple LOPO folds.")
        if len(group) < 3:
            raise ValueError(
                f"LOPO test patient {patient_id} has only {len(group)} predicted slides."
            )

        true_selected, true_sum = representative_three(group["true_slide_grade"])
        hard_selected, hard_sum = representative_three(
            group["predicted_slide_grade"]
        )
        soft_selected, soft_sum = representative_three(
            group["expected_slide_grade"]
        )

        derived_true_label = aggregate_label(true_sum, thresholds)
        if "true_patient_label" in group:
            labels = group["true_patient_label"].dropna().unique()
            if len(labels) != 1:
                raise ValueError(
                    f"Patient {patient_id} has inconsistent patient labels."
                )
            true_label = str(labels[0])
        else:
            true_label = derived_true_label

        patient_rows.append(
            {
                "patient_id": patient_id,
                "fold": int(fold_values[0]),
                "n_observed_slides": len(group),
                "true_label": true_label,
                "derived_true_label": derived_true_label,
                "true_label_matches_derived": true_label == derived_true_label,
                "hard_prediction": aggregate_label(hard_sum, thresholds),
                "soft_prediction": aggregate_label(soft_sum, thresholds),
                "true_representative_grades": "|".join(
                    f"{value:g}" for value in true_selected
                ),
                "hard_representative_grades": "|".join(
                    f"{value:g}" for value in hard_selected
                ),
                "soft_representative_grades": "|".join(
                    f"{value:.6f}" for value in soft_selected
                ),
                "true_grade_sum": true_sum,
                "hard_grade_sum": hard_sum,
                "soft_grade_sum": soft_sum,
                "slide_ids": "|".join(group["slide_id"].astype(str)),
            }
        )

    return pd.DataFrame(patient_rows), merged


def compute_metrics(
    patient_predictions: pd.DataFrame,
    prediction_column: str,
    labels: list[str],
) -> dict[str, float | int | str]:
    true_labels = patient_predictions["true_label"].astype(str)
    predicted_labels = patient_predictions[prediction_column].astype(str)
    ordinal = {label: index + 1 for index, label in enumerate(labels)}
    per_class = f1_score(
        true_labels,
        predicted_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics: dict[str, float | int | str] = {
        "prediction_type": prediction_column,
        "n_patients": len(patient_predictions),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_F1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_F1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "MAE": float(
            mean_absolute_error(
                true_labels.map(ordinal),
                predicted_labels.map(ordinal),
            )
        ),
        "QWK": float(
            cohen_kappa_score(
                true_labels.map(ordinal),
                predicted_labels.map(ordinal),
                labels=list(ordinal.values()),
                weights="quadratic",
            )
        ),
    }
    for label, score in zip(labels, per_class):
        metrics[f"{label}_F1"] = float(score)
        class_mask = true_labels == label
        metrics[f"{label}_accuracy"] = (
            float(accuracy_score(true_labels[class_mask], predicted_labels[class_mask]))
            if class_mask.any()
            else float("nan")
        )
    return metrics


def compute_slide_grade_metrics(
    slide_predictions: pd.DataFrame,
    grade_values: list[object],
) -> dict[str, float | int]:
    true_grades = slide_predictions["true_slide_grade"].astype(float)
    predicted_grades = slide_predictions["predicted_slide_grade"].astype(float)
    expected_grades = slide_predictions["expected_slide_grade"].astype(float)
    numeric_grade_values = [float(value) for value in grade_values]

    return {
        "n_slides": len(slide_predictions),
        "slide_accuracy": float(accuracy_score(true_grades, predicted_grades)),
        "slide_macro_F1": float(
            f1_score(
                true_grades,
                predicted_grades,
                labels=numeric_grade_values,
                average="macro",
                zero_division=0,
            )
        ),
        "slide_QWK": float(
            cohen_kappa_score(
                true_grades,
                predicted_grades,
                labels=numeric_grade_values,
                weights="quadratic",
            )
        ),
        "continuous_MAE": float(
            mean_absolute_error(true_grades, expected_grades)
        ),
    }


def build_metric_table(
    slide_metrics: dict[str, float | int],
    patient_metrics: pd.DataFrame,
    ips_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for _, patient_row in patient_metrics.iterrows():
        row: dict[str, float | int | str] = {
            "prediction_type": patient_row["prediction_type"],
            **slide_metrics,
            "n_patients": int(patient_row["n_patients"]),
            "patient_macro_F1": float(patient_row["macro_F1"]),
            "IPS_accuracy": float(patient_row["accuracy"]),
        }
        for label in ips_labels:
            row[f"{label}_accuracy"] = float(patient_row[f"{label}_accuracy"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    split_dir = Path(config["splits"]["directory"]).expanduser()
    run_dir = args.run_dir or Path(config["training"]["run_directory"]).expanduser()
    out_dir = args.out_dir or Path(
        config["evaluation"]["output_directory"]
    ).expanduser()

    manifest_path = split_dir / "fold_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing LOPO fold manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"test_patient": str})
    expected_folds = sorted(manifest["fold"].astype(int).tolist())

    label_values = config["dataset"]["slide_label"]["values"]
    slide_frames = [
        load_fold_results(find_fold_result(run_dir, fold), fold, label_values)
        for fold in expected_folds
    ]
    slide_predictions = pd.concat(slide_frames, ignore_index=True)
    if slide_predictions["slide_id"].duplicated().any():
        duplicates = slide_predictions.loc[
            slide_predictions["slide_id"].duplicated(keep=False), "slide_id"
        ].tolist()
        raise ValueError(f"Duplicate OOF slide predictions: {duplicates[:20]}")

    metadata = prepare_metadata(config)
    thresholds = config["evaluation"]["aggregation"]["thresholds"]
    patient_predictions, merged_slides = build_patient_predictions(
        slide_predictions,
        metadata,
        thresholds,
    )

    expected_patients = set(manifest["test_patient"].map(normalize_scalar))
    observed_patients = set(patient_predictions["patient_id"].map(normalize_scalar))
    if expected_patients != observed_patients:
        raise ValueError(
            "OOF patient coverage differs from the LOPO manifest. "
            f"Missing={sorted(expected_patients - observed_patients)}, "
            f"unexpected={sorted(observed_patients - expected_patients)}"
        )

    patient_label = config["dataset"].get("patient_label")
    labels = (
        [str(value) for value in patient_label["values"]]
        if patient_label
        else [str(item["label"]) for item in thresholds]
    )
    metric_rows = [
        compute_metrics(patient_predictions, "hard_prediction", labels),
        compute_metrics(patient_predictions, "soft_prediction", labels),
    ]
    metrics = pd.DataFrame(metric_rows)
    slide_metrics = compute_slide_grade_metrics(merged_slides, label_values)
    metric_table = build_metric_table(slide_metrics, metrics, labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    patient_predictions.to_csv(
        out_dir / "patient_oof_predictions.csv", index=False
    )
    merged_slides.to_csv(out_dir / "slide_oof_predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    metric_table.to_csv(out_dir / "metric_table.csv", index=False)

    for prediction_column in ["hard_prediction", "soft_prediction"]:
        matrix = confusion_matrix(
            patient_predictions["true_label"],
            patient_predictions[prediction_column],
            labels=labels,
        )
        pd.DataFrame(
            matrix,
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        ).to_csv(out_dir / f"{prediction_column}_confusion_matrix.csv")

    print(metric_table.to_string(index=False))
    print(f"Evaluated LOPO patients: {len(patient_predictions)}")
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
