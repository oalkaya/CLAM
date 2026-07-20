#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import re
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


IPS_LABELS = ["IPS1", "IPS2", "IPS3"]
IPS_TO_NUM = {"IPS1": 1, "IPS2": 2, "IPS3": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool CLAM out-of-fold slide predictions from training result PKLs, "
            "aggregate them into PanNET IPS predictions, and summarize metrics across seeds."
        )
    )

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--run-id-template", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--method-name", default=None)

    return parser.parse_args()


def normalize_slide_id(x: object) -> str:
    s = str(x).strip()

    for ext in [".tiff", ".tif", ".svs", ".ndpi", ".mrxs"]:
        if s.lower().endswith(ext):
            return s[: -len(ext)]

    return s


def normalize_ips(x: object) -> str:
    s = str(x).strip().upper()

    mapping = {
        "A": "IPS1",
        "B": "IPS2",
        "C": "IPS3",
        "1": "IPS1",
        "2": "IPS2",
        "3": "IPS3",
        "IPS1": "IPS1",
        "IPS2": "IPS2",
        "IPS3": "IPS3",
    }

    if s not in mapping:
        raise ValueError(f"Unknown IPS label: {x}")

    return mapping[s]


def parse_pannet_slide_number(slide_id: str) -> int | None:
    match = re.match(r"^#(?P<case>\d+)-(?P<slide>\d+)\s+", slide_id)

    if not match:
        return None

    return int(match.group("slide"))


def ips_from_grade_sum(grade_sum: float) -> str:
    if grade_sum <= 6:
        return "IPS1"

    if grade_sum <= 9:
        return "IPS2"

    return "IPS3"


def load_training_pkl(pkl_path: Path, fold: int) -> pd.DataFrame:
    with pkl_path.open("rb") as f:
        results = pickle.load(f)

    rows = []

    for key, item in results.items():
        slide_id = str(key)

        if isinstance(item, dict):
            if "slide_id" in item:
                raw_slide_id = np.asarray(item["slide_id"]).reshape(-1)[0]
                slide_id = str(raw_slide_id)

            prob = np.asarray(item["prob"]).reshape(-1)
            label = int(np.asarray(item["label"]).reshape(-1)[0])
        else:
            raise TypeError(
                f"Unsupported result entry type in {pkl_path}: {type(item)}"
            )

        pred_0based = int(prob.argmax())
        true_grade = label + 1
        pred_grade = pred_0based + 1
        expected_grade = float(sum((i + 1) * float(p) for i, p in enumerate(prob)))

        row = {
            "fold": fold,
            "slide_id": normalize_slide_id(slide_id),
            "true_grade_from_pkl": true_grade,
            "pred_grade": pred_grade,
            "expected_grade": expected_grade,
        }

        for i, p in enumerate(prob):
            row[f"prob_grade_{i + 1}"] = float(p)

        rows.append(row)

    return pd.DataFrame(rows)


def find_fold_pkl(run_dir: Path, fold: int) -> Path:
    matches = sorted(run_dir.glob(f"results/**/split_{fold}_results.pkl"))

    if not matches:
        raise FileNotFoundError(
            f"No split_{fold}_results.pkl found under {run_dir}/results"
        )

    if len(matches) > 1:
        joined = "\n".join(str(x) for x in matches)
        raise RuntimeError(
            f"Multiple split_{fold}_results.pkl files found for one fold:\n{joined}"
        )

    return matches[0]


def prepare_dataset(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)

    required = {"case_id", "slide_id", "grade", "ips"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            "Dataset CSV missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["slide_id"] = df["slide_id"].map(normalize_slide_id)
    df["case_id"] = df["case_id"].astype(str)
    df["grade"] = df["grade"].astype(int)
    df["true_ips"] = df["ips"].map(normalize_ips)

    if "slide_number" not in df.columns:
        df["slide_number"] = df["slide_id"].map(parse_pannet_slide_number)

    return df[
        ["case_id", "slide_id", "slide_number", "grade", "true_ips"]
    ].copy()


def build_case_predictions(
    slide_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = slide_predictions.merge(
        dataset,
        on="slide_id",
        how="left",
        validate="many_to_one",
    )

    missing_meta = merged[merged["case_id"].isna()].copy()

    if not missing_meta.empty:
        merged = merged[merged["case_id"].notna()].copy()

    grade_mismatch = merged[
        merged["true_grade_from_pkl"].astype(int) != merged["grade"].astype(int)
    ].copy()

    rows = []
    skipped = []

    for case_id, group in merged.groupby("case_id", sort=True):
        true_ips_values = sorted(group["true_ips"].dropna().unique())

        if len(true_ips_values) != 1:
            skipped.append(
                {
                    "case_id": case_id,
                    "reason": "inconsistent_or_missing_true_ips",
                    "slides_seen": len(group),
                }
            )
            continue

        reps = group[group["slide_number"].isin([1, 2, 3])].copy()

        if set(reps["slide_number"].dropna().astype(int)) != {1, 2, 3}:
            skipped.append(
                {
                    "case_id": case_id,
                    "reason": "missing_required_slides_1_2_3",
                    "slides_seen": len(group),
                }
            )
            continue

        if reps["slide_number"].duplicated().any():
            skipped.append(
                {
                    "case_id": case_id,
                    "reason": "duplicate_required_slide_number",
                    "slides_seen": len(group),
                }
            )
            continue

        reps = reps.sort_values("slide_number")

        true_grade_sum = int(reps["grade"].sum())
        pred_grade_sum = int(reps["pred_grade"].sum())
        expected_grade_sum = float(reps["expected_grade"].sum())

        rows.append(
            {
                "case_id": case_id,
                "fold": int(reps["fold"].iloc[0]),
                "true_ips": true_ips_values[0],
                "true_ips_num": IPS_TO_NUM[true_ips_values[0]],
                "pred_ips": ips_from_grade_sum(pred_grade_sum),
                "pred_ips_num": IPS_TO_NUM[ips_from_grade_sum(pred_grade_sum)],
                "soft_pred_ips": ips_from_grade_sum(expected_grade_sum),
                "soft_pred_ips_num": IPS_TO_NUM[ips_from_grade_sum(expected_grade_sum)],
                "true_grade_sum": true_grade_sum,
                "pred_grade_sum": pred_grade_sum,
                "expected_grade_sum": expected_grade_sum,
                "slide_ids": "|".join(reps["slide_id"].astype(str)),
                "true_grades": "|".join(reps["grade"].astype(str)),
                "pred_grades": "|".join(reps["pred_grade"].astype(str)),
            }
        )

    skipped_df = pd.DataFrame(skipped)

    return pd.DataFrame(rows), skipped_df, grade_mismatch


def compute_metrics(case_df: pd.DataFrame, pred_col: str = "pred_ips") -> dict[str, float]:
    y_true = case_df["true_ips"].astype(str)
    y_pred = case_df[pred_col].astype(str)

    y_true_num = y_true.map(IPS_TO_NUM)
    y_pred_num = y_pred.map(IPS_TO_NUM)

    per_class = f1_score(
        y_true,
        y_pred,
        labels=IPS_LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "n_cases": float(len(case_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "IPS1_F1": float(per_class[0]),
        "IPS2_F1": float(per_class[1]),
        "IPS3_F1": float(per_class[2]),
        "macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=IPS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=IPS_LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "MAE": float(mean_absolute_error(y_true_num, y_pred_num)),
        "QWK": float(
            cohen_kappa_score(
                y_true_num,
                y_pred_num,
                labels=[1, 2, 3],
                weights="quadratic",
            )
        ),
    }


def format_mean_std(mean: float, std: float, metric: str) -> str:
    if metric in {
        "accuracy",
        "IPS1_F1",
        "IPS2_F1",
        "IPS3_F1",
        "macro_F1",
        "weighted_F1",
    }:
        return f"{mean * 100:.2f} ± {std * 100:.2f}"

    return f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    args = parse_args()

    repo_dir = Path.cwd()
    config_path = args.config if args.config.is_absolute() else repo_dir / args.config
    cfg = json.loads(config_path.read_text())

    dataset_name = cfg["dataset_name"]
    dataset_csv = repo_dir / cfg["dataset_csv"]
    k = int(cfg["k"])
    seeds = args.seeds if args.seeds else [int(x) for x in cfg["seeds"]]

    training_cfg = cfg.get("training", {})
    feature_run_id = training_cfg.get("feature_run_id", "pannet_virchow2_40x1024")
    model_type = training_cfg.get("model_type", "clam_mb")

    run_id_template = args.run_id_template
    if run_id_template is None:
        run_id_template = (
            "{feature_run_id}_{dataset_name}_grade_{model_type}"
            "_cv{k}_split{seed}_train{seed}"
        )

    method_name = args.method_name
    if method_name is None:
        method_name = f"{model_type} + {feature_run_id}"

    out_dir = args.out_dir
    if out_dir is None:
        safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_name)
        out_dir = (
            repo_dir
            / "runs"
            / "evaluation"
            / "ips_oof"
            / f"{dataset_name}_{safe_method}_cv{k}"
        )
    elif not out_dir.is_absolute():
        out_dir = repo_dir / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(dataset_csv)

    seed_metric_rows = []
    all_case_rows = []
    all_slide_rows = []

    for seed in seeds:
        run_id = run_id_template.format(
            seed=seed,
            split_seed=seed,
            train_seed=seed,
            feature_run_id=feature_run_id,
            dataset_name=dataset_name,
            model_type=model_type,
            k=k,
        )

        run_dir = repo_dir / "runs" / "training" / run_id

        if not run_dir.is_dir():
            raise SystemExit(f"Missing training run directory: {run_dir}")

        slide_frames = []

        for fold in range(k):
            pkl_path = find_fold_pkl(run_dir, fold)
            fold_df = load_training_pkl(pkl_path, fold)
            fold_df["seed"] = seed
            fold_df["training_run_id"] = run_id
            fold_df["result_pkl"] = str(pkl_path)
            slide_frames.append(fold_df)

        seed_slide_df = pd.concat(slide_frames, ignore_index=True)

        duplicate_slide_predictions = seed_slide_df[
            seed_slide_df.duplicated(subset=["slide_id"], keep=False)
        ]

        if not duplicate_slide_predictions.empty:
            duplicates_path = out_dir / f"seed_{seed}_duplicate_slide_predictions.csv"
            duplicate_slide_predictions.to_csv(duplicates_path, index=False)
            raise SystemExit(
                f"Seed {seed} has duplicate OOF slide predictions across folds. "
                f"Saved duplicates to {duplicates_path}"
            )

        case_df, skipped_df, grade_mismatch_df = build_case_predictions(
            seed_slide_df,
            dataset,
        )

        if case_df.empty:
            raise SystemExit(f"No case-level IPS predictions produced for seed {seed}")

        hard_metrics = compute_metrics(case_df, pred_col="pred_ips")
        soft_metrics = compute_metrics(case_df, pred_col="soft_pred_ips")

        hard_metrics.update(
            {
                "seed": seed,
                "training_run_id": run_id,
                "prediction_type": "hard_grade_sum",
            }
        )

        soft_metrics.update(
            {
                "seed": seed,
                "training_run_id": run_id,
                "prediction_type": "soft_expected_grade_sum",
            }
        )

        seed_metric_rows.append(hard_metrics)
        seed_metric_rows.append(soft_metrics)

        case_df["seed"] = seed
        case_df["training_run_id"] = run_id
        seed_slide_df["seed"] = seed

        case_df.to_csv(out_dir / f"seed_{seed}_case_oof_predictions.csv", index=False)
        seed_slide_df.to_csv(out_dir / f"seed_{seed}_slide_oof_predictions.csv", index=False)
        skipped_df.to_csv(out_dir / f"seed_{seed}_skipped_cases.csv", index=False)
        grade_mismatch_df.to_csv(out_dir / f"seed_{seed}_grade_mismatches.csv", index=False)

        cm = confusion_matrix(
            case_df["true_ips"],
            case_df["pred_ips"],
            labels=IPS_LABELS,
        )
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in IPS_LABELS],
            columns=[f"pred_{x}" for x in IPS_LABELS],
        )
        cm_df.to_csv(out_dir / f"seed_{seed}_confusion_matrix.csv")

        all_case_rows.append(case_df)
        all_slide_rows.append(seed_slide_df)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_metrics = seed_metrics[
        [
            "seed",
            "prediction_type",
            "training_run_id",
            "n_cases",
            "accuracy",
            "IPS1_F1",
            "IPS2_F1",
            "IPS3_F1",
            "macro_F1",
            "weighted_F1",
            "MAE",
            "QWK",
        ]
    ]

    seed_metrics.to_csv(out_dir / "seed_metrics.csv", index=False)

    metric_cols = [
        "accuracy",
        "IPS1_F1",
        "IPS2_F1",
        "IPS3_F1",
        "macro_F1",
        "weighted_F1",
        "MAE",
        "QWK",
    ]

    summary_rows = []

    for prediction_type, group in seed_metrics.groupby("prediction_type"):
        row = {
            "method": method_name,
            "prediction_type": prediction_type,
            "n_seeds": len(group),
            "mean_n_cases": group["n_cases"].mean(),
        }

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{metric}_var"] = group[metric].var(ddof=1) if len(group) > 1 else 0.0

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_mean_std_var.csv", index=False)

    thesis_rows = []

    for _, row in summary.iterrows():
        out = {
            "Method": row["method"],
            "Prediction": row["prediction_type"],
            "IPS1 F1": format_mean_std(row["IPS1_F1_mean"], row["IPS1_F1_std"], "IPS1_F1"),
            "IPS2 F1": format_mean_std(row["IPS2_F1_mean"], row["IPS2_F1_std"], "IPS2_F1"),
            "IPS3 F1": format_mean_std(row["IPS3_F1_mean"], row["IPS3_F1_std"], "IPS3_F1"),
            "Macro F1": format_mean_std(row["macro_F1_mean"], row["macro_F1_std"], "macro_F1"),
            "Weighted F1": format_mean_std(row["weighted_F1_mean"], row["weighted_F1_std"], "weighted_F1"),
            "MAE": format_mean_std(row["MAE_mean"], row["MAE_std"], "MAE"),
            "QWK": format_mean_std(row["QWK_mean"], row["QWK_std"], "QWK"),
        }

        thesis_rows.append(out)

    thesis_table = pd.DataFrame(thesis_rows)
    thesis_table.to_csv(out_dir / "thesis_style_table.csv", index=False)

    pd.concat(all_case_rows, ignore_index=True).to_csv(
        out_dir / "all_case_oof_predictions.csv",
        index=False,
    )

    pd.concat(all_slide_rows, ignore_index=True).to_csv(
        out_dir / "all_slide_oof_predictions.csv",
        index=False,
    )

    print()
    print("Seed metrics:")
    print(seed_metrics.to_string(index=False))
    print()
    print("Thesis-style table:")
    print(thesis_table.to_string(index=False))
    print()
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
