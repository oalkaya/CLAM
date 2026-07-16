#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make imports work whether this file is run from repo root or elsewhere.
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

from models.model_clam import CLAM_MB, CLAM_SB
from models.model_mil import MIL_fc_mc


GRADE_LABELS = [1, 2, 3, 4, 5]
IPS_LABELS = ["IPS1", "IPS2", "IPS3"]
IPS_TO_INT = {"IPS1": 1, "IPS2": 2, "IPS3": 3}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature_dir", required=True, type=Path)
    parser.add_argument("--dataset_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)

    parser.add_argument("--model_type", required=True, choices=["mil", "clam_sb", "clam_mb"])
    parser.add_argument("--embed_dim", required=True, type=int)
    parser.add_argument("--n_classes", default=5, type=int)

    parser.add_argument("--drop_out", default=0.25, type=float)
    parser.add_argument("--model_size", default="small", choices=["small", "big"])
    parser.add_argument("--B", default=8, type=int)
    parser.add_argument("--subtyping", action="store_true")

    parser.add_argument("--split_csv", default=None, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    return parser.parse_args()


def normalize_slide_id(value) -> str:
    value = str(value).strip()

    for suffix in [".tiff", ".tif", ".svs", ".ndpi", ".mrxs", ".h5", ".pt"]:
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]

    return value


def get_slide_number(slide_id: str):
    """
    Extract slide number from IDs like:
      #49-1 7815-B7463-1F
      #44-10 B23-06684-B48
    """
    match = re.match(r"#\d+-(\d+)\s", str(slide_id))
    return int(match.group(1)) if match else None


def normalize_ips(value) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip().upper().replace(" ", "").replace("_", "")

    if s in {"A", "1", "IPS1"}:
        return "IPS1"
    if s in {"B", "2", "IPS2"}:
        return "IPS2"
    if s in {"C", "3", "IPS3"}:
        return "IPS3"

    return None


def grade_sum_to_ips(grade_sum: int) -> str:
    """
    Thesis-style hard IPS mapping:
      3-6   -> IPS1
      7-9   -> IPS2
      10-15 -> IPS3
    """
    if 3 <= grade_sum <= 6:
        return "IPS1"
    if 7 <= grade_sum <= 9:
        return "IPS2"
    if 10 <= grade_sum <= 15:
        return "IPS3"

    raise ValueError(f"Invalid 3-slide grade sum for IPS: {grade_sum}")


def expected_grade_sum_to_ips(expected_sum: float) -> str:
    """
    Soft IPS from summed expected grades.

    Since the hard integer boundaries are:
      IPS1: sums 3-6
      IPS2: sums 7-9
      IPS3: sums 10-15

    the midpoint thresholds are:
      <= 6.5 -> IPS1
      <= 9.5 -> IPS2
      > 9.5  -> IPS3
    """
    if expected_sum <= 6.5:
        return "IPS1"
    if expected_sum <= 9.5:
        return "IPS2"
    return "IPS3"


def boolish_series(series: pd.Series) -> bool:
    values = set(series.dropna().astype(str).str.strip().str.lower().unique())
    return bool(values) and values.issubset({"true", "false", "1", "0"})


def series_to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def get_split_slide_ids(split_csv: Path, split: str) -> set[str]:
    """
    Supports both CLAM split formats:

    1. Normal split file:
       train,val,test
       slideA,slideB,slideC
       ...

    2. Boolean split file:
       ,train,val,test
       slideA,True,False,False
       ...
    """
    split_df = pd.read_csv(split_csv)

    if split not in split_df.columns:
        raise ValueError(
            f"Split CSV does not contain column '{split}'. "
            f"Columns: {list(split_df.columns)}"
        )

    # Boolean style: split column contains True/False.
    if boolish_series(split_df[split]):
        id_candidates = [c for c in split_df.columns if c not in {"train", "val", "test"}]

        if "slide_id" in split_df.columns:
            id_col = "slide_id"
        elif id_candidates:
            id_col = id_candidates[0]
        else:
            raise ValueError("Could not identify slide-id column in boolean split CSV.")

        selected = split_df.loc[series_to_bool(split_df[split]), id_col]
        return {normalize_slide_id(v) for v in selected.dropna().astype(str)}

    # Standard CLAM style: split column contains slide IDs.
    values = split_df[split].dropna().astype(str).tolist()
    return {normalize_slide_id(v) for v in values}


def build_model(args):
    instance_loss_fn = torch.nn.CrossEntropyLoss()

    if args.model_type == "mil":
        model = MIL_fc_mc(
            dropout=args.drop_out,
            n_classes=args.n_classes,
            embed_dim=args.embed_dim,
        )

    elif args.model_type == "clam_sb":
        model = CLAM_SB(
            gate=True,
            size_arg=args.model_size,
            dropout=args.drop_out,
            k_sample=args.B,
            n_classes=args.n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=args.subtyping,
            embed_dim=args.embed_dim,
        )

    elif args.model_type == "clam_mb":
        model = CLAM_MB(
            gate=True,
            size_arg=args.model_size,
            dropout=args.drop_out,
            k_sample=args.B,
            n_classes=args.n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=args.subtyping,
            embed_dim=args.embed_dim,
        )

    else:
        raise ValueError(args.model_type)

    return model


def load_state_dict(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)


def read_feature_tensor(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.float()

    if isinstance(obj, dict) and "features" in obj:
        return obj["features"].float()

    raise TypeError(f"Expected Tensor or dict with 'features' in {path}")


def predict_one_slide(model, features: torch.Tensor, device: torch.device):
    features = features.to(device)

    with torch.no_grad():
        output = model(features)

    # CLAM/MIL forward outputs start with:
    # logits, Y_prob, Y_hat, ...
    probs = output[1].detach().cpu().squeeze(0)

    pred_class = int(torch.argmax(probs).item())
    pred_grade = pred_class + 1

    expected_grade = float(
        sum((idx + 1) * float(prob) for idx, prob in enumerate(probs))
    )

    return pred_class, pred_grade, expected_grade, probs.numpy()


def add_metric_lines_for_grade(metrics_lines, y_true, y_pred, prefix="Slide-level grade"):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=GRADE_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=GRADE_LABELS,
        average="weighted",
        zero_division=0,
    )
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, labels=GRADE_LABELS, weights="quadratic")

    metrics_lines.append(prefix)
    metrics_lines.append(f"Accuracy: {acc:.4f}")
    metrics_lines.append(f"Macro F1: {macro_f1:.4f}")
    metrics_lines.append(f"Weighted F1: {weighted_f1:.4f}")
    metrics_lines.append(f"MAE: {mae:.4f}")
    metrics_lines.append(f"QWK: {qwk:.4f}")
    metrics_lines.append("")
    metrics_lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=GRADE_LABELS,
            zero_division=0,
        )
    )


def add_metric_lines_for_ips(metrics_lines, y_true, y_pred, prefix="Case-level IPS"):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=IPS_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=IPS_LABELS,
        average="weighted",
        zero_division=0,
    )

    y_true_int = [IPS_TO_INT[x] for x in y_true]
    y_pred_int = [IPS_TO_INT[x] for x in y_pred]

    mae = mean_absolute_error(y_true_int, y_pred_int)
    qwk = cohen_kappa_score(y_true_int, y_pred_int, labels=[1, 2, 3], weights="quadratic")

    metrics_lines.append(prefix)
    metrics_lines.append(f"Accuracy: {acc:.4f}")
    metrics_lines.append(f"Macro F1: {macro_f1:.4f}")
    metrics_lines.append(f"Weighted F1: {weighted_f1:.4f}")
    metrics_lines.append(f"MAE: {mae:.4f}")
    metrics_lines.append(f"QWK: {qwk:.4f}")
    metrics_lines.append("")
    metrics_lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=IPS_LABELS,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=IPS_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in IPS_LABELS],
        columns=[f"pred_{x}" for x in IPS_LABELS],
    )

    metrics_lines.append("")
    metrics_lines.append("IPS confusion matrix")
    metrics_lines.append(str(cm_df))

    return cm_df


def main():
    args = parse_args()

    pt_dir = args.feature_dir / "pt_files"

    if not pt_dir.is_dir():
        raise SystemExit(f"Missing pt_files directory: {pt_dir}")

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    if not args.dataset_csv.is_file():
        raise SystemExit(f"Missing dataset CSV: {args.dataset_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = pd.read_csv(args.dataset_csv)

    if "slide_id" not in dataset.columns:
        if "filename" in dataset.columns:
            dataset["slide_id"] = dataset["filename"].map(normalize_slide_id)
        else:
            raise SystemExit("Dataset CSV needs either 'slide_id' or 'filename' column.")

    required_cols = {"case_id", "slide_id", "grade", "ips"}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        raise SystemExit(f"Dataset CSV missing columns: {sorted(missing_cols)}")

    dataset["slide_id"] = dataset["slide_id"].map(normalize_slide_id)
    dataset["true_grade"] = dataset["grade"].astype(int)
    dataset["true_ips_norm"] = dataset["ips"].map(normalize_ips)

    if args.split_csv is not None and args.split != "all":
        if not args.split_csv.is_file():
            raise SystemExit(f"Missing split CSV: {args.split_csv}")

        split_ids = get_split_slide_ids(args.split_csv, args.split)
        dataset = dataset[dataset["slide_id"].isin(split_ids)].copy()

    if dataset.empty:
        raise SystemExit("No slides left after split filtering.")

    model = build_model(args)
    load_state_dict(model, args.checkpoint)
    model.to(device)
    model.eval()

    slide_rows = []

    for _, row in dataset.iterrows():
        slide_id = row["slide_id"]
        pt_path = pt_dir / f"{slide_id}.pt"

        if not pt_path.is_file():
            raise FileNotFoundError(f"Missing feature bag: {pt_path}")

        features = read_feature_tensor(pt_path)

        if features.ndim != 2:
            raise ValueError(f"{pt_path} should be 2D, got shape {tuple(features.shape)}")

        if features.shape[1] != args.embed_dim:
            raise ValueError(
                f"{pt_path} has embed_dim {features.shape[1]}, "
                f"but --embed_dim is {args.embed_dim}"
            )

        pred_class, pred_grade, expected_grade, probs = predict_one_slide(
            model,
            features,
            device,
        )

        out = {
            "case_id": row["case_id"],
            "slide_id": slide_id,
            "slide_number": get_slide_number(slide_id),
            "true_grade": int(row["true_grade"]),
            "true_ips": row["true_ips_norm"],
            "pred_class_0based": pred_class,
            "pred_grade": pred_grade,
            "expected_grade": expected_grade,
            "num_patches": int(features.shape[0]),
        }

        for idx, prob in enumerate(probs):
            out[f"prob_grade_{idx + 1}"] = float(prob)

        slide_rows.append(out)

    slide_pred = pd.DataFrame(slide_rows)
    slide_pred.to_csv(args.out_dir / "slide_predictions.csv", index=False)

    case_rows = []
    skipped_cases = []

    for case_id, group in slide_pred.groupby("case_id"):
        group = group.copy()

        true_ips_values = sorted(v for v in group["true_ips"].dropna().unique())

        if len(true_ips_values) != 1:
            skipped_cases.append((case_id, len(group), "missing_or_inconsistent_true_ips"))
            continue

        # Thesis-style IPS aggregation:
        # use only representative slides #case-1, #case-2, #case-3.
        # Extra slides #case-4+ are ignored for IPS aggregation.
        ips_group = group[group["slide_number"].isin([1, 2, 3])].copy()

        if len(ips_group) != 3:
            skipped_cases.append((case_id, len(group), "missing_one_of_slides_1_2_3"))
            continue

        ips_group = ips_group.sort_values("slide_number")

        pred_grade_sum = int(ips_group["pred_grade"].sum())
        true_grade_sum = int(ips_group["true_grade"].sum())
        expected_grade_sum = float(ips_group["expected_grade"].sum())

        case_rows.append(
            {
                "case_id": case_id,
                "n_available_slides_in_split": len(group),
                "n_ips_slides_used": len(ips_group),
                "slide_ids": "|".join(ips_group["slide_id"].astype(str)),
                "true_grades": "|".join(ips_group["true_grade"].astype(str)),
                "pred_grades": "|".join(ips_group["pred_grade"].astype(str)),
                "true_grade_sum": true_grade_sum,
                "pred_grade_sum": pred_grade_sum,
                "expected_grade_sum": expected_grade_sum,
                "true_ips": true_ips_values[0],
                "pred_ips": grade_sum_to_ips(pred_grade_sum),
                "soft_pred_ips": expected_grade_sum_to_ips(expected_grade_sum),
            }
        )

    case_pred = pd.DataFrame(case_rows)
    case_pred.to_csv(args.out_dir / "case_ips_predictions.csv", index=False)

    skipped_df = pd.DataFrame(
        skipped_cases,
        columns=["case_id", "n_available_slides_in_split", "reason"],
    )
    skipped_df.to_csv(args.out_dir / "skipped_cases.csv", index=False)

    metrics_lines = []

    metrics_lines.append(f"Checkpoint: {args.checkpoint}")
    metrics_lines.append(f"Feature dir: {args.feature_dir}")
    metrics_lines.append(f"Dataset CSV: {args.dataset_csv}")
    metrics_lines.append(f"Split CSV: {args.split_csv}")
    metrics_lines.append(f"Split: {args.split}")
    metrics_lines.append(f"Model type: {args.model_type}")
    metrics_lines.append(f"Embed dim: {args.embed_dim}")
    metrics_lines.append(f"Device: {device}")
    metrics_lines.append("")
    metrics_lines.append(f"Slides evaluated: {len(slide_pred)}")
    metrics_lines.append(f"Cases evaluated for IPS: {len(case_pred)}")
    metrics_lines.append(f"Cases skipped for IPS: {len(skipped_df)}")
    metrics_lines.append("")

    add_metric_lines_for_grade(
        metrics_lines,
        slide_pred["true_grade"],
        slide_pred["pred_grade"],
        prefix="Slide-level grade metrics",
    )

    if len(case_pred) > 0:
        metrics_lines.append("")
        cm_hard = add_metric_lines_for_ips(
            metrics_lines,
            case_pred["true_ips"],
            case_pred["pred_ips"],
            prefix="Case-level IPS metrics, hard argmax-grade aggregation",
        )
        cm_hard.to_csv(args.out_dir / "case_ips_confusion_matrix.csv")

        metrics_lines.append("")
        cm_soft = add_metric_lines_for_ips(
            metrics_lines,
            case_pred["true_ips"],
            case_pred["soft_pred_ips"],
            prefix="Case-level IPS metrics, soft expected-grade aggregation",
        )
        cm_soft.to_csv(args.out_dir / "case_ips_soft_confusion_matrix.csv")

    else:
        metrics_lines.append("")
        metrics_lines.append("Case-level IPS metrics")
        metrics_lines.append("No cases evaluated.")
        metrics_lines.append(
            "Reason: no evaluated case had representative slides 1, 2, and 3 "
            "with a valid consistent true IPS label."
        )

    metrics_text = "\n".join(metrics_lines)
    print(metrics_text)

    with open(args.out_dir / "metrics.txt", "w") as f:
        f.write(metrics_text)
        f.write("\n")


if __name__ == "__main__":
    main()