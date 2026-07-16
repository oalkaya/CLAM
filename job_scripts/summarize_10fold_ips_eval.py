#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


GRADE_LABELS = [1, 2, 3, 4, 5]
IPS_LABELS = ["IPS1", "IPS2", "IPS3"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def metric_summary(values):
    values = pd.Series(values, dtype=float)
    return {
        "mean": values.mean(),
        "std": values.std(ddof=1) if len(values) > 1 else 0.0,
        "min": values.min(),
        "max": values.max(),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    all_slide_preds = []
    all_case_preds = []

    for fold in range(args.n_folds):
        fold_dir = args.eval_root / f"fold_{fold}_{args.split}_ips"

        slide_path = fold_dir / "slide_predictions.csv"
        case_path = fold_dir / "case_ips_predictions.csv"

        if not slide_path.is_file():
            print(f"WARNING: missing {slide_path}")
            continue

        if not case_path.is_file():
            print(f"WARNING: missing {case_path}")
            continue

        slide_df = pd.read_csv(slide_path)
        case_df = pd.read_csv(case_path)

        slide_df["fold"] = fold
        case_df["fold"] = fold

        all_slide_preds.append(slide_df)
        all_case_preds.append(case_df)

        slide_acc = accuracy_score(slide_df["true_grade"], slide_df["pred_grade"])
        slide_macro_f1 = f1_score(
            slide_df["true_grade"],
            slide_df["pred_grade"],
            labels=GRADE_LABELS,
            average="macro",
            zero_division=0,
        )
        slide_weighted_f1 = f1_score(
            slide_df["true_grade"],
            slide_df["pred_grade"],
            labels=GRADE_LABELS,
            average="weighted",
            zero_division=0,
        )

        if len(case_df) > 0:
            ips_acc = accuracy_score(case_df["true_ips"], case_df["pred_ips"])
            ips_macro_f1 = f1_score(
                case_df["true_ips"],
                case_df["pred_ips"],
                labels=IPS_LABELS,
                average="macro",
                zero_division=0,
            )
            ips_weighted_f1 = f1_score(
                case_df["true_ips"],
                case_df["pred_ips"],
                labels=IPS_LABELS,
                average="weighted",
                zero_division=0,
            )
        else:
            ips_acc = float("nan")
            ips_macro_f1 = float("nan")
            ips_weighted_f1 = float("nan")

        fold_rows.append({
            "fold": fold,
            "n_slide_predictions": len(slide_df),
            "n_case_predictions": len(case_df),
            "slide_accuracy": slide_acc,
            "slide_macro_f1": slide_macro_f1,
            "slide_weighted_f1": slide_weighted_f1,
            "ips_accuracy": ips_acc,
            "ips_macro_f1": ips_macro_f1,
            "ips_weighted_f1": ips_weighted_f1,
        })

    fold_metrics = pd.DataFrame(fold_rows)

    if fold_metrics.empty:
        raise SystemExit("No fold metrics found.")

    fold_metrics.to_csv(args.out_dir / "fold_metrics.csv", index=False)

    summary_rows = []

    for metric in [
        "slide_accuracy",
        "slide_macro_f1",
        "slide_weighted_f1",
        "ips_accuracy",
        "ips_macro_f1",
        "ips_weighted_f1",
    ]:
        stats = metric_summary(fold_metrics[metric].dropna())
        summary_rows.append({
            "metric": metric,
            **stats,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary_mean_std.csv", index=False)

    if all_slide_preds:
        all_slide = pd.concat(all_slide_preds, ignore_index=True)
        all_slide.to_csv(args.out_dir / "all_slide_predictions.csv", index=False)

    if all_case_preds:
        all_case = pd.concat(all_case_preds, ignore_index=True)
        all_case.to_csv(args.out_dir / "all_case_ips_predictions.csv", index=False)

        cm = confusion_matrix(
            all_case["true_ips"],
            all_case["pred_ips"],
            labels=IPS_LABELS,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{x}" for x in IPS_LABELS],
            columns=[f"pred_{x}" for x in IPS_LABELS],
        )
        cm_df.to_csv(args.out_dir / "pooled_case_ips_confusion_matrix.csv")

        with open(args.out_dir / "pooled_case_ips_report.txt", "w") as f:
            f.write(
                classification_report(
                    all_case["true_ips"],
                    all_case["pred_ips"],
                    labels=IPS_LABELS,
                    zero_division=0,
                )
            )
            f.write("\n\n")
            f.write(str(cm_df))
            f.write("\n")

    print()
    print("Per-fold metrics:")
    print(fold_metrics.to_string(index=False))

    print()
    print("Mean ± std:")
    for _, row in summary.iterrows():
        print(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}")

    print()
    print(f"Wrote summary to: {args.out_dir}")


if __name__ == "__main__":
    main()
