#!/usr/bin/env python

from pathlib import Path
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


IPS_LABELS = ["IPS1", "IPS2", "IPS3"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--split", default="test")
    p.add_argument("--prediction_col", default="pred_ips", choices=["pred_ips", "soft_pred_ips"])
    return p.parse_args()


def mean_std(values):
    s = pd.Series(values, dtype=float).dropna()
    return s.mean(), s.std(ddof=1)


def fmt(mean, std):
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for fold in range(args.n_folds):
        case_path = args.eval_root / f"fold_{fold}_{args.split}_ips" / "case_ips_predictions.csv"

        if not case_path.exists():
            print(f"WARNING: missing {case_path}")
            continue

        df = pd.read_csv(case_path)

        if len(df) == 0:
            print(f"WARNING: no IPS cases in fold {fold}")
            continue

        y_true = df["true_ips"]
        y_pred = df[args.prediction_col]

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            labels=IPS_LABELS,
            average=None,
            zero_division=0,
        )

        rows.append({
            "fold": fold,
            "n_cases": len(df),
            "accuracy": accuracy_score(y_true, y_pred),
            "IPS1_F1": per_class_f1[0],
            "IPS2_F1": per_class_f1[1],
            "IPS3_F1": per_class_f1[2],
            "macro_F1": f1_score(y_true, y_pred, labels=IPS_LABELS, average="macro", zero_division=0),
            "weighted_F1": f1_score(y_true, y_pred, labels=IPS_LABELS, average="weighted", zero_division=0),
        })

    fold_df = pd.DataFrame(rows)

    if fold_df.empty:
        raise SystemExit("No fold results found.")

    fold_df.to_csv(args.out_dir / f"fold_metrics_{args.prediction_col}.csv", index=False)

    metric_cols = ["IPS1_F1", "IPS2_F1", "IPS3_F1", "macro_F1", "weighted_F1", "accuracy"]

    summary = []
    pretty = {}

    for metric in metric_cols:
        mean, std = mean_std(fold_df[metric])
        summary.append({
            "metric": metric,
            "mean": mean,
            "std": std,
            "mean_plus_minus_std_percent": fmt(mean, std),
        })
        pretty[metric] = fmt(mean, std)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir / f"summary_{args.prediction_col}.csv", index=False)

    thesis_table = pd.DataFrame([{
        "Method": f"CLAM-MB Virchow2 ({args.prediction_col})",
        "IPS1 F1": pretty["IPS1_F1"],
        "IPS2 F1": pretty["IPS2_F1"],
        "IPS3 F1": pretty["IPS3_F1"],
        "Macro F1": pretty["macro_F1"],
        "Weighted F1": pretty["weighted_F1"],
        "Accuracy": pretty["accuracy"],
    }])

    thesis_table.to_csv(args.out_dir / f"thesis_style_table_{args.prediction_col}.csv", index=False)

    print()
    print("Per-fold metrics:")
    print(fold_df.to_string(index=False))

    print()
    print("Thesis-style table:")
    print(thesis_table.to_string(index=False))

    print()
    print("Saved:")
    print(args.out_dir / f"fold_metrics_{args.prediction_col}.csv")
    print(args.out_dir / f"summary_{args.prediction_col}.csv")
    print(args.out_dir / f"thesis_style_table_{args.prediction_col}.csv")


if __name__ == "__main__":
    main()
