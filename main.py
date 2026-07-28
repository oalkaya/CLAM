#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.core_utils import train
from utils.file_utils import save_pkl


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_list(value: str) -> list[object]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) < 2:
        raise argparse.ArgumentTypeError("Expected a JSON list with at least two values.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLAM from explicit dataset settings.")
    parser.add_argument("--dataset_csv", type=Path, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--slide_id_col", required=True)
    parser.add_argument("--patient_id_col", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--label_values", type=json_list, required=True)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_data", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--drop_out", type=float, default=0.25)
    parser.add_argument("--bag_loss", choices=["svm", "ce"], default="ce")
    parser.add_argument(
        "--model_type",
        choices=["clam_sb", "clam_mb", "mil"],
        default="clam_sb",
    )
    parser.add_argument("--weighted_sample", action="store_true")
    parser.add_argument("--model_size", choices=["small", "big"], default="small")
    parser.add_argument("--no_inst_cluster", action="store_true")
    parser.add_argument("--inst_loss", choices=["svm", "ce"], default="ce")
    parser.add_argument("--subtyping", action="store_true")
    parser.add_argument("--bag_weight", type=float, default=0.7)
    parser.add_argument("--B", type=int, default=8)
    return parser.parse_args()


def seed_torch(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args: argparse.Namespace) -> None:
    split_path = args.split_dir / f"splits_{args.fold}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    if not args.dataset_csv.is_file():
        raise FileNotFoundError(f"Missing dataset CSV: {args.dataset_csv}")
    if not args.features_dir.is_dir():
        raise FileNotFoundError(f"Missing feature bag directory: {args.features_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.n_classes = len(args.label_values)
    args.label_frac = 1.0
    args.exp_code = args.results_dir.name

    label_dict = {str(value): index for index, value in enumerate(args.label_values)}
    dataset = Generic_MIL_Dataset(
        csv_path=str(args.dataset_csv),
        data_dir=str(args.features_dir),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=label_dict,
        label_col=args.label_col,
        patient_strat=False,
        patient_id_col=args.patient_id_col,
        slide_id_col=args.slide_id_col,
        ignore=[],
    )
    train_dataset, val_dataset, test_dataset = dataset.return_splits(
        from_id=False,
        csv_path=str(split_path),
    )

    if train_dataset is None or test_dataset is None:
        raise ValueError(f"Fold {args.fold} must contain train and test slides.")
    if val_dataset is None and args.early_stopping:
        raise ValueError("Early stopping requires a non-empty validation split.")

    seed_torch(args.seed)
    results, test_auc, val_auc, test_acc, val_acc = train(
        (train_dataset, val_dataset, test_dataset),
        args.fold,
        args,
    )
    save_pkl(
        args.results_dir / f"split_{args.fold}_results.pkl",
        results,
    )
    pd.DataFrame(
        [
            {
                "fold": args.fold,
                "test_auc": test_auc,
                "val_auc": val_auc,
                "test_acc": test_acc,
                "val_acc": val_acc,
            }
        ]
    ).to_csv(args.results_dir / f"fold_{args.fold}_metrics.csv", index=False)
    print(f"Finished fold {args.fold}.")


if __name__ == "__main__":
    main(parse_args())
