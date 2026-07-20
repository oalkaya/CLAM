#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def normalize_slide_id(x: str) -> str:
    x = str(x).strip()
    for suffix in [".tiff", ".tif", ".svs", ".ndpi", ".mrxs", ".h5", ".pt"]:
        if x.lower().endswith(suffix):
            return x[: -len(suffix)]
    return x


def normalize_pannet_ips(x):
    if pd.isna(x):
        return None

    s = str(x).strip().upper().replace(" ", "").replace("_", "")

    if s in {"A", "1", "IPS1"}:
        return "A"
    if s in {"B", "2", "IPS2"}:
        return "B"
    if s in {"C", "3", "IPS3"}:
        return "C"

    return None


def pannet_slide_number(slide_id: str):
    m = re.match(r"^#?\d+-(\d+)(?:\s|$)", str(slide_id))
    return int(m.group(1)) if m else None


TASKS = {
    "pannet": {
        "task_name": "task_pannet_grade",
        "dataset_csv": "dataset_csv/pannet_wsi_grade.csv",
        "group_col": "case_id",
        "slide_col": "slide_id",
        "stratify_col": "ips",
        "label_order": ["A", "B", "C"],
        "normalize_label": normalize_pannet_ips,

        # PanNET-specific IPS protocol.
        "slide_number_fn": pannet_slide_number,
        "required_eval_slide_numbers": {1, 2, 3},
        "val_test_slide_numbers": {1, 2, 3},
        "ineligible_group_policy": "train_only",
    },

    # Future example:
    # "tissuenet": {
    #     "task_name": "task_tissuenet_cervix",
    #     "dataset_csv": "dataset_csv/tissuenet.csv",
    #     "group_col": "slide_id",          # or patient_id if available
    #     "slide_col": "slide_id",
    #     "stratify_col": "label",
    #     "label_order": [0, 1, 2, 3],
    #     "normalize_label": lambda x: int(x) if not pd.isna(x) else None,
    #     "slide_number_fn": None,
    #     "required_eval_slide_numbers": None,
    #     "val_test_slide_numbers": None,
    #     "ineligible_group_policy": "drop",
    # },
}


def parse_args():
    p = argparse.ArgumentParser(description="Create stratified group-level CV splits in CLAM format.")
    p.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    p.add_argument("--config", required=True, type=Path)
    return p.parse_args()


def pad(xs, length):
    return list(xs) + [""] * (length - len(xs))


def save_split_csv(path: Path, train, val, test):
    max_len = max(len(train), len(val), len(test))

    out = pd.DataFrame(
        {
            "train": pad(train, max_len),
            "val": pad(val, max_len),
            "test": pad(test, max_len),
        }
    )

    out.to_csv(path, index=False)


def save_bool_csv(path: Path, train, val, test):
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    all_ids = list(train) + list(val) + list(test)

    out = pd.DataFrame(
        {
            "train": [x in train_set for x in all_ids],
            "val": [x in val_set for x in all_ids],
            "test": [x in test_set for x in all_ids],
        },
        index=all_ids,
    )

    out.to_csv(path)


def build_group_table(df: pd.DataFrame, spec: dict):
    group_col = spec["group_col"]
    strat_col = "_strat_label"
    slide_number_col = "_slide_number"

    rows = []

    for group_id, g in df.groupby(group_col, sort=False):
        labels = sorted(v for v in g[strat_col].dropna().unique())

        valid_label = len(labels) == 1 and labels[0] in spec["label_order"]

        has_required = True
        required_nums = spec.get("required_eval_slide_numbers")

        if required_nums is not None:
            counts = g[slide_number_col].value_counts().to_dict()
            has_required = all(counts.get(n, 0) == 1 for n in required_nums)

        eligible = valid_label and has_required

        rows.append(
            {
                "group_id": str(group_id),
                "stratify_label": labels[0] if valid_label else None,
                "valid_label": valid_label,
                "has_required_eval_slides": has_required,
                "eligible_for_eval": eligible,
                "n_slides": len(g),
            }
        )

    return pd.DataFrame(rows)


def slides_for_groups(df: pd.DataFrame, group_ids, spec: dict, val_or_test: bool):
    group_ids = set(str(x) for x in group_ids)
    sub = df[df[spec["group_col"]].astype(str).isin(group_ids)].copy()

    allowed_nums = spec.get("val_test_slide_numbers")

    if val_or_test and allowed_nums is not None:
        sub = sub[sub["_slide_number"].isin(allowed_nums)].copy()

    return sub[spec["slide_col"]].astype(str).tolist()


def check_disjoint(train_groups, val_groups, test_groups, seed, fold):
    train_groups = set(train_groups)
    val_groups = set(val_groups)
    test_groups = set(test_groups)

    if train_groups & val_groups:
        raise RuntimeError(f"Seed {seed}, fold {fold}: train/val group leakage.")
    if train_groups & test_groups:
        raise RuntimeError(f"Seed {seed}, fold {fold}: train/test group leakage.")
    if val_groups & test_groups:
        raise RuntimeError(f"Seed {seed}, fold {fold}: val/test group leakage.")


def make_val_split(trainval: pd.DataFrame, seed: int, fold: int, val_frac: float):
    if val_frac <= 0:
        return trainval.copy(), trainval.iloc[0:0].copy()

    n_classes = trainval["stratify_label"].nunique()
    n_val = max(n_classes, round(len(trainval) * val_frac))

    if len(trainval) - n_val < n_classes:
        raise RuntimeError(
            f"Cannot create stratified validation split for seed {seed}, fold {fold}. "
            f"Lower val_frac."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_val,
        random_state=seed * 1000 + fold,
    )

    train_idx, val_idx = next(
        splitter.split(trainval["group_id"], trainval["stratify_label"])
    )

    return trainval.iloc[train_idx].copy(), trainval.iloc[val_idx].copy()


def main():
    args = parse_args()

    spec = TASKS[args.task]

    with open(args.config) as f:
        cfg = json.load(f)

    dataset_csv = Path(cfg.get("dataset_csv", spec["dataset_csv"]))
    k = int(cfg["k"])
    seeds = [int(s) for s in cfg["seeds"]]
    val_frac = float(cfg.get("val_frac", 0.15))

    df = pd.read_csv(dataset_csv)

    required_cols = {spec["group_col"], spec["slide_col"], spec["stratify_col"]}
    missing = required_cols - set(df.columns)

    if missing:
        raise RuntimeError(f"Dataset CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df[spec["group_col"]] = df[spec["group_col"]].astype(str)
    df[spec["slide_col"]] = df[spec["slide_col"]].map(normalize_slide_id)
    df["_strat_label"] = df[spec["stratify_col"]].map(spec["normalize_label"])

    if spec.get("slide_number_fn") is not None:
        df["_slide_number"] = df[spec["slide_col"]].map(spec["slide_number_fn"])
    else:
        df["_slide_number"] = None

    group_df = build_group_table(df, spec)

    eligible = group_df[group_df["eligible_for_eval"]].copy()
    ineligible = group_df[~group_df["eligible_for_eval"]].copy()

    if spec.get("ineligible_group_policy") == "drop":
        train_only = group_df.iloc[0:0].copy()
    elif spec.get("ineligible_group_policy") == "train_only":
        train_only = ineligible.copy()
    else:
        raise RuntimeError(f"Unknown ineligible_group_policy: {spec.get('ineligible_group_policy')}")

    if eligible.empty:
        raise RuntimeError("No eligible groups found.")

    class_counts = eligible["stratify_label"].value_counts().reindex(spec["label_order"], fill_value=0)

    if class_counts.min() < k:
        raise RuntimeError(
            f"k={k} is too large for stratification counts:\n{class_counts.to_string()}"
        )

    print()
    print(f"Task: {args.task}")
    print(f"Dataset: {dataset_csv}")
    print()
    print("Eligible groups by stratification label:")
    print(class_counts.to_string())
    print()
    print(f"Train-only ineligible groups: {len(train_only)}")
    print()

    all_summary = []

    for seed in seeds:
        split_dir = Path("splits") / f"{spec['task_name']}_seed{seed}_k{k}"
        split_dir.mkdir(parents=True, exist_ok=True)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        X = eligible["group_id"].tolist()
        y = eligible["stratify_label"].tolist()

        assignments = []
        test_counts = {}

        for fold, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
            trainval_groups = eligible.iloc[trainval_idx].copy()
            test_groups = eligible.iloc[test_idx].copy()

            train_groups, val_groups = make_val_split(
                trainval_groups,
                seed=seed,
                fold=fold,
                val_frac=val_frac,
            )

            train_group_ids = train_groups["group_id"].tolist() + train_only["group_id"].tolist()
            val_group_ids = val_groups["group_id"].tolist()
            test_group_ids = test_groups["group_id"].tolist()

            check_disjoint(train_group_ids, val_group_ids, test_group_ids, seed, fold)

            for group_id in test_group_ids:
                test_counts[group_id] = test_counts.get(group_id, 0) + 1

            train_slides = slides_for_groups(df, train_group_ids, spec, val_or_test=False)
            val_slides = slides_for_groups(df, val_group_ids, spec, val_or_test=True)
            test_slides = slides_for_groups(df, test_group_ids, spec, val_or_test=True)

            save_split_csv(split_dir / f"splits_{fold}.csv", train_slides, val_slides, test_slides)
            save_bool_csv(split_dir / f"splits_{fold}_bool.csv", train_slides, val_slides, test_slides)

            row = {
                "seed": seed,
                "fold": fold,
                "train_groups": len(set(train_group_ids)),
                "val_groups": len(set(val_group_ids)),
                "test_groups": len(set(test_group_ids)),
                "train_slides": len(train_slides),
                "val_slides": len(val_slides),
                "test_slides": len(test_slides),
            }

            for label in spec["label_order"]:
                row[f"test_{label}_groups"] = int((test_groups["stratify_label"] == label).sum())

            pd.DataFrame([row]).to_csv(split_dir / f"splits_{fold}_descriptor.csv", index=False)
            all_summary.append(row)

            for role, group_ids in [
                ("train", train_group_ids),
                ("val", val_group_ids),
                ("test", test_group_ids),
            ]:
                for group_id in group_ids:
                    g = group_df[group_df["group_id"] == group_id].iloc[0]
                    assignments.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "group_id": group_id,
                            "role": role,
                            "stratify_label": g["stratify_label"],
                            "eligible_for_eval": g["eligible_for_eval"],
                            "n_slides": g["n_slides"],
                        }
                    )

        bad = {group_id: n for group_id, n in test_counts.items() if n != 1}

        if bad:
            raise RuntimeError(
                f"Seed {seed}: some eligible groups were not tested exactly once: {bad}"
            )

        pd.DataFrame(assignments).to_csv(split_dir / "group_assignments.csv", index=False)

        seed_summary = pd.DataFrame([r for r in all_summary if r["seed"] == seed])
        seed_summary.to_csv(split_dir / "split_summary.csv", index=False)

        split_config = {
            "task": args.task,
            "task_name": spec["task_name"],
            "dataset_csv": str(dataset_csv),
            "group_col": spec["group_col"],
            "slide_col": spec["slide_col"],
            "stratify_col": spec["stratify_col"],
            "k": k,
            "seed": seed,
            "val_frac": val_frac,
            "eligible_groups": int(len(eligible)),
            "train_only_ineligible_groups": int(len(train_only)),
            "label_order": spec["label_order"],
            "required_eval_slide_numbers": sorted(spec["required_eval_slide_numbers"])
            if spec.get("required_eval_slide_numbers") is not None
            else None,
            "val_test_slide_numbers": sorted(spec["val_test_slide_numbers"])
            if spec.get("val_test_slide_numbers") is not None
            else None,
        }

        with open(split_dir / "split_config.json", "w") as f:
            json.dump(split_config, f, indent=2)

        print(f"Wrote {split_dir}")

    summary_path = Path("splits") / f"{spec['task_name']}_multiseed_summary.csv"
    pd.DataFrame(all_summary).to_csv(summary_path, index=False)

    print()
    print(f"Wrote multiseed summary: {summary_path}")


if __name__ == "__main__":
    main()
