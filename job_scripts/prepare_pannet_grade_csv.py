#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


REPO_DIR = Path("/home/hpc-oalkaya/repos/CLAM")

INPUT_CSV = (
    REPO_DIR
    / "dataset_csv"
    / "pannet_wsi_labels.csv"
)

OUTPUT_CSV = (
    REPO_DIR
    / "dataset_csv"
    / "pannet_wsi_grade.csv"
)


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input metadata CSV does not exist:\n{INPUT_CSV}"
        )

    metadata = pd.read_csv(INPUT_CSV)

    required_columns = {
        "filename",
        "case",
        "grade",
        "ips",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    filenames = metadata["filename"].astype(str).str.strip()

    valid_extensions = filenames.str.lower().str.endswith(
        (".tif", ".tiff")
    )

    if not valid_extensions.all():
        invalid_names = filenames[~valid_extensions].tolist()

        raise ValueError(
            "The following filenames do not end in .tif or .tiff:\n"
            + "\n".join(invalid_names)
        )

    slide_ids = filenames.str.replace(
        r"(?i)\.(?:tif|tiff)$",
        "",
        regex=True,
    )

    if slide_ids.duplicated().any():
        duplicates = slide_ids[
            slide_ids.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            "Duplicate slide IDs found:\n"
            + "\n".join(duplicates)
        )

    case_ids = pd.to_numeric(
        metadata["case"],
        errors="raise",
    ).astype(int)

    grades = pd.to_numeric(
        metadata["grade"],
        errors="raise",
    ).astype(int)

    valid_grades = {1, 2, 3, 4, 5}
    observed_grades = set(grades.unique())

    if not observed_grades.issubset(valid_grades):
        raise ValueError(
            "Invalid grade values: "
            + str(sorted(observed_grades - valid_grades))
        )
    
    ips = (
        metadata["ips"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_ips = {"A", "B", "C"}
    observed_ips = set(ips.unique())

    if not observed_ips.issubset(valid_ips):
        raise ValueError(
            "Invalid IPS values: "
            + str(sorted(observed_ips - valid_ips))
        )

    clean = pd.DataFrame(
        {
            "case_id": case_ids,
            "slide_id": slide_ids,
            "grade": grades,
            "ips": ips,
        }
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Slides: {len(clean)}")
    print(f"Cases: {clean['case_id'].nunique()}")
    print()
    print("Grade distribution:")
    print(
        clean["grade"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()