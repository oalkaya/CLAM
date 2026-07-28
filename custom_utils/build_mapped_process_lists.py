#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-preset CLAM process lists from dataset metadata."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--patient-id-column", required=True)
    parser.add_argument("--slide-id-column", required=True)
    parser.add_argument("--slide-extensions-json", required=True)
    parser.add_argument("--patient-map", type=Path, required=True)
    parser.add_argument("--presets-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_preset(value: object) -> str:
    preset = str(value).strip()
    if not preset:
        raise ValueError("Preset names must be non-empty.")
    return preset if preset.endswith(".csv") else f"{preset}.csv"


def expand_patient_ids(row: dict[str, str], fields: set[str]) -> list[str]:
    if "patient_id" in fields:
        return [str(row["patient_id"]).strip()]

    bounds = None
    if {"patient_start", "patient_end"} <= fields:
        bounds = (row["patient_start"], row["patient_end"])
    elif {"start", "end"} <= fields:
        bounds = (row["start"], row["end"])

    if bounds is not None:
        start, end = (int(value) for value in bounds)
        return [str(value) for value in range(start, end + 1)]

    if "patient_range" in fields:
        value = row["patient_range"].replace(" ", "")
        if "-" not in value:
            return [value]
        start, end = (int(part) for part in value.split("-", 1))
        return [str(item) for item in range(start, end + 1)]

    raise ValueError(
        "Mapping must use patient_id, patient_start/patient_end, start/end, "
        "or patient_range."
    )


def read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Mapping has no header: {path}")
        fields = {field.strip() for field in reader.fieldnames}

        for line, raw_row in enumerate(reader, start=2):
            row = {
                str(key).strip(): str(value or "").strip()
                for key, value in raw_row.items()
            }
            if "preset" not in row:
                raise ValueError("Mapping requires a preset column.")
            preset = normalize_preset(row["preset"])
            for patient_id in expand_patient_ids(row, fields):
                if patient_id in mapping and mapping[patient_id] != preset:
                    raise ValueError(
                        f"Line {line}: patient {patient_id} has multiple presets."
                    )
                mapping[patient_id] = preset
    if not mapping:
        raise ValueError(f"No patient mappings found in {path}.")
    return mapping


def locate_slide(
    source_dir: Path,
    slide_id: str,
    extensions: list[str],
) -> Path:
    slide_path = Path(slide_id)
    candidates = (
        [source_dir / slide_path.name]
        if slide_path.suffix
        else [source_dir / f"{slide_id}{extension}" for extension in extensions]
    )
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source WSI for slide {slide_id!r}; "
            f"found {matches}."
        )
    return matches[0]


def write_process_list(path: Path, slide_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["slide_id", "process"])
        writer.writerows((slide_name, 1) for slide_name in slide_names)


def main() -> None:
    args = parse_args()
    extensions = json.loads(args.slide_extensions_json)
    mapping = read_mapping(args.patient_map)
    metadata = pd.read_csv(
        args.metadata_csv,
        dtype={args.patient_id_column: str, args.slide_id_column: str},
    )
    missing = {
        args.patient_id_column,
        args.slide_id_column,
    } - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    args.config_dir.mkdir(parents=True, exist_ok=True)
    process_dir = args.config_dir / "mapped_process_lists"
    process_dir.mkdir(parents=True, exist_ok=True)

    preset_to_slides: dict[str, list[str]] = defaultdict(list)
    slide_rows: list[dict[str, str]] = []
    for _, row in metadata.iterrows():
        patient_id = str(row[args.patient_id_column]).strip()
        slide_id = str(row[args.slide_id_column]).strip()
        if patient_id not in mapping:
            raise ValueError(f"Patient {patient_id} has no preset assignment.")

        preset = mapping[patient_id]
        preset_path = args.presets_dir / preset
        if not preset_path.is_file():
            raise FileNotFoundError(f"Missing preset: {preset_path}")
        slide_path = locate_slide(args.source_dir, slide_id, extensions)
        preset_to_slides[preset].append(slide_path.name)
        slide_rows.append(
            {
                "slide_id": slide_id,
                "patient_id": patient_id,
                "source_filename": slide_path.name,
                "preset": preset,
            }
        )

    groups_path = args.config_dir / "mapped_groups.tsv"
    with groups_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["preset", "preset_path", "process_list_path"])
        for preset in sorted(preset_to_slides):
            process_path = process_dir / f"process_list_{Path(preset).stem}.csv"
            write_process_list(process_path, sorted(preset_to_slides[preset]))
            writer.writerow(
                [Path(preset).stem, args.presets_dir / preset, process_path]
            )

    pd.DataFrame(slide_rows).to_csv(
        args.config_dir / "slide_preset_map.csv", index=False
    )
    shutil.copy2(args.patient_map, args.config_dir / args.patient_map.name)
    print(f"Mapped slides: {len(slide_rows)}")
    print(f"Mapped preset groups: {len(preset_to_slides)}")


if __name__ == "__main__":
    main()
