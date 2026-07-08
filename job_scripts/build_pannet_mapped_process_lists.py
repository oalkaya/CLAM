#!/usr/bin/env python3
"""Build mapped CLAM process lists from a case-to-preset CSV."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

SLIDE_RE = re.compile(r"^#(?P<case>\d+)-(?P<slide>\d+)\s")
SAFE_PRESET_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--case-map", required=True, type=Path)
    parser.add_argument("--presets-dir", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    return parser.parse_args()


def read_case_map(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Mapping CSV is empty: {path}")

        required = {"case_id", "preset"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Mapping CSV is missing columns: {sorted(missing)}")

        mapping: dict[int, str] = {}
        for row_number, row in enumerate(reader, start=2):
            case_text = (row["case_id"] or "").strip()
            preset = (row["preset"] or "").strip()
            if not case_text or not preset:
                raise ValueError(f"Empty case_id or preset on row {row_number}")

            case_id = int(case_text)
            if case_id in mapping:
                raise ValueError(f"Duplicate case_id: {case_id}")
            if not SAFE_PRESET_RE.fullmatch(preset):
                raise ValueError(f"Unsafe preset name for case {case_id}: {preset!r}")
            mapping[case_id] = preset

    if not mapping:
        raise ValueError("Mapping CSV contains no assignments")
    return mapping


def main() -> None:
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(args.source_dir)
    if not args.case_map.is_file():
        raise FileNotFoundError(args.case_map)
    if not args.presets_dir.is_dir():
        raise FileNotFoundError(args.presets_dir)

    case_map = read_case_map(args.case_map)
    slides: list[tuple[int, int, str]] = []

    for path in args.source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = SLIDE_RE.match(path.name)
        if match is None:
            raise ValueError(f"Could not parse case/slide prefix: {path.name}")

        slides.append((int(match.group("case")), int(match.group("slide")), path.name))

    if not slides:
        raise ValueError(f"No TIFF slides found in {args.source_dir}")

    slides.sort(key=lambda item: (item[0], item[1], item[2]))
    source_cases = {case_id for case_id, _, _ in slides}
    mapped_cases = set(case_map)

    missing_cases = sorted(source_cases - mapped_cases)
    extra_cases = sorted(mapped_cases - source_cases)
    if missing_cases:
        raise ValueError(f"No preset assignment for source cases: {missing_cases}")
    if extra_cases:
        raise ValueError(f"Mapped cases absent from source directory: {extra_cases}")

    presets = sorted(set(case_map.values()))
    args.config_dir.mkdir(parents=True, exist_ok=True)
    process_dir = args.config_dir / "process_lists"
    snapshot_dir = args.config_dir / "presets"
    process_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.case_map, args.config_dir / "case_preset_map.csv")

    process_paths: dict[str, Path] = {}
    preset_paths: dict[str, Path] = {}
    handles = {}
    writers = {}

    try:
        for preset in presets:
            source_preset = args.presets_dir / f"clam_pannet_{preset}.csv"
            if not source_preset.is_file() or source_preset.stat().st_size == 0:
                raise FileNotFoundError(f"Preset is missing or empty: {source_preset}")

            preset_snapshot = snapshot_dir / f"{preset}.csv"
            shutil.copy2(source_preset, preset_snapshot)
            preset_paths[preset] = preset_snapshot.resolve()

            process_path = process_dir / f"{preset}.csv"
            process_paths[preset] = process_path.resolve()
            handle = process_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["slide_id", "process"])
            handles[preset] = handle
            writers[preset] = writer

        counts: Counter[str] = Counter()
        slide_map = args.config_dir / "slide_preset_map.csv"
        with slide_map.open("w", newline="", encoding="utf-8") as handle:
            slide_writer = csv.writer(handle, lineterminator="\n")
            slide_writer.writerow(["slide_id", "case_id", "preset"])

            for case_id, _, slide_name in slides:
                preset = case_map[case_id]
                slide_writer.writerow([slide_name, case_id, preset])
                writers[preset].writerow([slide_name, 1])
                counts[preset] += 1
    finally:
        for handle in handles.values():
            handle.close()

    groups_path = args.config_dir / "mapped_groups.tsv"
    with groups_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["preset", "preset_path", "process_list_path"])
        for preset in presets:
            writer.writerow([preset, preset_paths[preset], process_paths[preset]])

    print(f"Slides: {len(slides)}")
    print(f"Cases:  {len(source_cases)}")
    for preset in presets:
        print(f"{preset}: {counts[preset]} slides")
    print(f"Slide map: {slide_map}")
    print(f"Groups:    {groups_path}")


if __name__ == "__main__":
    main()
