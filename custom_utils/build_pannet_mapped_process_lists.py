#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path


SLIDE_RE = re.compile(r"^#(?P<case>\d+)-(?P<slide>\d+)\s+.*\.(tif|tiff)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-preset CLAM process lists for PanNET mapped patching. "
            "Input mapping assigns case IDs or case ranges to preset CSV files."
        )
    )

    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--presets-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)

    return parser.parse_args()


def normalize_preset_name(name: str) -> str:
    name = str(name).strip()

    if not name:
        raise ValueError("Empty preset name in mapping file.")

    if not name.endswith(".csv"):
        name = f"{name}.csv"

    return name


def read_case_map(path: Path) -> dict[int, str]:
    """
    Supported formats:

    1) case_id,preset
       1,pannet_loose.csv
       2,pannet_loose.csv

    2) case_start,case_end,preset
       1,10,pannet_loose.csv
       11,15,pannet_loose.csv

    3) start,end,preset
       1,10,pannet_loose.csv

    4) case_range,preset
       1-10,pannet_loose.csv
       16,pannet_strict.csv
    """

    case_to_preset: dict[int, str] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Mapping file has no header: {path}")

        fields = {name.strip() for name in reader.fieldnames}

        for row_idx, raw_row in enumerate(reader, start=2):
            row = {k.strip(): (v.strip() if v is not None else "") for k, v in raw_row.items()}

            if "preset" not in row or not row["preset"]:
                raise ValueError(f"Missing preset at row {row_idx} in {path}")

            preset = normalize_preset_name(row["preset"])

            if "case_id" in fields:
                case_ids = [int(row["case_id"])]

            elif {"case_start", "case_end"}.issubset(fields):
                start = int(row["case_start"])
                end = int(row["case_end"])
                case_ids = list(range(start, end + 1))

            elif {"start", "end"}.issubset(fields):
                start = int(row["start"])
                end = int(row["end"])
                case_ids = list(range(start, end + 1))

            elif "case_range" in fields:
                case_range = row["case_range"].replace(" ", "")

                if "-" in case_range:
                    start_s, end_s = case_range.split("-", 1)
                    start = int(start_s)
                    end = int(end_s)
                    case_ids = list(range(start, end + 1))
                else:
                    case_ids = [int(case_range)]

            else:
                raise ValueError(
                    "Unsupported mapping format. Expected one of: "
                    "case_id,preset OR case_start,case_end,preset OR "
                    "start,end,preset OR case_range,preset."
                )

            for case_id in case_ids:
                if case_id in case_to_preset and case_to_preset[case_id] != preset:
                    raise ValueError(
                        f"Case {case_id} assigned to multiple presets: "
                        f"{case_to_preset[case_id]} and {preset}"
                    )

                case_to_preset[case_id] = preset

    if not case_to_preset:
        raise ValueError(f"No case mappings found in {path}")

    return case_to_preset


def find_slides(source_dir: Path) -> list[tuple[str, int, int]]:
    slides: list[tuple[str, int, int]] = []

    for path in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = SLIDE_RE.match(path.name)
        if not match:
            raise ValueError(
                "Unexpected PanNET slide filename. Expected: "
                "#<case>-<slide> <id>.tif/.tiff\n"
                f"Bad filename: {path.name}"
            )

        case_id = int(match.group("case"))
        slide_number = int(match.group("slide"))
        slides.append((path.name, case_id, slide_number))

    if not slides:
        raise ValueError(f"No .tif/.tiff slides found in {source_dir}")

    return slides


def write_process_list(path: Path, slide_names: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["slide_id", "process"])

        for slide_name in slide_names:
            writer.writerow([slide_name, 1])


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    case_map_path = args.case_map.resolve()
    presets_dir = args.presets_dir.resolve()
    config_dir = args.config_dir.resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"Missing source directory: {source_dir}")

    if not case_map_path.is_file():
        raise SystemExit(f"Missing case mapping file: {case_map_path}")

    if not presets_dir.is_dir():
        raise SystemExit(f"Missing presets directory: {presets_dir}")

    config_dir.mkdir(parents=True, exist_ok=True)

    case_to_preset = read_case_map(case_map_path)
    slides = find_slides(source_dir)

    preset_to_slides: dict[str, list[str]] = defaultdict(list)
    slide_rows: list[dict[str, str | int]] = []
    unmapped_cases: set[int] = set()

    for slide_name, case_id, slide_number in slides:
        preset = case_to_preset.get(case_id)

        if preset is None:
            unmapped_cases.add(case_id)
            continue

        preset_path = presets_dir / preset

        if not preset_path.is_file():
            raise SystemExit(
                f"Preset listed in mapping does not exist:\n"
                f"  {preset_path}\n"
                f"Referenced by case {case_id}"
            )

        preset_to_slides[preset].append(slide_name)

        slide_rows.append(
            {
                "slide_id": slide_name,
                "case_id": case_id,
                "slide_number": slide_number,
                "preset": preset,
                "preset_path": str(preset_path),
            }
        )

    if unmapped_cases:
        raise SystemExit(
            "Some cases in the source directory are not assigned in the mapping file:\n"
            + ", ".join(str(x) for x in sorted(unmapped_cases))
        )

    if not preset_to_slides:
        raise SystemExit("No slides were assigned to any preset.")

    process_lists_dir = config_dir / "mapped_process_lists"
    process_lists_dir.mkdir(parents=True, exist_ok=True)

    mapped_groups_path = config_dir / "mapped_groups.tsv"

    with mapped_groups_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["preset", "preset_path", "process_list_path"])

        for preset in sorted(preset_to_slides):
            preset_stem = Path(preset).stem
            process_list_path = process_lists_dir / f"process_list_{preset_stem}.csv"

            write_process_list(process_list_path, preset_to_slides[preset])

            writer.writerow(
                [
                    preset_stem,
                    str(presets_dir / preset),
                    str(process_list_path),
                ]
            )

    slide_map_path = config_dir / "slide_preset_map.csv"

    with slide_map_path.open("w", newline="") as f:
        fieldnames = ["slide_id", "case_id", "slide_number", "preset", "preset_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(slide_rows)

    shutil.copy2(case_map_path, config_dir / case_map_path.name)

    summary_path = config_dir / "mapped_process_lists_summary.txt"

    with summary_path.open("w") as f:
        f.write(f"source_dir={source_dir}\n")
        f.write(f"case_map={case_map_path}\n")
        f.write(f"presets_dir={presets_dir}\n")
        f.write(f"total_slides={len(slides)}\n")
        f.write(f"assigned_slides={len(slide_rows)}\n")
        f.write(f"mapped_cases={len(case_to_preset)}\n")
        f.write("\nslides_per_preset:\n")

        for preset in sorted(preset_to_slides):
            f.write(f"{preset}: {len(preset_to_slides[preset])}\n")

    print(f"Source slides: {len(slides)}")
    print(f"Assigned slides: {len(slide_rows)}")
    print(f"Mapped groups: {mapped_groups_path}")
    print(f"Slide preset map: {slide_map_path}")
    print(f"Summary: {summary_path}")

    for preset in sorted(preset_to_slides):
        print(f"{preset}: {len(preset_to_slides[preset])} slides")


if __name__ == "__main__":
    main()
