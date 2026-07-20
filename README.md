# CLAM on VALAR

VALAR-specific scripts and configuration for running CLAM-style whole-slide preprocessing, feature extraction, training, and evaluation.

The current main dataset is PanNET, but the structure is intended to support additional datasets with minimal script changes.

## Repository

```text
/home/hpc-oalkaya/repos/CLAM
```

## Project structure

```text
CLAM/
├── configs/
│   └── pannet_k4.json
│
├── presets/
│   └── pannet/
│       ├── loose.csv
│       ├── medium.csv
│       ├── strict.csv
│       └── case_to_preset_map.csv
│
├── custom_utils/
│   └── build_pannet_mapped_process_lists.py
│
├── job_scripts/
│   ├── run_create_patches_fp.sh
│   └── create_patches_fp.sbatch
│
├── dataset_csv/
│   └── pannet_wsi_grade.csv
│
└── runs/
    ├── patching/
    ├── feature_extraction/
    ├── training/
    ├── evaluation/
    └── heatmaps/
```

Generated outputs are stored under `runs/` and should not be committed.

## Environment

The main Conda environment is:

```text
clam_latest_valar
```

The Slurm scripts activate the required environment internally. Submitting from a clean shell is recommended to avoid inheriting conflicting Conda or module variables.

## Dataset config

Dataset-level settings live in JSON config files under:

```text
configs/
```

Current PanNET config:

```text
configs/pannet_k4.json
```

Example:

```json
{
  "dataset_name": "pannet",
  "dataset_csv": "dataset_csv/pannet_wsi_grade.csv",
  "k": 4,
  "seeds": [1, 2, 3, 4, 5],
  "val_frac": 0.15,
  "patching": {
    "source_dir": "/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides",
    "patch_size": 1024,
    "step_size": 1024,
    "patch_level": 0,
    "default_preset": "medium.csv",
    "default_mapping": "case_to_preset_map.csv",
    "mapping_helper": "custom_utils/build_pannet_mapped_process_lists.py"
  }
}
```

Patching-related fields:

| Field | Description |
|---|---|
| `dataset_name` | Dataset namespace used for run names and preset lookup |
| `patching.source_dir` | Folder containing WSI files |
| `patching.patch_size` | Patch width/height passed to `create_patches_fp.py` |
| `patching.step_size` | Stride between extracted patches |
| `patching.patch_level` | WSI pyramid level used for patch extraction |
| `patching.default_preset` | Default preset from `presets/<dataset_name>/` |
| `patching.default_mapping` | Default case-to-preset map from `presets/<dataset_name>/` |
| `patching.mapping_helper` | Dataset-specific helper used for mapped patching |

## Presets

Patching presets are namespaced by dataset:

```text
presets/<dataset_name>/
```

For PanNET:

```text
presets/pannet/
├── loose.csv
├── medium.csv
├── strict.csv
└── case_to_preset_map.csv
```

Preset CSVs contain CLAM segmentation parameters such as `seg_level`, `sthresh`, `mthresh`, `a_t`, `a_h`, `max_n_holes`, and related options.

For non-mapped patching, one preset is used for all selected slides:

```bash
--preset medium.csv
```

For mapped patching, `case_to_preset_map.csv` assigns cases to presets:

```csv
case_id,preset
1,loose
2,loose
16,strict
19,medium
```

The preset names resolve to:

```text
presets/pannet/loose.csv
presets/pannet/medium.csv
presets/pannet/strict.csv
```

## Patching scripts

Main wrapper:

```text
job_scripts/run_create_patches_fp.sh
```

Slurm job script:

```text
job_scripts/create_patches_fp.sbatch
```

Underlying CLAM script:

```text
create_patches_fp.py
```

The wrapper supports these modes:

| Mode | Description |
|---|---|
| `bulk mapped` | Patch all slides using case-specific presets |
| `bulk all` | Patch all slides with one preset |
| `bulk N` | Patch the first `N` slides in sorted filename order |
| `bulk representatives` | Patch one representative slide per PanNET case |
| `single <slide_filename>` | Patch one specified slide |

Additional options:

| Option | Description |
|---|---|
| `--config` | Dataset config JSON |
| `--preset` | Preset file from `presets/<dataset_name>/` |
| `--mapping` | Mapping file from `presets/<dataset_name>/` |
| `--seg-only` | Run segmentation/mask generation only |
| `--resume` | Reuse an existing run folder |
| `--run-id` | Override the automatic run name |

## Patching output layout

Each patching run creates:

```text
runs/patching/<run_id>/
├── config/
├── logs/
└── results/
    ├── masks/
    ├── patches/
    └── stitches/
```

The `.h5` files under `results/patches/` contain patch coordinates, not image patches.

Each run stores a snapshot of the config and process lists under its own `config/` directory.

## Run naming

Standard automatic names are intentionally short:

```text
pannet_mapped_1024
pannet_medium_1024
pannet_first10_strict_1024
pannet_single_#1-1_7817B8509_medium_1024
```

If `step_size != patch_size` or `patch_level != 0`, those values are included:

```text
pannet_mapped_1024_s512_l0
pannet_medium_1024_s1024_l1
```

Passing `--run-id` overrides the automatic name.

Using `--seg-only` appends:

```text
_seg_only
```

## Mapped patching

Mapped patching allows different cases to use different segmentation presets.

For PanNET:

```text
presets/pannet/case_to_preset_map.csv
custom_utils/build_pannet_mapped_process_lists.py
```

The mapping helper converts:

```text
case_id -> preset
```

into per-preset process lists, because `create_patches_fp.py` can only use one preset per call.

Generated mapped-patching config files:

```text
runs/patching/<run_id>/config/
├── mapped_groups.tsv
├── slide_preset_map.csv
└── mapped_process_lists/
    ├── process_list_loose.csv
    ├── process_list_medium.csv
    └── process_list_strict.csv
```

The Slurm script loops over `mapped_groups.tsv` and runs `create_patches_fp.py` once per preset group.

Run full mapped PanNET patching:

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk mapped \
  --config configs/pannet_k4.json
```

Expected output:

```text
runs/patching/pannet_mapped_1024/
```

Segmentation-only mapped run:

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk mapped \
  --config configs/pannet_k4.json \
  --seg-only \
  --run-id test_mapped
```

## Single-preset bulk patching

Patch all slides with one preset:

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk all \
  --config configs/pannet_k4.json \
  --preset medium.csv
```

Expected output:

```text
runs/patching/pannet_medium_1024/
```

Segmentation-only version:

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk all \
  --config configs/pannet_k4.json \
  --preset medium.csv \
  --seg-only \
  --run-id test_all_medium
```

## First-N patching

Patch only the first `N` slides in sorted filename order:

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk 3 \
  --config configs/pannet_k4.json \
  --preset strict.csv \
  --seg-only \
  --run-id test_first3_strict
```

## Single-slide patching

Always quote slide filenames because they may contain spaces and `#`.

Segmentation-only:

```bash
./job_scripts/run_create_patches_fp.sh \
  single "#1-1 7817B8509.tiff" \
  --config configs/pannet_k4.json \
  --preset medium.csv \
  --seg-only \
  --run-id test_single_medium
```

Full patch/stitch for one slide:

```bash
./job_scripts/run_create_patches_fp.sh \
  single "#1-1 7817B8509.tiff" \
  --config configs/pannet_k4.json \
  --preset medium.csv \
  --run-id test_single_medium_full
```

## Representatives mode

This selects one representative slide per PanNET case.

`representatives` mode is currently PanNET-specific. It assumes slide filenames follow the pattern `#<case>-<slide> <id>.tif/.tiff`, and it selects the lowest slide number for each case. For datasets with a different naming convention, use `bulk all`, `bulk N`, `single`, or write a dataset-specific representative-selection helper before using this mode.

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk representatives \
  --config configs/pannet_k4.json \
  --preset loose.csv \
  --seg-only \
  --run-id test_representatives_loose
```

## Resume a patching run

The wrapper refuses to overwrite an existing run by default. Use `--resume` to reuse an existing run folder.

```bash
./job_scripts/run_create_patches_fp.sh \
  bulk mapped \
  --config configs/pannet_k4.json \
  --resume
```

CLAM skips slides that already have completed patch `.h5` outputs and continues with the remaining slides.

## Adapting patching to another dataset

For a new dataset, add:

```text
configs/<dataset>.json
presets/<dataset_name>/
```

For mapped presets, add a dataset-specific helper:

```text
custom_utils/build_<dataset_name>_mapped_process_lists.py
```

The helper must generate:

```text
config/mapped_groups.tsv
config/slide_preset_map.csv
config/mapped_process_lists/*.csv
```

The wrapper and Slurm script should not need to change unless the dataset uses different WSI extensions or requires a different representative-slide selection rule for visual inspection.