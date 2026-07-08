# CLAM on VALAR

VALAR-specific scripts for CLAM whole-slide image preprocessing, feature extraction, and training preparation on the PanNET dataset.

## Paths

Repository:

```text
/home/hpc-oalkaya/repos/CLAM
```

WSIs:

```text
/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides
```

Conda environment:

```text
clam_latest_valar
```

Generated outputs are stored under:

```text
runs/
├── patching/
├── feature_extraction/
├── training/
├── evaluation/
└── heatmaps/
```

## Environment

```bash
module purge
module load conda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam_latest_valar

cd /home/hpc-oalkaya/repos/CLAM
```

The Slurm scripts activate the required environment themselves. Submitting from a clean shell is recommended to avoid inheriting conflicting Conda or module variables.

## Patching preset

The active PanNET preset is:

```text
presets/clam_pannet_preset.csv
```

Each patching run stores a snapshot of the preset under its own `config/` directory.

## Patch one slide

Always quote filenames because they may contain spaces and `#`.

```bash
./job_scripts/run_clam_create_patches.sh \
    "#1-1 7817B8509.tiff"
```

Output:

```text
runs/patching/#1-1 7817B8509/
├── config/
├── logs/
│   ├── job.out
│   └── job.err
└── results/
    ├── masks/
    ├── patches/
    ├── stitches/
    └── process_list_autogen.csv
```

The slide is temporarily copied to scratch and removed when the job exits.

## Bulk patching

Patch all slides:

```bash
./job_scripts/run_clam_create_patches_bulk.sh all
```

Patch the first 25 slides:

```bash
./job_scripts/run_clam_create_patches_bulk.sh 25
```

Output example:

```text
runs/patching/pannet_all/
├── config/
│   ├── patching_preset.csv
│   └── process_list_input.csv
├── logs/
└── results/
    ├── masks/
    ├── patches/
    ├── stitches/
    └── process_list_autogen.csv
```

The `.h5` files under `results/patches/` contain patch coordinates, not patch images.

### Resume a failed bulk run

By default, the wrapper refuses to overwrite an existing run. Pass `--resume` to reuse it:

```bash
./job_scripts/run_clam_create_patches_bulk.sh all --resume
```

CLAM skips slides that already have a patch `.h5` file and continues with the remaining slides. Resume logs are stored separately from the original logs.

## Feature extraction

Extract features from a completed patching run:

```bash
./job_scripts/run_clam_extract_features.sh pannet_all
```

The current encoder is:

```text
resnet50_trunc
```

with 1024-dimensional patch embeddings.

Output:

```text
runs/feature_extraction/pannet_all_resnet50_trunc/
├── config/
├── logs/
│   ├── job.out
│   └── job.err
└── results/
    ├── h5_files/
    └── pt_files/
```

Files:

- `pt_files/<slide_id>.pt`: patch embeddings used for CLAM training
- `h5_files/<slide_id>.h5`: patch embeddings together with coordinates

CLAM training should receive this directory as its data root:

```text
runs/feature_extraction/pannet_all_resnet50_trunc/results
```

It then loads:

```text
<data_root_dir>/pt_files/<slide_id>.pt
```

## PanNET grade task

The custom task name is:

```text
task_pannet_grade
```

The dataset CSV is expected at:

```text
dataset_csv/pannet_grade.csv
```

Required columns:

```csv
case_id,slide_id,grade,ips
```

- `case_id`: groups slides from the same case to prevent split leakage
- `slide_id`: must match the corresponding `.pt` filename
- `grade`: slide-level infiltration grade from 1 to 5
- `ips`: case-level infiltration category used when stratifying splits

The `main.py` task trains a five-class CLAM model using `grade` as the target.

## Monitoring jobs

List jobs:

```bash
squeue -u hpc-oalkaya
```

Follow a log:

```bash
tail -f runs/patching/pannet_all/logs/job.out
```

Follow errors:

```bash
tail -f runs/patching/pannet_all/logs/job.err
```

Count completed patch files:

```bash
find runs/patching/pannet_all/results/patches \
    -maxdepth 1 -type f -name '*.h5' | wc -l
```

Count completed feature bags:

```bash
find runs/feature_extraction/pannet_all_resnet50_trunc/results/pt_files \
    -maxdepth 1 -type f -name '*.pt' | wc -l
```

## Git

Generated runs should remain untracked:

```gitignore
runs/
```