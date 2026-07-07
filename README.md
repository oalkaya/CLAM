# CLAM on VALAR

This repository contains the CLAM pipeline and VALAR-specific scripts for whole-slide image preprocessing and model training.

## Create patches for a single slide

For run_clam_create_patches slides are expected under:

```text
/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides/
```

Run the wrapper from the repository root and pass the slide filename as the only argument:

```bash
./job_scripts/run_clam_create_patches.sh "#1-1 7817B8509.tiff"
```

Because slide filenames may contain spaces and characters such as `#`, always quote the filename.

The wrapper:

* verifies that the slide exists
* creates a run folder based on the slide filename without the extension
* copies the slide temporarily to scratch
* submits the CLAM patching job through Slurm
* deletes the temporary scratch copy when the job finishes

For the example above, outputs are stored under:

```text
runs/#1-1 7817B8509/
├── logs/
│   ├── job.out
│   └── job.err
└── results/
    ├── masks/
    ├── patches/
    ├── stitches/
    └── process_list_autogen.csv
```

## Monitor the job

```bash
squeue -u hpc-oalkaya
```

Follow the standard output log:

```bash
tail -f "runs/#1-1 7817B8509/logs/job.out"
```

Follow the error log:

```bash
tail -f "runs/#1-1 7817B8509/logs/job.err"
```

