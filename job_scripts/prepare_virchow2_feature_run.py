from pathlib import Path
import os

import h5py
import torch


SOURCE_DIR = Path(
    "/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/"
    "pannet_wsi_features/40x_1024px_0px_overlap/features_virchow2"
)

REPO_DIR = Path("/home/hpc-oalkaya/repos/CLAM")

FEATURE_RUN_ID = "pannet_virchow2_40x1024"
OUT_DIR = REPO_DIR / "runs" / "feature_extraction" / FEATURE_RUN_ID / "results"

PT_DIR = OUT_DIR / "pt_files"
H5_DIR = OUT_DIR / "h5_files"

PT_DIR.mkdir(parents=True, exist_ok=True)
H5_DIR.mkdir(parents=True, exist_ok=True)

h5_files = sorted(SOURCE_DIR.glob("*.h5"))

if not h5_files:
    raise SystemExit(f"No .h5 files found in {SOURCE_DIR}")

converted = 0
bad = []
embedding_dims = set()

for h5_path in h5_files:
    slide_id = h5_path.stem

    try:
        with h5py.File(h5_path, "r") as f:
            if "features" not in f:
                raise KeyError("Missing dataset: features")

            features = torch.from_numpy(f["features"][:]).float()

            if features.ndim != 2:
                raise ValueError(f"Expected 2D features, got {tuple(features.shape)}")

            if features.shape[0] == 0:
                raise ValueError("Zero patches")

            embedding_dims.add(features.shape[1])

        torch.save(features, PT_DIR / f"{slide_id}.pt")

        link_path = H5_DIR / h5_path.name
        if not link_path.exists():
            os.symlink(h5_path, link_path)

        converted += 1

    except Exception as e:
        bad.append((h5_path.name, type(e).__name__, str(e)))

print(f"Source H5 files: {len(h5_files)}")
print(f"Converted PT files: {converted}")
print(f"Embedding dimensions: {sorted(embedding_dims)}")
print(f"Feature run ID: pannet_virchow2_40x1024")
print(f"Output directory: {OUT_DIR}")

if bad:
    print("\nBad files:")
    for name, error_type, message in bad:
        print(f"{name}: {error_type}: {message}")
    raise SystemExit("Some files failed conversion.")