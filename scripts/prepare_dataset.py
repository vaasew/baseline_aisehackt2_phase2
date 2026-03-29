import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from scipy import io

from src.utils.config import load_config

# -----------------------
# Args
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("--raw_path", type=str, default=None)
parser.add_argument("--min_max_file", type=str, default=None)

args = parser.parse_args()

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/prepare_dataset.yaml")

RAW_PATH = args.raw_path if args.raw_path is not None else cfg.paths.raw_path
MIN_MAX_FILE = args.min_max_file if args.min_max_file is not None else cfg.paths.min_max_file

# -----------------------
# Load min–max stats
# -----------------------

min_max = io.loadmat(MIN_MAX_FILE)
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw

min_vals = {f: min_max[f"{f}_min"].item() for f in all_features}
max_vals = {f: min_max[f"{f}_max"].item() for f in all_features}

# -----------------------
# Helper Functions
# -----------------------

def train_val_split(samples, val_frac=0.2, seed=0):
    np.random.seed(seed)
    N = samples.shape[0]
    idx = np.random.permutation(N)
    n_val = int(val_frac * N)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return samples[train_idx], samples[val_idx]


def create_timeseries_samples(
    month,
    feature_list,
    val_frac,
    seed,
    horizon,
    stride,
):
    train_data = {}
    val_data = {}

    for feat in tqdm(feature_list):

        file_path = os.path.join(RAW_PATH, month, f"{feat}.npy")
        arr = np.load(file_path).astype(np.float32)

        minn = min_vals[feat]
        maxx = max_vals[feat]
        den  = maxx - minn

        arr = (arr - minn) / den

        if feat in ["u10", "v10"]:
            arr = 2.0 * arr - 1.0

        if feat in cfg.features.emission_variables_raw:
            arr = np.clip(arr, 0, 1)

        print("Original shape:", arr.shape)

        T = arr.shape[0]
        idx = range(0, T - horizon + 1, stride)

        samples = np.stack([arr[i:i+horizon] for i in idx], axis=0)

        print("Total samples created -", samples.shape[0])

        train_samples, val_samples = train_val_split(
            samples, val_frac=val_frac, seed=seed
        )

        train_data[feat] = train_samples
        val_data[feat]   = val_samples

        del arr, samples

    return train_data, val_data


# -----------------------
# Run
# -----------------------

os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

print(f"\n==============================")
print(f"Train Save path: {cfg.paths.train_savepath}")
print(f"Val Save path: {cfg.paths.val_savepath}") 
print(f"Horizon={cfg.data.horizon}, Stride={cfg.data.stride}")
print(f"RAW_PATH={RAW_PATH}")
print(f"MIN_MAX_FILE={MIN_MAX_FILE}")
print(f"==============================\n")

for feat in all_features:

    print("\n===================================")
    print("Processing feature:", feat)
    print("===================================\n")

    train_chunks = []
    val_chunks   = []

    for month in cfg.data.months:

        print(f"Month: {month}")
        print(f"==============================\n")

        train_m, val_m = create_timeseries_samples(
            month=month,
            feature_list=[feat],
            val_frac=cfg.data.val_frac,
            seed=cfg.data.seed,
            horizon=cfg.data.horizon,
            stride=cfg.data.stride,
        )

        train_chunks.append(train_m[feat])
        val_chunks.append(val_m[feat])

        del train_m, val_m

    train_merged = np.concatenate(train_chunks, axis=0)
    val_merged   = np.concatenate(val_chunks, axis=0)

    print(
        f"\nFinal {feat} -> Train:", train_merged.shape,
        "Val:", val_merged.shape
    )

    np.save(
        os.path.join(cfg.paths.train_savepath, f"train_{feat}.npy"),
        train_merged.astype(np.float32)
    )
    np.save(
        os.path.join(cfg.paths.val_savepath, f"val_{feat}.npy"),
        val_merged.astype(np.float32)
    )

    del train_chunks, val_chunks, train_merged, val_merged