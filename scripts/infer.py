import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from scipy import io
import warnings

warnings.filterwarnings("ignore")

import torch

from models.baseline_model import FNO2D
from src.utils.config import load_config
from src.utils.utilities3 import *

# -----------------------
# Args
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("--input_loc", type=str, default=None)
parser.add_argument("--min_max_file", type=str, default=None)

args = parser.parse_args()

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/infer.yaml")

if args.input_loc is not None:
    cfg.paths.input_loc = args.input_loc

if args.min_max_file is not None:
    cfg.paths.min_max_file = args.min_max_file

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"INPUT_LOC: {cfg.paths.input_loc}")
print(f"MIN_MAX_FILE: {cfg.paths.min_max_file}")

# -----------------------
# Load min-max
# -----------------------

min_max = io.loadmat(cfg.paths.min_max_file)

max_pm = float(min_max["cpm25_max"])
min_pm = float(min_max["cpm25_min"])

def denorm(x):
    return x * (max_pm - min_pm) + min_pm

# -----------------------
# Settings
# -----------------------

ntest  = cfg.data.ntest
time_input = cfg.data.time_input
time_out = cfg.data.time_out
T  = time_input + time_out
S1 = cfg.data.S1
S2 = cfg.data.S2
lat, long = S1, S2

met_variables = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
all_features = met_variables + emission_variables

savepath = cfg.paths.input_loc
V = cfg.features.V

# -----------------------
# Dataset
# -----------------------

class DataLoaders(torch.utils.data.Dataset):

    def __init__(self, cfg, min_max):

        self.time_input = cfg.data.time_input
        self.time_out   = cfg.data.time_out
        self.T = self.time_input + self.time_out

        self.S1 = cfg.data.S1
        self.S2 = cfg.data.S2

        self.met_variables = cfg.features.met_variables
        self.emi_variables = cfg.features.emission_variables
        self.all_features  = self.met_variables + self.emi_variables

        self.min_max = min_max

        self.arrs = {}
        for feat in self.all_features:
            path = os.path.join(cfg.paths.input_loc, f"{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        self.N = self.arrs[self.all_features[0]].shape[0]

    def __len__(self):
        return self.N

    def _normalize(self, x, key):

        maxx = float(self.min_max[f"{key}_max"])
        minn = float(self.min_max[f"{key}_min"])
        den = maxx - minn

        if key in ["u10", "v10"]:
            x = (2 * (x - minn) / den) - 1
        else:
            x = (x - minn) / den

        if key in self.emi_variables:
            x = np.clip(x, 0, 1)

        return x.astype(np.float32)

    def __getitem__(self, idx):

        x = np.empty((self.time_input, S1, S2, V), dtype=np.float32)

        for c, feat in enumerate(self.all_features):
            arr = self.arrs[feat][idx, :self.time_input]
            arr = self._normalize(arr, feat)
            x[..., c] = arr

        return torch.from_numpy(x)


test_dataset = DataLoaders(cfg, min_max)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# -----------------------
# Model
# -----------------------

checkpoint = torch.load(cfg.paths.checkpoint, map_location=device)

model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# -----------------------
# Inference
# -----------------------

os.makedirs(os.path.dirname(cfg.paths.output_loc), exist_ok=True)

prediction = np.zeros((len(test_dataset), lat, long, time_out), dtype=np.float32)

with torch.no_grad():
    for i, x in enumerate(tqdm(test_loader)):
        x = x.to(device, non_blocking=True)
        out = model(x).view(lat, long, time_out)
        prediction[i] = out.cpu().numpy()

prediction = denorm(prediction)

np.save(os.path.join(cfg.paths.output_loc, 'preds.npy'), prediction)

print("Saved predictions to:", cfg.paths.output_loc)