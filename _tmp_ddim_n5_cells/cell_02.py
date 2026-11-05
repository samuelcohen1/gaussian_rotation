from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from scipy.stats import pearsonr

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUM_SAMPLES = 5
SEED = 42
REFERENCE_SAMPLE_COUNT = 100
OUTPUT_DIR = Path("results/ddim_affine_sign_registration_n5")
LATENT_DIR = OUTPUT_DIR / "latents"
IMAGE_DIR = OUTPUT_DIR / "images"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_ID = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 50
DEVICE = "cuda"
DTYPE = "float16"
IMAGE_SIZE = 512
PROMPT = "a photograph of an astronaut riding a horse"
GUIDANCE_SCALE = 7.5
VAE_SCALE_FACTOR = 8

# Pixel-space affine conditions. Small set keeps GPU + registration under ~10 min.
# params = (angle_deg, scale_x, scale_y, shear, tx_px, ty_px)
CONDITIONS = [
    {"name": "baseline", "params": (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)},
    {"name": "rotation_10", "params": (10.0, 1.0, 1.0, 0.0, 0.0, 0.0)},
    {"name": "hshift_10", "params": (0.0, 1.0, 1.0, 0.0, 10.0, 0.0)},
    {"name": "vshift_10", "params": (0.0, 1.0, 1.0, 0.0, 0.0, 10.0)},
    {"name": "combined_mild", "params": (8.0, 1.05, 0.98, 0.05, 6.0, -4.0)},
]

PARAM_NAMES = ["angle_deg", "scale_x", "scale_y", "shear", "tx", "ty"]
ANGLE_RANGE = (-20.0, 20.0)
SCALE_RANGE = (0.85, 1.15)
SHEAR_RANGE = (-0.2, 0.2)
TRANSLATION_RANGE = (-4.0, 4.0)
BOUNDS = [ANGLE_RANGE, SCALE_RANGE, SCALE_RANGE, SHEAR_RANGE, TRANSLATION_RANGE, TRANSLATION_RANGE]
LB = np.array([b[0] for b in BOUNDS], dtype=float)
UB = np.array([b[1] for b in BOUNDS], dtype=float)
SCIPY_BOUNDS = list(zip(LB, UB))

WARP_ORDER = 1
VALID_RADIUS_FRAC = 0.32
COARSE_ANGLE_STEP = 4.0
COARSE_TRANSLATION_STEP = 1.0

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 11})
for path in (OUTPUT_DIR, LATENT_DIR, IMAGE_DIR, FIGURES_DIR):
    path.mkdir(parents=True, exist_ok=True)

print(f"samples={NUM_SAMPLES}, conditions={len(CONDITIONS)}")
print(f"inversions if nothing cached: {NUM_SAMPLES * len(CONDITIONS)}")
print(f"output -> {OUTPUT_DIR.resolve()}")
