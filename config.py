# =============================================================================
#  config.py  —  All paths and hyperparameters in one place.
# =============================================================================

import os

# ── PATHS ─────────────────────────────────────────────────────────────────────
BRATS_ROOT     = r"archive/BraTS2021_Training_Data"

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
PATCH_SIZE     = (64, 64, 64)
PATCH_STRIDE   = (64, 64, 64)

# ── DATA SPLIT ────────────────────────────────────────────────────────────────
VAL_FRACTION   = 0.20
TEST_FRACTION  = 0.10
RANDOM_SEED    = 42

# ── 3D U-NET ──────────────────────────────────────────────────────────────────
IN_CHANNELS    = 4
OUT_CHANNELS   = 1
BASE_FILTERS   = 16
ENCODER_DEPTHS = 3

# ── TRAINING ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 4
NUM_EPOCHS     = 150
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
LR_PATIENCE    = 15
EARLY_STOP     = 30

# ── SUPERVOXEL (SLIC) ─────────────────────────────────────────────────────────
# N_SUPERVOXELS    = 2000
N_SUPERVOXELS    = 1000
SLIC_COMPACTNESS = 0.1

# ── GRAPH CUT ─────────────────────────────────────────────────────────────────
# GC_LAMBDA      = 5.0
# GC_SIGMA       = 0.3
GC_LAMBDA = 100.0      # increase from 5.0 to 50.0
GC_SIGMA  = 0.05     # decrease from 0.3 to 0.1
# ── MISC ──────────────────────────────────────────────────────────────────────
NUM_WORKERS    = 0
PIN_MEMORY     = True
DEVICE         = "cuda"
MAX_SUBJECTS   = 200