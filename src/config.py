# ============================================================
# config.py
# GeoPINONet — Global configuration, hyperparameters and paths
# ============================================================

import os
from pathlib import Path

import torch

# ============================================================
# DEVICE & PRECISION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
amp_dtype = torch.float16  # switch to torch.bfloat16 if training is unstable

torch.set_default_dtype(torch.float32)

if device.type == "cuda":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Default full-dataset path.
# This can be overridden by src/domain.py when using example_lug mode.
BASE_PATH = str(ROOT / "data_generation" / "Lug_3D" / "model_train")

# Output folder for local training runs.
OUTPUT_DIR = ROOT / "results" / "training_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = str(OUTPUT_DIR / "geo_pino_model.pth")
LOG_CSV_PATH = str(OUTPUT_DIR / "training_log.csv")
SCAN_LOG_PATH = str(OUTPUT_DIR / "scan_log.csv")

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

E_physical  = 2e11   # Young's modulus [Pa] — structural steel
NU_physical = 0.3    # Poisson's ratio
NU_norm = NU_physical

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

ADAM_EPOCHS          = 1500
LEARNING_RATE        = 2e-4
BATCH_SIZE           = 1000
LATENT_DIM           = 1024
DATA_SCALAR          = 250
SIGMA_SCALAR         = 100
CHECKPOINT_FREQUENCY = 50

# ============================================================
# EPSILON-INSENSITIVE LOSS THRESHOLDS
# ============================================================

EPS_U_COMP   = 0.004
EPS_U_LAT    = 0.004
EPS_UZ_COMP  = 0.004
EPS_UZ_LAT   = 0.004
EPS_SIG_COMP = 0.004
EPS_SIG_LAT  = 0.004

# ============================================================
# POSITIONAL ENCODING
# ============================================================

N_FOURIER_FEATURES = 256
FOURIER_SCALES     = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

# ============================================================
# CURRICULUM SCHEDULE
# ============================================================

CURRICULUM_PHASE1_EPOCHS = 150   # data-only phase
CURRICULUM_PHASE2_EPOCHS = 600   # linear physics ramp

# ============================================================
# ACTIVE CRITICAL SELECTIVE VALIDATION (ActiveCSV)
# ============================================================

ACTIVE_BUDGET          = 8000
ACTIVE_N_CRITICAL      = 4000
ACTIVE_MIN_RADIUS      = 0.01
ACTIVE_CHECK_FREQUENCY = 100
DEADZONE_THRESHOLD_VM  = 0.0

# ============================================================
# CRITICAL-FOCUSED SUPERVISION
# ============================================================

SUP_BATCH        = 4096   # supervised batch size per geometry
CRITICAL_FRAC    = 0.75   # fraction of critical nodes (vs dynamic)
VM_LOSS_WEIGHT   = 35.0   # extra weight on Von Mises loss
VM_WEIGHT_BOOST  = 3.0    # additional weight on stress hotspots
VM_EPS           = 0.002  # deadzone threshold on normalized Von Mises

# ============================================================
# MATPLOTLIB — PUBLICATION QUALITY
# ============================================================

import matplotlib
matplotlib.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 11,
    'axes.labelsize' : 12,
    'axes.titlesize' : 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
    'lines.linewidth': 1.5,
})