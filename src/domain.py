# ============================================================
# domain.py
# GeoPINONet — Geometry definitions and active training set
#
# Edit this file to switch between:
#   - example sanity training
#   - full Lug_3D training
#   - full Plate_with_a_hole training
# ============================================================

import os
from pathlib import Path

from .config import BASE_PATH as CONFIG_BASE_PATH


# ============================================================
# MODE SELECTION
# ============================================================
# Options:
#   "example_lug"  -> use examples/Lug_3D/case_001
#   "full_lug"     -> use CONFIG_BASE_PATH from config.py
#
# For now, use the example geometry.
# ============================================================

DOMAIN_MODE = "example_lug"


# ============================================================
# NESTED MAXIMIN SUBSETS
# ============================================================

SUBSET_4  = [1, 30, 4, 13]
SUBSET_8  = [1, 30, 4, 13, 16, 57, 5, 25]
SUBSET_16 = [15, 1, 4, 3, 5, 6, 32, 16, 25, 11, 30, 13, 50, 26, 57, 14]
SUBSET_32 = [6, 24, 15, 20, 31, 3, 23, 32, 4, 42, 30, 50, 13, 1, 56, 57,
             7, 14, 60, 18, 34, 22, 16, 10, 25, 5, 53, 26, 2, 11, 64, 17]
SUBSET_64 = list(range(1, 65))


# ============================================================
# PATH AND ACTIVE SET
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if DOMAIN_MODE == "example_lug":
    # Flat example folder:
    # examples/Lug_3D/case_001/
    BASE_PATH = str(ROOT / "examples" / "Lug_3D" / "case_001")
    ACTIVE_SET = [1]

elif DOMAIN_MODE == "full_lug":
    # Use whatever full dataset path is defined in config.py
    BASE_PATH = CONFIG_BASE_PATH
    ACTIVE_SET = SUBSET_64

else:
    raise ValueError(f"Unknown DOMAIN_MODE: {DOMAIN_MODE}")


# ============================================================
# TRAINING GEOMETRIES
# File naming convention:
#   {i}_body.stl
#   {i}_load.stl
#   {i}_fixed.stl
#   {i}_compression.csv
#   {i}_lateral_bending.csv
# ============================================================

GEOMETRIES = [
    {
        "vol":   f"{i}_body.stl",
        "load":  f"{i}_load.stl",
        "fixed": f"{i}_fixed.stl",
        "comp":  f"{i}_compression.csv",
        "lat":   f"{i}_lateral_bending.csv",
    }
    for i in ACTIVE_SET
]


# ============================================================
# VALIDATION GEOMETRIES
# For the example sanity training, leave empty.
# ============================================================

GEOMETRIES_VAL = []


# ============================================================
# PATH RESOLVER
# ============================================================

def get_path(filename: str) -> str:
    """Resolve a geometry filename to its full absolute path."""
    return os.path.join(BASE_PATH, filename)