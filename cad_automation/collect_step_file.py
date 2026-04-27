import os
import shutil

# ============================================================
# CONFIG
# ============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Folder where Autodesk Inventor exports the temporary STEP file.
# Default: C:/Users/<user>/Documents/Lug3D
source_folder = str(Path.home() / "Documents" / "Lug3D")

# Repository target folder.
target_folder = str(ROOT / "data_generation" / "Lug_3D" / "model")

os.makedirs(target_folder, exist_ok=True)

state_file = os.path.join(target_folder, "state_stp.txt")

original_name = "solid.stp"

# ============================================================
# ENSURE BASE FOLDER EXISTS
# ============================================================

os.makedirs(target_folder, exist_ok=True)

if not os.path.exists(state_file):
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("1")

# ============================================================
# READ STATE
# ============================================================

with open(state_file, "r", encoding="utf-8") as f:
    i = int(f.read().strip())

print(f"Processing iteration: {i}")

# ============================================================
# PATHS
# ============================================================

src_original = os.path.join(source_folder, original_name)
renamed_name = f"{i}_solid.stp"
src_renamed   = os.path.join(source_folder, renamed_name)

case_folder = os.path.join(target_folder, str(i))
os.makedirs(case_folder, exist_ok=True)

dst_copy = os.path.join(case_folder, renamed_name)

# ============================================================
# VALIDATIONS
# ============================================================


# ============================================================
# 1) RENAME AT SOURCE
# ============================================================

os.rename(src_original, src_renamed)
print(f"OK: Renamed at source: {original_name} -> {renamed_name}")

# ============================================================
# 2) COPY TO CASE FOLDER
# ============================================================

shutil.copy2(src_renamed, dst_copy)
print(f"OK: Copied to case folder: {dst_copy}")

# ============================================================
# 3) UPDATE STATE
# ============================================================

with open(state_file, "w", encoding="utf-8") as f:
    f.write(str(i + 1))

print("State updated.")