import os
import shutil

# ============================================================
# CONFIG
# ============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Folder where Autodesk Inventor exports the temporary STL files.
# Default: C:/Users/<user>/Documents/Lug3D
source_folder = str(Path.home() / "Documents" / "Lug3D")

# Repository target folder.
target_folder = str(ROOT / "data_generation" / "Lug_3D" / "model")

os.makedirs(target_folder, exist_ok=True)

state_file = os.path.join(target_folder, "state_stl.txt")

# ============================================================
# ENSURE BASE FOLDER EXISTS
# ============================================================

os.makedirs(target_folder, exist_ok=True)

# ============================================================
# READ STATE
# ============================================================

if not os.path.exists(state_file):
    with open(state_file, "w", encoding="utf-8") as f:
        f.write("1")

with open(state_file, "r", encoding="utf-8") as f:
    i = int(f.read().strip())

print(f"Processing iteration: {i}")

# ============================================================
# CREATE CASE FOLDER
# ============================================================

case_folder = os.path.join(target_folder, str(i))
os.makedirs(case_folder, exist_ok=True)

# ============================================================
# EXPECTED FILES
# ============================================================

files = {
    "solid.stl": f"{i}_body.stl",
    "load.stl":  f"{i}_load.stl",
    "fixed.stl": f"{i}_fixed.stl",
}

# ============================================================
# MOVE AND RENAME FILES DIRECTLY INTO CASE FOLDER
# ============================================================

moved_any = False

for src_name, dst_name in files.items():
    src_path = os.path.join(source_folder, src_name)
    dst_path = os.path.join(case_folder, dst_name)

    if not os.path.exists(src_path):
        print(f"WARNING: File not found: {src_name}")
        continue

    shutil.move(src_path, dst_path)
    print(f"OK: {src_name} -> {dst_path}")
    moved_any = True

# ============================================================
# UPDATE STATE ONLY IF FILES WERE MOVED
# ============================================================

if moved_any:
    with open(state_file, "w", encoding="utf-8") as f:
        f.write(str(i + 1))
    print("State updated.")
else:
    print("No files were moved. State unchanged.")