import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================

base_dir = r"C:\Users\..."
output_path = r"C:\Users\...\subsets_scaling_study.txt"

subset_sizes = [4, 8, 16, 32]
tol = 1e-6

# Define both geometries: name, reference CSV, subset CSV pattern, geometry columns
geometries = [
    {
        "name": "Plate with a Hole",
        "csv_64": os.path.join(base_dir, "plate_hole_train_64.csv"),
        "csv_pattern": os.path.join(base_dir, "plate_hole_train_{size}.csv"),
        "cols": ["W", "H", "D"],
    },
    {
        "name": "Lug",
        "csv_64": os.path.join(base_dir, "lug_train_64.csv"),
        "csv_pattern": os.path.join(base_dir, "lug_train_{size}.csv"),
        "cols": ["W_mm", "e_mm", "t_mm"],
    },
]

# ============================================================
# PROCESS BOTH GEOMETRIES
# ============================================================

all_results = {}

for geom in geometries:
    name = geom["name"]
    cols = geom["cols"]
    print(f"\n{'='*40}")
    print(f"Processing: {name}")
    print(f"{'='*40}")

    if not os.path.exists(geom["csv_64"]):
        print(f"  WARNING: Reference file not found: {geom['csv_64']}")
        all_results[name] = None
        continue

    df64 = pd.read_csv(geom["csv_64"])
    geom_results = {}

    for size in subset_sizes:
        csv_small = geom["csv_pattern"].format(size=size)

        if not os.path.exists(csv_small):
            print(f"  WARNING: File not found: {csv_small}")
            geom_results[size] = None
            continue

        df_small = pd.read_csv(csv_small)
        indices = []

        for i, row_small in df_small.iterrows():
            found = False
            for j, row_64 in df64.iterrows():
                if np.all(np.abs(row_small[cols].values - row_64[cols].values) < tol):
                    indices.append(j + 1)  # +1 because folders start at 1
                    found = True
                    break
            if not found:
                print(f"  WARNING [subset_{size}]: Row {i} not found in the 64-sample dataset.")

        geom_results[size] = sorted(indices)
        print(f"  Subset {size}: {geom_results[size]}")

    all_results[name] = geom_results

# ============================================================
# WRITE SUMMARY TXT
# ============================================================

with open(output_path, "w") as f:
    f.write("SCALING STUDY - GEOMETRY INDICES\n")
    f.write("=" * 40 + "\n")
    f.write("All subsets are subsets of the 64-sample dataset.\n")
    f.write("=" * 40 + "\n")

    for geom in geometries:
        name = geom["name"]
        f.write(f"\n{'='*40}\n")
        f.write(f"GEOMETRY: {name}\n")
        f.write(f"{'='*40}\n\n")

        geom_results = all_results.get(name)
        if geom_results is None:
            f.write("  Reference file not found.\n")
            continue

        for size in subset_sizes:
            indices = geom_results.get(size)
            if indices is None:
                f.write(f"SUBSET {size}:\n  File not found.\n\n")
            else:
                f.write(f"SUBSET_{size} = {indices}\n\n")

print(f"\nSummary saved to: {output_path}")
