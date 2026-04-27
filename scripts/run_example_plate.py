from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CASE_DIR = ROOT / "examples" / "Plate_with_a_hole" / "case_001"
MODEL_PATH = ROOT / "trained_models" / "Plate_with_a_hole" / "64_hole_final.pth"
OUT_DIR = ROOT / "results" / "example_runs" / "Plate_with_a_hole"

REQUIRED_COLUMNS = [
    "x", "y", "z",
    "ux", "uy", "uz",
    "sxx", "syy", "szz",
    "sxy", "syz", "sxz",
]


def check_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def load_solution(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    ux, uy, uz = df["ux"], df["uy"], df["uz"]
    sxx, syy, sxy = df["sxx"], df["syy"], df["sxy"]

    df["u_mag"] = np.sqrt(ux**2 + uy**2 + uz**2)
    df["vm_ip"] = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)

    return df


def summarize(df: pd.DataFrame, load_case: str) -> dict:
    return {
        "load_case": load_case,
        "n_nodes": len(df),
        "u_mag_max": df["u_mag"].max(),
        "u_mag_mean": df["u_mag"].mean(),
        "u_mag_p95": np.percentile(df["u_mag"], 95),
        "vm_ip_max": df["vm_ip"].max(),
        "vm_ip_mean": df["vm_ip"].mean(),
        "vm_ip_p95": np.percentile(df["vm_ip"], 95),
    }


def main() -> None:
    print("[INFO] Running Plate_with_a_hole example")

    files = [
        CASE_DIR / "1_body.stl",
        CASE_DIR / "1_load.stl",
        CASE_DIR / "1_fixed.stl",
        CASE_DIR / "1_compression.csv",
        CASE_DIR / "1_lateral_bending.csv",
    ]

    for path in files:
        check_exists(path)
        print(f"[OK] Found {path.relative_to(ROOT)}")

    comp = load_solution(CASE_DIR / "1_compression.csv")
    lat = load_solution(CASE_DIR / "1_lateral_bending.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([
        summarize(comp, "compression"),
        summarize(lat, "lateral_bending"),
    ])

    out_path = OUT_DIR / "example_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"[OK] Summary written to {out_path.relative_to(ROOT)}")

    if MODEL_PATH.exists():
        print(f"[OK] Trained model found: {MODEL_PATH.relative_to(ROOT)}")
    else:
        print(f"[WARN] Trained model not found: {MODEL_PATH.relative_to(ROOT)}")
        print("[WARN] Run trained_models/download_models.py after the zenodo archive is available.")

    print("[DONE] Plate_with_a_hole example check completed.")


if __name__ == "__main__":
    main()