from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "src.config",
    "src.domain",
    "src.dataset",
    "src.model",
    "src.physics",
    "src.metrics",
    "src.active_sampler",
]

def main() -> None:
    print("[INFO] Testing GeoPINONet imports...\n")

    for module_name in MODULES:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:
            print(f"[FAIL] {module_name}")
            print(f"       {type(exc).__name__}: {exc}")
            raise

    print("\n[DONE] All imports passed.")

if __name__ == "__main__":
    main()	