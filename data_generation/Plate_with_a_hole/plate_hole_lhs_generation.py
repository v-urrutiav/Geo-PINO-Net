import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.spatial import distance_matrix
from dataclasses import dataclass
from typing import Dict, List


# ============================================================
# BENCHMARK CONFIGURATION: PLATE WITH CENTRAL HOLE
# ============================================================

@dataclass
class PlateHoleDomain:
    T: float = 20.0
    EDGE_CLEARANCE: float = 8.0

    # [W, H, D]
    lower: np.ndarray = None
    upper: np.ndarray = None
    names: tuple = ("W", "H", "D")

    def __post_init__(self):
        if self.lower is None:
            self.lower = np.array([
                100.0,  # W
                140.0,  # H
                20.0,   # D
            ], dtype=float)

        if self.upper is None:
            self.upper = np.array([
                180.0,  # W
                240.0,  # H
                70.0,   # D
            ], dtype=float)


DOMAIN = PlateHoleDomain()


# ============================================================
# UTILITIES
# ============================================================

def scale_unit_to_domain(X_unit: np.ndarray, domain: PlateHoleDomain) -> np.ndarray:
    return domain.lower + X_unit * (domain.upper - domain.lower)


def descale_domain_to_unit(X: np.ndarray, domain: PlateHoleDomain) -> np.ndarray:
    return (X - domain.lower) / (domain.upper - domain.lower)


def pairwise_min_distance(X: np.ndarray) -> float:
    if len(X) < 2:
        return np.inf
    D = distance_matrix(X, X)
    np.fill_diagonal(D, np.inf)
    return np.min(D)


def min_distance_to_set(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    D = distance_matrix(A, B)
    return np.min(D, axis=1)


def farthest_point_sampling(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    first_index: int = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    if k > n:
        raise ValueError("k cannot be greater than len(X).")

    if first_index is None:
        first_index = rng.integers(0, n)

    selected = [first_index]
    remaining = set(range(n))
    remaining.remove(first_index)

    while len(selected) < k:
        rem_idx = np.array(sorted(list(remaining)))
        D = distance_matrix(X[rem_idx], X[selected])
        dmin = np.min(D, axis=1)
        next_idx = rem_idx[np.argmax(dmin)]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return np.array(selected, dtype=int)


def nested_maximin_subsets(
    X: np.ndarray,
    subset_sizes: List[int],
    seed: int = 42
) -> Dict[int, np.ndarray]:
    subset_sizes = sorted(subset_sizes)
    max_k = subset_sizes[-1]

    idx_max = farthest_point_sampling(X, max_k, seed=seed)
    nested = {max_k: idx_max}

    current_idx = idx_max.copy()
    current_X = X[current_idx]

    for k in reversed(subset_sizes[:-1]):
        local_idx = farthest_point_sampling(current_X, k, seed=seed)
        current_idx = current_idx[local_idx]
        current_X = X[current_idx]
        nested[k] = current_idx.copy()

    return {k: nested[k] for k in subset_sizes}


# ============================================================
# GEOMETRIC VALIDATION
# ============================================================

def validate_geometry(row: np.ndarray, domain: PlateHoleDomain) -> bool:
    W, H, D = row
    R = D / 2.0

    # Minimum ligament to each edge
    if (W / 2.0 - R) < domain.EDGE_CLEARANCE:
        return False
    if (H / 2.0 - R) < domain.EDGE_CLEARANCE:
        return False

    return True


def enrich_dataframe(X: np.ndarray, domain: PlateHoleDomain) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=domain.names)

    df["T"] = domain.T
    df["Xc"] = df["W"] / 2.0
    df["Yc"] = df["H"] / 2.0
    df["R"] = df["D"] / 2.0

    df["ligament_x"] = df["W"] / 2.0 - df["R"]
    df["ligament_y"] = df["H"] / 2.0 - df["R"]
    df["clear_ok"] = (df["ligament_x"] >= domain.EDGE_CLEARANCE) & (df["ligament_y"] >= domain.EDGE_CLEARANCE)

    return df


def save_csv(filename: str, X: np.ndarray, domain: PlateHoleDomain) -> None:
    df = enrich_dataframe(X, domain)
    df.to_csv(filename, index=False)


# ============================================================
# TRAINING SET — LHS GENERATION
# ============================================================

def generate_lhs_candidates(
    n_candidates: int,
    domain: PlateHoleDomain,
    seed: int = 42,
    optimization: str = "random-cd"
) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(domain.names), seed=seed, optimization=optimization)
    X_unit = sampler.random(n_candidates)
    return scale_unit_to_domain(X_unit, domain)


def generate_valid_pool(
    n_target: int,
    domain: PlateHoleDomain,
    seed: int = 42,
    oversample_factor: int = 12
) -> np.ndarray:
    print(f"[1/4] Generating candidate pool for training set (n={n_target})...")
    candidates = generate_lhs_candidates(
        n_candidates=n_target * oversample_factor,
        domain=domain,
        seed=seed
    )

    print("[1/4] Filtering geometrically valid samples...")
    valid_mask = np.array([validate_geometry(row, domain) for row in candidates], dtype=bool)
    valid = candidates[valid_mask]

    if len(valid) < n_target:
        raise RuntimeError(
            f"Could not generate {n_target} valid geometries. "
            f"Only {len(valid)} were obtained."
        )

    print(f"[1/4] Valid pool size: {len(valid)}")
    return valid[:n_target]


# ============================================================
# VALIDATION SET
# ============================================================

def generate_inside_validation(
    train_X: np.ndarray,
    n_inside: int,
    domain: PlateHoleDomain,
    seed: int = 123,
    n_candidates: int = 8000,
    min_dist_to_train_unit: float = 0.16,
    min_dist_between_val_unit: float = 0.10,
) -> np.ndarray:
    print("[3/4] Generating inside validation set...")

    train_unit = descale_domain_to_unit(train_X, domain)

    sampler = qmc.LatinHypercube(d=len(domain.names), seed=seed, optimization="random-cd")
    C_unit = sampler.random(n_candidates)
    C = scale_unit_to_domain(C_unit, domain)

    valid_mask = np.array([validate_geometry(row, domain) for row in C], dtype=bool)
    C = C[valid_mask]
    C_unit = C_unit[valid_mask]
    print(f"[3/4] Geometrically valid inside candidates: {len(C)}")

    train_min = train_unit.min(axis=0)
    train_max = train_unit.max(axis=0)

    inside_mask = np.all((C_unit >= train_min) & (C_unit <= train_max), axis=1)
    C = C[inside_mask]
    C_unit = C_unit[inside_mask]
    print(f"[3/4] Candidates within training bounding box: {len(C)}")

    d_to_train = min_distance_to_set(C_unit, train_unit)
    keep = d_to_train >= min_dist_to_train_unit
    C = C[keep]
    C_unit = C_unit[keep]
    print(f"[3/4] Inside candidates after distance filter: {len(C)}")

    if len(C) < n_inside:
        raise RuntimeError("Not enough inside candidates. Lower thresholds or increase n_candidates.")

    selected = []
    selected_unit = []

    order = np.argsort(-min_distance_to_set(C_unit, train_unit))
    for idx in order:
        p = C[idx]
        pu = C_unit[idx]

        if len(selected) == 0:
            selected.append(p)
            selected_unit.append(pu)
            if len(selected) == n_inside:
                break
            continue

        selu = np.array(selected_unit)
        d_to_sel = np.min(np.linalg.norm(selu - pu, axis=1))
        if d_to_sel >= min_dist_between_val_unit:
            selected.append(p)
            selected_unit.append(pu)
            if len(selected) == n_inside:
                break

    if len(selected) < n_inside:
        raise RuntimeError("Could not build inside validation set with current thresholds.")

    return np.array(selected)


def generate_outside_validation(
    train_X: np.ndarray,
    n_outside: int,
    domain: PlateHoleDomain,
    seed: int = 456,
    n_candidates: int = 12000,
    outward_margin_unit: float = 0.08,
    min_dist_to_train_unit: float = 0.08,
    max_dist_to_train_unit: float = 0.22,
    min_dist_between_val_unit: float = 0.08,
) -> np.ndarray:
    print("[4/4] Generating outside validation set...")

    train_unit = descale_domain_to_unit(train_X, domain)

    train_min = train_unit.min(axis=0)
    train_max = train_unit.max(axis=0)

    lower_ext = np.maximum(0.0, train_min - outward_margin_unit)
    upper_ext = np.minimum(1.0, train_max + outward_margin_unit)

    sampler = qmc.LatinHypercube(d=len(domain.names), seed=seed, optimization="random-cd")
    C_local = sampler.random(n_candidates)
    C_unit = lower_ext + C_local * (upper_ext - lower_ext)
    C = scale_unit_to_domain(C_unit, domain)

    valid_mask = np.array([validate_geometry(row, domain) for row in C], dtype=bool)
    C = C[valid_mask]
    C_unit = C_unit[valid_mask]
    print(f"[4/4] Geometrically valid outside candidates: {len(C)}")

    outside_mask = np.any((C_unit < train_min) | (C_unit > train_max), axis=1)
    C = C[outside_mask]
    C_unit = C_unit[outside_mask]
    print(f"[4/4] Candidates outside training bounding box: {len(C)}")

    d_to_train = min_distance_to_set(C_unit, train_unit)
    keep = (d_to_train >= min_dist_to_train_unit) & (d_to_train <= max_dist_to_train_unit)
    C = C[keep]
    C_unit = C_unit[keep]
    print(f"[4/4] Outside candidates after distance filter: {len(C)}")

    if len(C) < n_outside:
        raise RuntimeError("Not enough outside candidates. Adjust thresholds or increase n_candidates.")

    selected = []
    selected_unit = []

    order = np.argsort(-min_distance_to_set(C_unit, train_unit))
    for idx in order:
        p = C[idx]
        pu = C_unit[idx]

        if len(selected) == 0:
            selected.append(p)
            selected_unit.append(pu)
            if len(selected) == n_outside:
                break
            continue

        selu = np.array(selected_unit)
        d_to_sel = np.min(np.linalg.norm(selu - pu, axis=1))
        if d_to_sel >= min_dist_between_val_unit:
            selected.append(p)
            selected_unit.append(pu)
            if len(selected) == n_outside:
                break

    if len(selected) < n_outside:
        raise RuntimeError("Could not build outside validation set with current thresholds.")

    return np.array(selected)


# ============================================================
# REPORTING
# ============================================================

def summarize_set(name: str, X: np.ndarray, train_X: np.ndarray, domain: PlateHoleDomain) -> None:
    X_unit = descale_domain_to_unit(X, domain)
    train_unit = descale_domain_to_unit(train_X, domain)

    dmin_train = min_distance_to_set(X_unit, train_unit)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"N = {len(X)}")
    print(f"Min pairwise distance within set (unit space): {pairwise_min_distance(X_unit):.4f}")
    print(
        f"Distance to training set (min / mean / max): "
        f"{dmin_train.min():.4f} / {dmin_train.mean():.4f} / {dmin_train.max():.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    domain = DOMAIN

    # 1) Master training pool
    train64 = generate_valid_pool(
        n_target=64,
        domain=domain,
        seed=42,
        oversample_factor=15
    )

    # 2) Nested subsets
    print("[2/4] Building nested maximin subsets...")
    nested = nested_maximin_subsets(train64, subset_sizes=[4, 8, 16, 32, 64], seed=42)

    train4  = train64[nested[4]]
    train8  = train64[nested[8]]
    train16 = train64[nested[16]]
    train32 = train64[nested[32]]

    # 3) Validation sets
    val_inside = generate_inside_validation(
        train_X=train64,
        n_inside=12,
        domain=domain,
        seed=123,
        n_candidates=8000,
        min_dist_to_train_unit=0.16,
        min_dist_between_val_unit=0.10,
    )

    val_outside = generate_outside_validation(
        train_X=train64,
        n_outside=4,
        domain=domain,
        seed=456,
        n_candidates=12000,
        outward_margin_unit=0.08,
        min_dist_to_train_unit=0.08,
        max_dist_to_train_unit=0.22,
        min_dist_between_val_unit=0.08,
    )

    val16 = np.vstack([val_inside, val_outside])

    # 4) Save CSVs
    save_csv("plate_hole_train_4.csv",      train4,      domain)
    save_csv("plate_hole_train_8.csv",      train8,      domain)
    save_csv("plate_hole_train_16.csv",     train16,     domain)
    save_csv("plate_hole_train_32.csv",     train32,     domain)
    save_csv("plate_hole_train_64.csv",     train64,     domain)

    save_csv("plate_hole_val_12_inside.csv",  val_inside,  domain)
    save_csv("plate_hole_val_4_outside.csv",  val_outside, domain)
    save_csv("plate_hole_val_16_total.csv",   val16,       domain)

    # 5) Summary report
    summarize_set("TRAIN 64",       train64,    train64, domain)
    summarize_set("TRAIN 32",       train32,    train64, domain)
    summarize_set("TRAIN 16",       train16,    train64, domain)
    summarize_set("TRAIN 8",        train8,     train64, domain)
    summarize_set("TRAIN 4",        train4,     train64, domain)
    summarize_set("VAL 12 INSIDE",  val_inside, train64, domain)
    summarize_set("VAL 4 OUTSIDE",  val_outside,train64, domain)
    summarize_set("VAL 16 TOTAL",   val16,      train64, domain)

    print("\nGenerated files:")
    print("  - plate_hole_train_4.csv")
    print("  - plate_hole_train_8.csv")
    print("  - plate_hole_train_16.csv")
    print("  - plate_hole_train_32.csv")
    print("  - plate_hole_train_64.csv")
    print("  - plate_hole_val_12_inside.csv")
    print("  - plate_hole_val_4_outside.csv")
    print("  - plate_hole_val_16_total.csv")


if __name__ == "__main__":
    main()