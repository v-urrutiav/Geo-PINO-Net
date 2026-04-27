import numpy as np
from scipy.stats import qmc
from scipy.spatial import ConvexHull, Delaunay, distance_matrix
from dataclasses import dataclass
from typing import Dict, Tuple, List


# ============================================================
# PROBLEM CONFIGURATION
# ============================================================

@dataclass
class Domain:
    # Lug 3D with fixed D = 20 mm
    # Dimensionless parameters:
    # x = [W/D, e/D, t/D]
    lower: np.ndarray
    upper: np.ndarray
    names: Tuple[str, ...] = ("W_over_D", "e_over_D", "t_over_D")
    D_mm: float = 20.0


DEFAULT_DOMAIN = Domain(
    lower=np.array([2.0, 1.5, 0.4], dtype=float),
    upper=np.array([3.0, 2.5, 0.8], dtype=float),
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def scale_unit_to_domain(X_unit: np.ndarray, domain: Domain) -> np.ndarray:
    return domain.lower + X_unit * (domain.upper - domain.lower)


def descale_domain_to_unit(X: np.ndarray, domain: Domain) -> np.ndarray:
    return (X - domain.lower) / (domain.upper - domain.lower)


def pairwise_min_distance(X: np.ndarray) -> float:
    if len(X) < 2:
        return np.inf
    D = distance_matrix(X, X)
    np.fill_diagonal(D, np.inf)
    return np.min(D)


def min_distance_to_set(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    For each row in A, returns the minimum distance to any point in B.
    """
    D = distance_matrix(A, B)
    return np.min(D, axis=1)


def farthest_point_sampling(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    first_index: int = None
) -> np.ndarray:
    """
    Greedy maximin selection of k points from X.
    """
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
    """
    Builds nested subsets using Farthest Point Sampling (FPS).
    Strategy:
    - first select a well-spread set of size max_k
    - then extract nested smaller subsets from it
    """
    subset_sizes = sorted(subset_sizes)
    max_k = subset_sizes[-1]

    idx_max = farthest_point_sampling(X, max_k, seed=seed)
    nested = {max_k: idx_max}

    current_idx = idx_max.copy()
    current_X = X[current_idx]

    for k in reversed(subset_sizes[:-1]):
        # Select a maximin subset from the current subset
        local_idx = farthest_point_sampling(current_X, k, seed=seed)
        current_idx = current_idx[local_idx]
        current_X = X[current_idx]
        nested[k] = current_idx.copy()

    return {k: nested[k] for k in subset_sizes}


def in_convex_hull(points: np.ndarray, hull_points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Tests whether each point in 'points' lies inside the convex hull of hull_points.
    """
    if len(hull_points) < hull_points.shape[1] + 1:
        raise ValueError("Not enough points to define a convex hull.")
    hull = Delaunay(hull_points)
    simplex = hull.find_simplex(points, tol=tol)
    return simplex >= 0


def save_csv(filename: str, X: np.ndarray, domain: Domain, include_mm: bool = True) -> None:
    header = list(domain.names)
    cols = [X[:, i] for i in range(X.shape[1])]

    if include_mm:
        W_mm = X[:, 0] * domain.D_mm
        e_mm = X[:, 1] * domain.D_mm
        t_mm = X[:, 2] * domain.D_mm
        header += ["W_mm", "e_mm", "t_mm"]
        cols += [W_mm, e_mm, t_mm]

    data = np.column_stack(cols)
    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.8f",
    )


# ============================================================
# TRAINING SET GENERATION
# ============================================================

def generate_lhs_train_pool(
    n_train: int,
    domain: Domain,
    seed: int = 42,
    optimization: str = "random-cd"
) -> np.ndarray:
    """
    Generates the base training pool using Latin Hypercube Sampling (LHS)
    over the physical domain.
    """
    sampler = qmc.LatinHypercube(d=len(domain.names), seed=seed, optimization=optimization)
    X_unit = sampler.random(n_train)
    X = scale_unit_to_domain(X_unit, domain)
    return X


# ============================================================
# INSIDE VALIDATION — WITHIN HULL, FAR FROM TRAINING SET
# ============================================================

def generate_inside_validation(
    train_X: np.ndarray,
    n_inside: int,
    domain: Domain,
    seed: int = 123,
    n_candidates: int = 5000,
    min_dist_to_train_unit: float = 0.18,
    min_dist_between_val_unit: float = 0.12,
) -> np.ndarray:
    """
    Generates validation points inside the convex hull of the training set,
    enforcing:
    - minimum distance to training points
    - minimum distance between validation points
    All distances computed in normalized [0,1]^3 space.
    """
    train_unit = descale_domain_to_unit(train_X, domain)

    sampler = qmc.LatinHypercube(d=len(domain.names), seed=seed + 1, optimization="random-cd")
    C_unit = sampler.random(n_candidates)

    inside_mask = in_convex_hull(C_unit, train_unit)
    C_unit = C_unit[inside_mask]

    if len(C_unit) == 0:
        raise RuntimeError("No candidates remained inside the convex hull.")

    # Filter by minimum distance to training set
    d_to_train = min_distance_to_set(C_unit, train_unit)
    C_unit = C_unit[d_to_train >= min_dist_to_train_unit]

    if len(C_unit) < n_inside:
        raise RuntimeError(
            f"Not enough inside-hull candidates satisfying "
            f"min_dist_to_train_unit={min_dist_to_train_unit}. "
            f"Try lowering that value or increasing n_candidates."
        )

    # Greedy maximin selection among candidates
    selected = []
    order = np.argsort(-min_distance_to_set(C_unit, train_unit))  # farthest from train first

    for idx in order:
        p = C_unit[idx]
        if len(selected) == 0:
            selected.append(p)
            if len(selected) == n_inside:
                break
            continue

        sel_arr = np.array(selected)
        d_to_sel = np.min(np.linalg.norm(sel_arr - p, axis=1))
        if d_to_sel >= min_dist_between_val_unit:
            selected.append(p)
            if len(selected) == n_inside:
                break

    if len(selected) < n_inside:
        raise RuntimeError(
            f"Could not build inside-hull validation set with n_inside={n_inside}, "
            f"min_dist_between_val_unit={min_dist_between_val_unit}. "
            f"Try lowering that value or increasing n_candidates."
        )

    selected = np.array(selected)
    return scale_unit_to_domain(selected, domain)


# ============================================================
# OUTSIDE VALIDATION — OUTSIDE HULL, BUT NEARBY
# ============================================================

def generate_outside_validation(
    train_X: np.ndarray,
    n_outside: int,
    domain: Domain,
    seed: int = 456,
    n_candidates: int = 10000,
    outward_margin_unit: float = 0.10,
    min_dist_to_train_unit: float = 0.08,
    max_dist_to_train_unit: float = 0.22,
    min_dist_between_val_unit: float = 0.10,
) -> np.ndarray:
    """
    Generates validation points slightly outside the convex hull, but not too far.
    Strategy:
    - sample candidates in an expanded box [-m, 1+m]^3
    - keep only those outside the training hull
    - enforce distance to training set within [min, max]
    - select maximin subset from remaining candidates
    """
    train_unit = descale_domain_to_unit(train_X, domain)

    lower_ext = np.full(train_unit.shape[1], -outward_margin_unit)
    upper_ext = np.full(train_unit.shape[1], 1.0 + outward_margin_unit)

    sampler = qmc.LatinHypercube(d=len(domain.names), seed=seed, optimization="random-cd")
    C_ext = sampler.random(n_candidates)
    C_unit = lower_ext + C_ext * (upper_ext - lower_ext)

    outside_mask = ~in_convex_hull(C_unit, train_unit)
    C_unit = C_unit[outside_mask]

    d_to_train = min_distance_to_set(C_unit, train_unit)
    mask = (d_to_train >= min_dist_to_train_unit) & (d_to_train <= max_dist_to_train_unit)
    C_unit = C_unit[mask]

    if len(C_unit) < n_outside:
        raise RuntimeError(
            f"Not enough outside-hull candidates with current thresholds. "
            f"Try increasing outward_margin_unit or n_candidates, "
            f"or relaxing min/max_dist_to_train_unit."
        )

    # Greedy maximin selection
    selected = []
    order = np.argsort(-min_distance_to_set(C_unit, train_unit))  # slightly farther from train first

    for idx in order:
        p = C_unit[idx]
        if len(selected) == 0:
            selected.append(p)
            if len(selected) == n_outside:
                break
            continue

        sel_arr = np.array(selected)
        d_to_sel = np.min(np.linalg.norm(sel_arr - p, axis=1))
        if d_to_sel >= min_dist_between_val_unit:
            selected.append(p)
            if len(selected) == n_outside:
                break

    if len(selected) < n_outside:
        raise RuntimeError(
            f"Could not build outside-hull validation set with n_outside={n_outside}. "
            f"Try relaxing min_dist_between_val_unit or increasing n_candidates."
        )

    selected = np.array(selected)
    return scale_unit_to_domain(selected, domain)


# ============================================================
# REPORTING
# ============================================================

def summarize_set(name: str, X: np.ndarray, train_X: np.ndarray, domain: Domain) -> None:
    X_unit = descale_domain_to_unit(X, domain)
    train_unit = descale_domain_to_unit(train_X, domain)

    dmin_train = min_distance_to_set(X_unit, train_unit)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"N = {len(X)}")
    print(f"Min pairwise distance within set (unit space): {pairwise_min_distance(X_unit):.4f}")
    print(f"Distance to training set (min / mean / max): "
          f"{dmin_train.min():.4f} / {dmin_train.mean():.4f} / {dmin_train.max():.4f}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    domain = DEFAULT_DOMAIN

    # --------------------------
    # 1) Base training pool
    # --------------------------
    n_train = 64
    train_X = generate_lhs_train_pool(
        n_train=n_train,
        domain=domain,
        seed=42,
        optimization="random-cd"
    )

    # --------------------------
    # 2) Nested subsets 4/8/16/32/64
    # --------------------------
    nested = nested_maximin_subsets(train_X, subset_sizes=[4, 8, 16, 32, 64], seed=42)

    train4  = train_X[nested[4]]
    train8  = train_X[nested[8]]
    train16 = train_X[nested[16]]
    train32 = train_X[nested[32]]
    train64 = train_X[nested[64]]

    # --------------------------
    # 3) Validation sets:
    #    - 12 inside hull, far from training points
    #    - 4 outside hull, close to training boundary
    # --------------------------
    val_inside = generate_inside_validation(
        train_X=train64,
        n_inside=12,
        domain=domain,
        seed=123,
        n_candidates=8000,
        min_dist_to_train_unit=0.18,
        min_dist_between_val_unit=0.12,
    )

    val_outside = generate_outside_validation(
        train_X=train64,
        n_outside=4,
        domain=domain,
        seed=456,
        n_candidates=15000,
        outward_margin_unit=0.10,
        min_dist_to_train_unit=0.08,
        max_dist_to_train_unit=0.22,
        min_dist_between_val_unit=0.10,
    )

    val16 = np.vstack([val_inside, val_outside])

    # --------------------------
    # 4) Save CSVs
    # --------------------------
    save_csv("lug_train_64.csv", train64, domain)
    save_csv("lug_train_32.csv", train32, domain)
    save_csv("lug_train_16.csv", train16, domain)
    save_csv("lug_train_8.csv",  train8,  domain)
    save_csv("lug_train_4.csv",  train4,  domain)

    save_csv("lug_val_12_inside.csv", val_inside,  domain)
    save_csv("lug_val_4_outside.csv", val_outside, domain)
    save_csv("lug_val_16_total.csv",  val16,        domain)

    # --------------------------
    # 5) Console report
    # --------------------------
    summarize_set("TRAIN 64",       train64,     train64, domain)
    summarize_set("TRAIN 32",       train32,     train64, domain)
    summarize_set("TRAIN 16",       train16,     train64, domain)
    summarize_set("TRAIN 8",        train8,      train64, domain)
    summarize_set("TRAIN 4",        train4,      train64, domain)
    summarize_set("VAL 12 INSIDE",  val_inside,  train64, domain)
    summarize_set("VAL 4 OUTSIDE",  val_outside, train64, domain)
    summarize_set("VAL 16 TOTAL",   val16,       train64, domain)

    print("\nGenerated files:")
    print("  - lug_train_4.csv")
    print("  - lug_train_8.csv")
    print("  - lug_train_16.csv")
    print("  - lug_train_32.csv")
    print("  - lug_train_64.csv")
    print("  - lug_val_12_inside.csv")
    print("  - lug_val_4_outside.csv")
    print("  - lug_val_16_total.csv")


if __name__ == "__main__":
    main()