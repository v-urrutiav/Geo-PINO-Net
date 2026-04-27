# ============================================================
# dataset.py
# GeoPINONet — Data loading, per-geometry scale computation
# ============================================================

import os
import numpy as np
import trimesh
from .domain import BASE_PATH, GEOMETRIES, GEOMETRIES_VAL
from .config import BASE_PATH, E_physical


# ============================================================
# PER-GEOMETRY SCALE COMPUTATION
# ============================================================

def compute_geometry_scales(geometries: list, base_path: str = BASE_PATH) -> None:
    """
    Compute and attach per-geometry normalization scales in-place.

    For each geometry dict, attaches:
      L_char        — characteristic length [m] (bounding box max extent)
      u_char_comp   — 99th percentile displacement norm, compression case
      u_char_lat    — 99th percentile displacement norm, lateral bending case
      sigma_char_*  — 99th percentile stress norm per case
      E_norm_comp   — normalized Young's modulus, compression case
      E_norm_lat    — normalized Young's modulus, lateral bending case
      z_mid         — normalized mid-thickness coordinate
      z_half        — normalized half-thickness
    """
    print("[INIT] Computing per-geometry scales...")

    for g in geometries:
        mesh      = trimesh.load(os.path.join(base_path, g['vol']))
        g['L_char'] = float(np.max(mesh.bounds[1] - mesh.bounds[0]) * 1e-3)

        c = np.loadtxt(os.path.join(base_path, g['comp']), delimiter=',', skiprows=1)
        l = np.loadtxt(os.path.join(base_path, g['lat']),  delimiter=',', skiprows=1)

        g['u_char_comp']     = float(np.percentile(np.linalg.norm(c[:, 3:6],  axis=1), 99))
        g['u_char_lat']      = float(np.percentile(np.linalg.norm(l[:, 3:6],  axis=1), 99))
        g['sigma_char_comp'] = float(np.percentile(np.linalg.norm(c[:, 6:12], axis=1), 99))
        g['sigma_char_lat']  = float(np.percentile(np.linalg.norm(l[:, 6:12], axis=1), 99))

        g['E_norm_comp'] = (E_physical * g['u_char_comp']) / (g['sigma_char_comp'] * g['L_char'])
        g['E_norm_lat']  = (E_physical * g['u_char_lat'])  / (g['sigma_char_lat']  * g['L_char'])

        g['z_mid']  = float((mesh.bounds[1][2] + mesh.bounds[0][2]) / 2 * 1e-3 / g['L_char'])
        g['z_half'] = float((mesh.bounds[1][2] - mesh.bounds[0][2]) / 2 * 1e-3 / g['L_char'])

        print(f"  {g['vol']}: L={g['L_char']:.4f}m | "
              f"u_comp={g['u_char_comp']:.2e} | E_norm_comp={g['E_norm_comp']:.3f}")


# ============================================================
# ANSYS FEM CSV LOADER
# ============================================================

def load_ansys_csv(path: str, g: dict, mode: str, name: str) -> dict:
    """
    Load and normalize an ANSYS FEM result CSV.

    Expected columns: [X, Y, Z, UX, UY, UZ, Sxx, Syy, Szz, Sxy, Syz, Sxz]

    Parameters
    ----------
    path : str
        Full path to the CSV file.
    g    : dict
        Geometry dict with pre-computed scales (output of compute_geometry_scales).
    mode : str
        Load case identifier: 'comp' or 'lat'.
    name : str
        Display name for logging.

    Returns
    -------
    dict with keys:
      'coords'      — normalized node coordinates (N, 3)
      'u'           — normalized displacement field (N, 3)
      'sigma'       — normalized stress tensor components (N, 6)
      'sigma_scale' — per-component physical scale (6,)
    """
    L_char     = g['L_char']
    u_char     = g[f'u_char_{mode}']
    sigma_char = g[f'sigma_char_{mode}']

    data        = np.loadtxt(path, delimiter=',', skiprows=1)
    coords_norm = data[:, 0:3] / L_char
    u_norm      = data[:, 3:6] / u_char

    sigma_raw   = data[:, 6:12]
    sigma_scale = np.maximum(np.abs(sigma_raw).max(axis=0), 1e-10)
    sigma_norm  = sigma_raw / sigma_scale

    u_max = np.linalg.norm(u_norm, axis=1).max()
    print(f"[OK] {name}: {len(coords_norm)} nodes | ||u||_max: {u_max:.4f} (norm)")

    return {
        'coords'     : coords_norm,
        'u'          : u_norm,
        'sigma'      : sigma_norm,
        'sigma_scale': sigma_scale,
    }