from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import GeoPINONet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoPINONet real inference demo for Lug_3D")
    parser.add_argument(
        "--body-stl",
        type=str,
        default=str(ROOT / "examples" / "Lug_3D" / "case_001" / "1_body.stl"),
        help="Path to the body STL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "trained_models" / "Lug_3D" / "64_lug_final.pth"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--n-query",
        type=int,
        default=50000,
        help="Number of surface query points.",
    )
    parser.add_argument(
        "--n-pc",
        type=int,
        default=2048,
        help="Number of point-cloud points used by the geometry encoder.",
    )
    parser.add_argument(
        "--f-comp",
        type=float,
        default=1000.0,
        help="Compression reference load contribution [N]. Default = 1000 N.",
    )
    parser.add_argument(
        "--f-lat",
        type=float,
        default=0.0,
        help="Lateral-bending reference load contribution [N]. Default = 0 N.",
    )
    parser.add_argument(
        "--ref-load",
        type=float,
        default=1000.0,
        help="Reference load used during training [N].",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "results" / "demo_inference" / "Lug_3D"),
        help="Output directory.",
    )
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    except Exception:
        pts = mesh.sample(n_points)
    return np.asarray(pts, dtype=np.float32)


def compute_geometry_scalars(points_xyz: np.ndarray) -> tuple[float, float, float]:
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)

    lengths = maxs - mins
    l_char = float(np.max(lengths))
    z_mid = float(0.5 * (mins[2] + maxs[2]))
    z_half = float(0.5 * (maxs[2] - mins[2]))

    if z_half <= 0.0:
        z_half = 1.0

    return l_char, z_mid, z_half


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def unpack_prediction(output):
    """
    Robust unpacker:
    - if model returns (u, sigma)
    - if model returns {"u":..., "sigma":...}
    - if model returns longer tuple/list, take first two
    """
    if isinstance(output, dict):
        if "u" in output and "sigma" in output:
            return output["u"], output["sigma"]
        raise ValueError(f"Unsupported dict output keys: {list(output.keys())}")

    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            return output[0], output[1]
        raise ValueError("Model output tuple/list has fewer than 2 elements.")

    raise ValueError(f"Unsupported model output type: {type(output)}")


def to_numpy_2d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def compute_vm_full(sig: np.ndarray) -> np.ndarray:
    sxx = sig[:, 0]
    syy = sig[:, 1]
    szz = sig[:, 2]
    sxy = sig[:, 3]
    syz = sig[:, 4]
    sxz = sig[:, 5]

    vm = np.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )
    return vm


def compute_vm_ip(sig: np.ndarray) -> np.ndarray:
    sxx = sig[:, 0]
    syy = sig[:, 1]
    sxy = sig[:, 3]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def main() -> None:
    args = parse_args()

    body_stl = Path(args.body_stl)
    model_path = Path(args.model)
    outdir = Path(args.outdir)

    if not body_stl.exists():
        raise FileNotFoundError(f"Body STL not found: {body_stl}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading mesh: {body_stl}")
    print(f"[INFO] Loading checkpoint: {model_path}")

    mesh = trimesh.load_mesh(body_stl, process=False)
    query_pts = sample_surface_points(mesh, args.n_query)
    pc_pts = sample_surface_points(mesh, args.n_pc)

    l_char, z_mid, z_half = compute_geometry_scalars(query_pts)

    # Normalize geometry by characteristic length
    query_norm = query_pts / l_char
    pc_norm = pc_pts / l_char

    model = GeoPINONet().to(device)
    load_checkpoint(model, model_path, device)

    pc_tensor = to_tensor(pc_norm, device)  # [Npc, 3]

    x = to_tensor(query_norm[:, 0:1], device)
    y = to_tensor(query_norm[:, 1:2], device)
    z = to_tensor(query_norm[:, 2:3], device)

    z_mid_t = torch.tensor([[z_mid / l_char]], dtype=torch.float32, device=device)
    z_half_t = torch.tensor([[z_half / l_char]], dtype=torch.float32, device=device)

    alpha_comp = args.f_comp / args.ref_load
    alpha_lat = args.f_lat / args.ref_load

    print(f"[INFO] alpha_comp = {alpha_comp:.4f}")
    print(f"[INFO] alpha_lat  = {alpha_lat:.4f}")
    print(f"[INFO] Query points = {len(query_pts)}")
    print(f"[INFO] Encoder point cloud = {len(pc_pts)}")

    with torch.no_grad():
        latent_comp = model.encode_geometry_comp(pc_tensor)
        latent_lat = model.encode_geometry_lat(pc_tensor)

        pred_comp = model.forward_comp(latent_comp, x, y, z, z_mid_t, z_half_t)
        pred_lat = model.forward_lat(latent_lat, x, y, z, z_mid_t, z_half_t)

        u_comp, sig_comp = unpack_prediction(pred_comp)
        u_lat, sig_lat = unpack_prediction(pred_lat)

    u_comp = to_numpy_2d(u_comp)
    u_lat = to_numpy_2d(u_lat)
    sig_comp = to_numpy_2d(sig_comp)
    sig_lat = to_numpy_2d(sig_lat)

    if u_comp.shape[1] != 3 or u_lat.shape[1] != 3:
        raise ValueError(f"Expected displacement with 3 columns, got {u_comp.shape} and {u_lat.shape}")

    if sig_comp.shape[1] < 6 or sig_lat.shape[1] < 6:
        raise ValueError(f"Expected stress tensor with at least 6 columns, got {sig_comp.shape} and {sig_lat.shape}")

    sig_comp = sig_comp[:, :6]
    sig_lat = sig_lat[:, :6]

    # Linear superposition
    u_comb = alpha_comp * u_comp + alpha_lat * u_lat
    sig_comb = alpha_comp * sig_comp + alpha_lat * sig_lat

    u_mag = np.linalg.norm(u_comb, axis=1)
    vm_full = compute_vm_full(sig_comb)
    vm_ip = compute_vm_ip(sig_comb)

    df = pd.DataFrame(
        {
            "X": query_pts[:, 0],
            "Y": query_pts[:, 1],
            "Z": query_pts[:, 2],
            "UX": u_comb[:, 0],
            "UY": u_comb[:, 1],
            "UZ": u_comb[:, 2],
            "u_mag": u_mag,
            "Sxx": sig_comb[:, 0],
            "Syy": sig_comb[:, 1],
            "Szz": sig_comb[:, 2],
            "Sxy": sig_comb[:, 3],
            "Syz": sig_comb[:, 4],
            "Sxz": sig_comb[:, 5],
            "vm_full": vm_full,
            "vm_ip": vm_ip,
        }
    )

    csv_path = outdir / "lug_prediction_combined.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Prediction CSV written to: {csv_path}")

    # Static PNG
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        query_pts[:, 0],
        query_pts[:, 1],
        query_pts[:, 2],
        c=vm_full,
        s=1.0,
        cmap="jet",
    )

    ax.set_title(
        f"GeoPINONet Lug Inference\n"
        f"F_comp = {args.f_comp:.1f} N, F_lat = {args.f_lat:.1f} N"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=80, azim=0, roll=90)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("von Mises stress")

    png_path = outdir / "lug_prediction_combined.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=250)
    plt.close(fig)

    print(f"[OK] PNG written to: {png_path}")
    print("[DONE] Real inference demo completed successfully.")


if __name__ == "__main__":
    main()