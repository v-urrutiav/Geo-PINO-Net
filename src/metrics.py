# ============================================================
# metrics.py
# GeoPINONet — Evaluation metrics for displacement and stress fields
# ============================================================

import numpy as np
import torch

from .physics import von_mises_from_sigma


def von_mises_ip_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute in-plane von Mises stress from Cauchy stress components.

    Expected stress order:
        [Sxx, Syy, Szz, Sxy, Syz, Sxz]

    In-plane von Mises uses:
        Sxx, Syy, Sxy

    sigma_vM,ip = sqrt(Sxx^2 - Sxx*Syy + Syy^2 + 3*Sxy^2)
    """
    sxx = sigma[:, 0]
    syy = sigma[:, 1]
    sxy = sigma[:, 3]

    return torch.sqrt(torch.clamp(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, min=0.0))


def _field_metrics(pred: torch.Tensor, true: torch.Tensor, global_max: torch.Tensor):
    """
    Generic vector-field percentile metrics normalized by a global maximum.
    Used for displacement.
    """
    l2_rel = (torch.norm(pred - true) / (torch.norm(true) + 1e-12)).item() * 100.0
    mae = torch.mean(torch.norm(pred - true, dim=1)).item()

    ss_res = torch.sum((pred - true) ** 2)
    ss_tot = torch.sum((true - true.mean(dim=0)) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()

    err_per_node = (
        torch.norm(pred - true, dim=1) / (global_max + 1e-12) * 100.0
    ).detach().cpu().numpy()

    return l2_rel, mae, r2, err_per_node


def _scalar_metrics(pred: torch.Tensor, true: torch.Tensor, global_max: torch.Tensor):
    """
    Generic scalar-field percentile metrics normalized by a global maximum.
    Used for von Mises stress.
    """
    l2_rel = (torch.norm(pred - true) / (torch.norm(true) + 1e-12)).item() * 100.0
    mae = torch.mean(torch.abs(pred - true)).item()

    ss_res = torch.sum((pred - true) ** 2)
    ss_tot = torch.sum((true - true.mean()) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()

    err_per_node = (
        torch.abs(pred - true) / (global_max + 1e-12) * 100.0
    ).detach().cpu().numpy()

    return l2_rel, mae, r2, err_per_node


def _percentiles(arr: np.ndarray):
    return {
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
    }


def compute_global_metrics(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    sigma_pred: torch.Tensor,
    sigma_true: torch.Tensor,
    sigma_scale: torch.Tensor,
    displacement_components: str = "xy",
) -> dict:
    """
    Compute displacement, full von Mises, and in-plane von Mises error metrics.

    Parameters
    ----------
    u_pred : torch.Tensor
        Predicted displacement field, shape (N, 3).
    u_true : torch.Tensor
        FEM reference displacement field, shape (N, 3).
    sigma_pred : torch.Tensor
        Predicted stress tensor components, normalized, shape (N, 6).
        Expected order: [Sxx, Syy, Szz, Sxy, Syz, Sxz].
    sigma_true : torch.Tensor
        FEM reference stress tensor components, normalized, shape (N, 6).
    sigma_scale : torch.Tensor
        Per-component physical stress scale for denormalization, shape (6,).
    displacement_components : str
        "xy" uses UX and UY only.
        "xyz" uses UX, UY, and UZ.

    Returns
    -------
    dict
        Metrics used in GeoPINONet tables and logs.

    Notes
    -----
    The paper primarily reports in-plane von Mises stress metrics,
    identified with the suffix `_vm_ip`.
    Full von Mises metrics are retained for diagnostic compatibility.
    """
    with torch.no_grad():
        sigma_scale = sigma_scale.to(sigma_pred.device)

        sigma_pred_phys = sigma_pred * sigma_scale
        sigma_true_phys = sigma_true * sigma_scale

        # Full 3D von Mises
        vm_pred = von_mises_from_sigma(sigma_pred_phys)
        vm_true = von_mises_from_sigma(sigma_true_phys)

        # In-plane von Mises
        vm_pred_ip = von_mises_ip_from_sigma(sigma_pred_phys)
        vm_true_ip = von_mises_ip_from_sigma(sigma_true_phys)

        # Displacement selection
        if displacement_components.lower() == "xy":
            u_pred_eval = u_pred[:, 0:2]
            u_true_eval = u_true[:, 0:2]
        elif displacement_components.lower() == "xyz":
            u_pred_eval = u_pred[:, 0:3]
            u_true_eval = u_true[:, 0:3]
        else:
            raise ValueError("displacement_components must be either 'xy' or 'xyz'.")

        u_global_max = torch.norm(u_true_eval, dim=1).max()

        l2u, maeu, r2u, pct_u = _field_metrics(
            u_pred_eval,
            u_true_eval,
            u_global_max,
        )

        l2vm, maevm, r2vm, pct_vm = _scalar_metrics(
            vm_pred,
            vm_true,
            vm_true.max(),
        )

        l2vm_ip, maevm_ip, r2vm_ip, pct_vm_ip = _scalar_metrics(
            vm_pred_ip,
            vm_true_ip,
            vm_true_ip.max(),
        )

        pu = _percentiles(pct_u)
        pvm = _percentiles(pct_vm)
        pvm_ip = _percentiles(pct_vm_ip)

        return {
            # Displacement
            "L2_rel_u": round(l2u, 4),
            "MAE_u": round(maeu, 8),
            "R2_u": round(r2u, 6),

            "p50_err": pu["p50"],
            "p90_err": pu["p90"],
            "p95_err": pu["p95"],
            "P95_err": pu["p95"],  # backward compatibility with old logs
            "p99_err": pu["p99"],

            # Full von Mises diagnostics
            "L2_rel_vm": round(l2vm, 4),
            "MAE_vm": round(maevm, 8),
            "R2_vm": round(r2vm, 6),

            "p50_vm": pvm["p50"],
            "p90_vm": pvm["p90"],
            "p95_vm": pvm["p95"],
            "p99_vm": pvm["p99"],

            # In-plane von Mises metrics used in the paper
            "L2_rel_vm_ip": round(l2vm_ip, 4),
            "MAE_vm_ip": round(maevm_ip, 8),
            "R2_vm_ip": round(r2vm_ip, 6),

            "p50_vm_ip": pvm_ip["p50"],
            "p90_vm_ip": pvm_ip["p90"],
            "p95_vm_ip": pvm_ip["p95"],
            "p99_vm_ip": pvm_ip["p99"],
        }