# ============================================================
# physics.py
# GeoPINONet — Physics utilities, PDE residual and constitutive losses
# ============================================================

import torch
import torch.nn as nn
from .config import (
    NU_physical,
    CURRICULUM_PHASE1_EPOCHS,
    CURRICULUM_PHASE2_EPOCHS,
)

# Normalized Poisson's ratio (material constant, does not change across geometries)
NU_norm = NU_physical


# ============================================================
# LAMÉ PARAMETERS
# ============================================================

def get_lame_params(e_norm: float, nu: float):
    """Return normalized Lamé parameters (lambda, mu) from E_norm and nu."""
    mu  = e_norm / (2 * (1 + nu))
    lam = (e_norm * nu) / ((1 + nu) * (1 - 2 * nu))
    return lam, mu


# ============================================================
# STRESS UTILITIES
# ============================================================

def von_mises_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """Von Mises stress from flat (N, 6) tensor [Sxx, Syy, Szz, Sxy, Syz, Sxz]."""
    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    sxy, syz, sxz = sigma[:, 3], sigma[:, 4], sigma[:, 5]
    return torch.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
               + 6 * (sxy**2 + syz**2 + sxz**2)) + 1e-8
    )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def epsilon_insensitive_loss(pred: torch.Tensor, target: torch.Tensor,
                              eps: float) -> torch.Tensor:
    """Deadzone squared loss: penalizes deviations beyond eps quadratically."""
    diff = torch.abs(pred - target) - eps
    return (torch.clamp(diff, min=0.0) ** 2).mean() + 1e-8


# ============================================================
# CURRICULUM SCHEDULE
# ============================================================

def get_physics_weight(epoch: int) -> float:
    """
    Curriculum schedule for physics loss activation:
      Phase 1 (0 – PHASE1)      : physics weight = 0 (data-only)
      Phase 2 (PHASE1 – PHASE2) : linear ramp 0 → 1
      Phase 3 (PHASE2+)         : physics weight = 1
    """
    if epoch <= CURRICULUM_PHASE1_EPOCHS:
        return 0.0
    if epoch <= CURRICULUM_PHASE2_EPOCHS:
        return ((epoch - CURRICULUM_PHASE1_EPOCHS) /
                (CURRICULUM_PHASE2_EPOCHS - CURRICULUM_PHASE1_EPOCHS))
    return 1.0


# ============================================================
# VISUALIZATION UTILITY
# ============================================================

def subsample_for_viz(coords: torch.Tensor, *arrays, max_points: int = 10000):
    """Random subsample for fast visualization."""
    n = len(coords)
    if n <= max_points:
        return (coords,) + arrays
    idx = torch.randperm(n)[:max_points]
    return (coords[idx],) + tuple(a[idx] for a in arrays)


# ============================================================
# PHYSICS RESIDUAL COMPUTATION
# ============================================================

def compute_physics_unified(model, latent_vector: torch.Tensor,
                             coords_norm: torch.Tensor,
                             mode: str = 'comp',
                             compute_pde: bool = True,
                             compute_neumann: bool = False,
                             normals_hat: torch.Tensor = None,
                             t_target_norm: torch.Tensor = None,
                             compute_hooke: bool = True,
                             e_norm: float = None,
                             z_mid: float = 0.0,
                             z_half: float = 1.0) -> dict:
    """
    Evaluate PDE residual, Neumann BC, and Hooke's law loss at given points.

    Returns a dict with keys:
      'pde_loss'      — mean equilibrium residual (∇·σ = 0)
      'pde_per_point' — per-node residual (N,)
      'neumann_loss'  — traction boundary condition loss
      'hooke_loss'    — constitutive consistency loss
    Keys are only present if their corresponding flag is True.
    """
    coords_norm = coords_norm.clone().detach().requires_grad_(True)

    if mode == 'comp':
        u_pred, sigma_pred_flat = model.forward_comp(
            latent_vector,
            coords_norm[:, 0:1], coords_norm[:, 1:2], coords_norm[:, 2:3],
            z_mid, z_half)
    else:
        u_pred, sigma_pred_flat = model.forward_lat(
            latent_vector,
            coords_norm[:, 0:1], coords_norm[:, 1:2], coords_norm[:, 2:3],
            z_mid, z_half)

    N       = sigma_pred_flat.shape[0]
    results = {}

    if compute_pde:
        grads = [
            torch.autograd.grad(
                sigma_pred_flat[:, k].sum(), coords_norm,
                create_graph=True, retain_graph=True)[0]
            for k in range(6)
        ]
        div_x = grads[0][:, 0] + grads[3][:, 1] + grads[5][:, 2]
        div_y = grads[3][:, 0] + grads[1][:, 1] + grads[4][:, 2]
        div_z = grads[5][:, 0] + grads[4][:, 1] + grads[2][:, 2]
        div_sigma             = torch.stack([div_x, div_y, div_z], dim=1)
        pde_per_point         = (div_sigma ** 2).sum(dim=1)
        results['pde_loss']      = pde_per_point.mean()
        results['pde_per_point'] = pde_per_point

    if compute_neumann and normals_hat is not None and t_target_norm is not None:
        s = torch.zeros(N, 3, 3, device=coords_norm.device, dtype=coords_norm.dtype)
        s[:, 0, 0] = sigma_pred_flat[:, 0]
        s[:, 1, 1] = sigma_pred_flat[:, 1]
        s[:, 2, 2] = sigma_pred_flat[:, 2]
        s[:, 0, 1] = s[:, 1, 0] = sigma_pred_flat[:, 3]
        s[:, 1, 2] = s[:, 2, 1] = sigma_pred_flat[:, 4]
        s[:, 0, 2] = s[:, 2, 0] = sigma_pred_flat[:, 5]
        t_pred = torch.einsum('bij,bj->bi', s, normals_hat)
        results['neumann_loss'] = nn.MSELoss()(t_pred, t_target_norm.expand_as(t_pred))

    if compute_hooke:
        _e       = e_norm if e_norm is not None else 1.0
        lam, mu  = get_lame_params(_e, NU_norm)
        J        = torch.zeros(N, 3, 3, dtype=coords_norm.dtype, device=coords_norm.device)
        for i in range(3):
            g = torch.autograd.grad(
                u_pred[:, i].sum(), coords_norm,
                create_graph=False, retain_graph=True)[0]
            J[:, i, :] = g
        eps_strain = 0.5 * (J + J.transpose(-2, -1))
        tr         = eps_strain.diagonal(dim1=-2, dim2=-1).sum(-1)
        I          = torch.eye(3, device=coords_norm.device,
                               dtype=coords_norm.dtype).expand(N, 3, 3)
        sigma_kin  = lam * tr[:, None, None] * I + 2 * mu * eps_strain

        sp = torch.zeros(N, 3, 3, device=coords_norm.device, dtype=coords_norm.dtype)
        sp[:, 0, 0] = sigma_pred_flat[:, 0]
        sp[:, 1, 1] = sigma_pred_flat[:, 1]
        sp[:, 2, 2] = sigma_pred_flat[:, 2]
        sp[:, 0, 1] = sp[:, 1, 0] = sigma_pred_flat[:, 3]
        sp[:, 1, 2] = sp[:, 2, 1] = sigma_pred_flat[:, 4]
        sp[:, 0, 2] = sp[:, 2, 0] = sigma_pred_flat[:, 5]
        results['hooke_loss'] = nn.MSELoss()(sp / _e, sigma_kin / _e)

    return results