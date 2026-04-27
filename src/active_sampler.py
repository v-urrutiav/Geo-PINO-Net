# ============================================================
# active_sampler.py
# GeoPINONet — Active Critical Selective Validation (ActiveCSV) sampler
#
# Standalone module. Import directly to use adaptive node sampling
# during training without the rest of the codebase.
# ============================================================

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import (
    DEADZONE_THRESHOLD_VM,
    EPS_U_COMP, EPS_U_LAT,
    ACTIVE_BUDGET, ACTIVE_N_CRITICAL, ACTIVE_MIN_RADIUS,
    SUP_BATCH, CRITICAL_FRAC,
)
from .physics import von_mises_from_sigma
from .physics import compute_physics_unified


# ============================================================
# ACTIVE CSV SAMPLER
# ============================================================

class ActiveCSV:
    """
    Active Critical Selective Validation (ActiveCSV) sampler.

    Maintains two pools of training nodes per geometry:
      - Critical squad : nodes with highest Von Mises stress (supervised)
                         or highest PDE residual (unsupervised)
      - Dynamic squad  : nodes with highest current prediction error,
                         refreshed periodically during training

    Parameters
    ----------
    data         : dict from load_ansys_csv — keys: coords, u, sigma, sigma_scale
    budget       : total active nodes (critical + dynamic)
    min_radius   : minimum spatial distance between critical nodes (normalized)
    n_critical   : size of the critical squad
    load_mode    : 'comp' or 'lat'
    ranking_mode : 'supervised' (Von Mises) or 'unsupervised' (PDE residual)
    z_mid        : normalized mid-thickness coordinate
    z_half       : normalized half-thickness
    """

    def __init__(self, data: dict, budget: int = ACTIVE_BUDGET,
                 min_radius: float = ACTIVE_MIN_RADIUS,
                 n_critical: int = ACTIVE_N_CRITICAL,
                 load_mode: str = 'comp',
                 ranking_mode: str = 'supervised',
                 z_mid: float = 0.0, z_half: float = 1.0):

        self.load_mode    = load_mode
        self.ranking_mode = ranking_mode

        self.all_coords  = torch.as_tensor(data['coords'])
        self.all_u       = torch.as_tensor(data['u'])
        self.all_sigma   = torch.as_tensor(data['sigma'])
        self.sigma_scale = torch.as_tensor(data['sigma_scale'])

        self.sigma_true_phys = self.all_sigma * self.sigma_scale.unsqueeze(0)
        self.vm_true         = von_mises_from_sigma(self.sigma_true_phys)

        self.u_norm_mag    = torch.norm(self.all_u, dim=1)
        self.u_global_max  = self.u_norm_mag.max().clamp(min=1e-12)
        self.vm_global_max = self.vm_true.max().clamp(min=1e-12)

        self.vm_ranking = self.vm_true.clone()

        self.pool_size  = len(self.all_coords)
        self.budget     = budget
        self.min_radius = min_radius
        self.n_critical = n_critical
        self.n_dynamic  = budget - n_critical

        self.vm_order         = torch.argsort(self.vm_ranking, descending=True).tolist()
        self.critical_indices = self._select_initial_critical()

        available = [i for i in range(self.pool_size)
                     if i not in set(self.critical_indices)]
        if len(available) < self.n_dynamic:
            self.n_dynamic = len(available)
        self.dynamic_indices = random.sample(available, self.n_dynamic)

        self.z_mid  = z_mid
        self.z_half = z_half

        print(f"[ActiveCSV] mode={self.ranking_mode} | pool={self.pool_size} | "
              f"critical={len(self.critical_indices)} | dynamic={len(self.dynamic_indices)}")

    # ── Initialization ────────────────────────────────────────────────────────

    def _select_initial_critical(self) -> list:
        """Top-N nodes by Von Mises stress with minimum spatial radius filter."""
        selected   = []
        sel_coords = torch.empty((0, 3), dtype=self.all_coords.dtype)

        for idx in self.vm_order:
            if len(selected) >= self.n_critical:
                break
            if self.vm_ranking[idx].item() < DEADZONE_THRESHOLD_VM:
                break
            candidate = self.all_coords[idx].unsqueeze(0)
            if sel_coords.shape[0] > 0:
                if torch.norm(sel_coords - candidate, dim=1).min() < self.min_radius:
                    continue
            selected.append(idx)
            sel_coords = torch.cat([sel_coords, candidate], dim=0)

        return selected

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def active_indices(self) -> list:
        return self.critical_indices + self.dynamic_indices

    def get_training_data(self):
        idx = self.critical_indices + self.dynamic_indices
        return self.all_coords[idx], self.all_u[idx], self.all_sigma[idx]

    # ── Ranking update ────────────────────────────────────────────────────────

    def update_pde_ranking(self, model, latent_vector: torch.Tensor,
                           device: torch.device):
        """Recompute per-node ranking via PDE residual (unsupervised mode only)."""
        if self.ranking_mode != 'unsupervised':
            return

        model.eval()
        residuals = []
        with torch.enable_grad():
            for i in range(0, self.pool_size, 2000):
                batch = self.all_coords[i:i + 2000].to(device)
                res   = compute_physics_unified(
                    model, latent_vector, batch, mode=self.load_mode,
                    compute_pde=True, compute_neumann=False, compute_hooke=False,
                    z_mid=self.z_mid, z_half=self.z_half
                )
                residuals.append(res['pde_per_point'].detach().cpu())

        self.vm_ranking = torch.cat(residuals)
        self.vm_order   = torch.argsort(self.vm_ranking, descending=True).tolist()
        model.train()

    # ── Scan and refresh ──────────────────────────────────────────────────────

    def scan_and_update(self, model, latent_vector: torch.Tensor,
                        device: torch.device, epoch: int,
                        name: str, save_path: str):
        """Full-pool evaluation, squad refresh, and visualization."""
        print(f"\n{'='*60}")
        print(f"[SCAN] {name} — Epoch {epoch} | Pool: {self.pool_size} nodes")
        print(f"{'='*60}")

        model.eval()
        u_preds, sigma_preds = [], []

        with torch.no_grad():
            for i in range(0, self.pool_size, 2000):
                batch = self.all_coords[i:i + 2000].to(device)
                if self.load_mode == 'comp':
                    u_p, s_p = model.forward_comp(
                        latent_vector,
                        batch[:, 0:1], batch[:, 1:2], batch[:, 2:3],
                        self.z_mid, self.z_half)
                else:
                    u_p, s_p = model.forward_lat(
                        latent_vector,
                        batch[:, 0:1], batch[:, 1:2], batch[:, 2:3],
                        self.z_mid, self.z_half)
                u_preds.append(u_p.detach().cpu())
                sigma_preds.append(s_p.detach().cpu())

        u_pred_all     = torch.cat(u_preds)
        sigma_pred_all = torch.cat(sigma_preds)

        if self.ranking_mode == 'supervised':
            eps_u = EPS_U_COMP if self.load_mode == 'comp' else EPS_U_LAT
            error_u_abs = torch.clamp(
                torch.norm(u_pred_all - self.all_u, dim=1) - eps_u, min=0.0)
            error_u_rel = error_u_abs / self.u_global_max

            sigma_pred_phys = sigma_pred_all * self.sigma_scale.unsqueeze(0)
            vm_pred         = von_mises_from_sigma(sigma_pred_phys)
            error_vm_rel    = torch.abs(vm_pred - self.vm_true) / self.vm_global_max
            error_rel       = error_u_rel + error_vm_rel

            print(f"   [SUPERVISED | VM-based] Mean: {error_rel.mean()*100:.3f}% | "
                  f"Max: {error_rel.max()*100:.3f}%")
        else:
            vm_max      = self.vm_ranking.max().clamp(min=1e-12)
            error_rel   = self.vm_ranking / vm_max
            error_u_rel = error_rel
            print(f"   [UNSUPERVISED] Mean: {error_rel.mean()*100:.3f}% | "
                  f"Max: {error_rel.max()*100:.3f}%")

        self._refresh_critical_squad(error_rel)
        self._refresh_dynamic_squad(error_rel)
        self._visualize_squad(epoch, name, save_path)

        model.train()
        return error_u_rel, error_rel

    # ── Squad refresh ─────────────────────────────────────────────────────────

    def _graduation_threshold(self, error_rel: torch.Tensor) -> float:
        """Adaptive graduation threshold based on median global error."""
        median_pct = error_rel.median().item() * 100
        if median_pct <= 5:  return 0.05
        if median_pct <= 10: return 0.08
        if median_pct <= 25: return 0.10
        return 0.50

    def _refresh_critical_squad(self, error_rel: torch.Tensor):
        threshold = self._graduation_threshold(error_rel)
        survivors = [i for i in self.critical_indices if error_rel[i].item() >= threshold]
        graduated = len(self.critical_indices) - len(survivors)
        self.critical_indices = survivors
        slots = self.n_critical - len(self.critical_indices)

        print(f"   [CRITICAL] Graduated: {graduated} | Slots available: {slots}")
        if slots <= 0:
            return

        excluded   = set(self.critical_indices) | set(self.dynamic_indices)
        cur_coords = (self.all_coords[self.critical_indices] if self.critical_indices
                      else torch.empty((0, 3), dtype=self.all_coords.dtype))
        recruited = skipped = 0

        for idx in self.vm_order:
            if idx in excluded:
                continue
            if self.vm_ranking[idx].item() < DEADZONE_THRESHOLD_VM:
                break
            if error_rel[idx].item() < threshold:
                skipped += 1
                continue
            candidate = self.all_coords[idx].unsqueeze(0)
            if cur_coords.shape[0] > 0:
                if torch.norm(cur_coords - candidate, dim=1).min() < self.min_radius:
                    continue
            self.critical_indices.append(idx)
            cur_coords = torch.cat([cur_coords, candidate], dim=0)
            recruited += 1
            if recruited == slots:
                break

        print(f"   [CRITICAL] Recruited: {recruited} | Skipped (learned): {skipped}")

    def _refresh_dynamic_squad(self, error_rel: torch.Tensor):
        threshold = self._graduation_threshold(error_rel)
        survivors = [i for i in self.dynamic_indices if error_rel[i].item() >= threshold]
        graduated = len(self.dynamic_indices) - len(survivors)
        self.dynamic_indices = survivors
        slots = self.n_dynamic - len(self.dynamic_indices)

        print(f"   [DYNAMIC] Graduated: {graduated} | Slots available: {slots}")
        if slots <= 0:
            return

        excluded   = set(self.critical_indices) | set(self.dynamic_indices)
        order      = torch.argsort(error_rel, descending=True).tolist()
        cur_coords = (self.all_coords[self.dynamic_indices] if self.dynamic_indices
                      else torch.empty((0, 3), dtype=self.all_coords.dtype))
        recruited = rejected = 0

        for idx in order:
            if idx in excluded:
                continue
            if self.vm_ranking[idx].item() < DEADZONE_THRESHOLD_VM:
                continue
            candidate = self.all_coords[idx].unsqueeze(0)
            if cur_coords.shape[0] > 0:
                if torch.norm(cur_coords - candidate, dim=1).min() < self.min_radius:
                    rejected += 1
                    continue
            self.dynamic_indices.append(idx)
            cur_coords = torch.cat([cur_coords, candidate], dim=0)
            recruited += 1
            if recruited == slots:
                break

        print(f"   [DYNAMIC] Recruited: {recruited} | Rejected (radius): {rejected}")

        if recruited < slots:
            remaining = slots - recruited
            excluded  = set(self.critical_indices) | set(self.dynamic_indices)
            print(f"   [INFO] Filling {remaining} slots without spatial filter")
            for idx in order:
                if idx not in excluded:
                    self.dynamic_indices.append(idx)
                    recruited += 1
                    if recruited == slots:
                        break

    # ── Visualization ─────────────────────────────────────────────────────────

    def _visualize_squad(self, epoch: int, name: str, save_path: str):
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax  = fig.add_subplot(111, projection='3d')

        all_c  = self.all_coords.detach().cpu().numpy()
        crit_c = self.all_coords[self.critical_indices].detach().cpu().numpy()
        din_c  = self.all_coords[self.dynamic_indices].detach().cpu().numpy()

        ax.scatter(all_c[:, 0], all_c[:, 1], all_c[:, 2],
                   c='lightgray', s=1, alpha=0.05, label='Domain')

        if len(din_c) > 0:
            ax.scatter(din_c[:, 0], din_c[:, 1], din_c[:, 2],
                       c='#6E6E6E', s=20, alpha=0.8, marker='o', edgecolors='none',
                       label=f'Dynamic ({len(self.dynamic_indices)})')

        if len(crit_c) > 0:
            ax.scatter(crit_c[:, 0], crit_c[:, 1], crit_c[:, 2],
                       c='#0B5C84', s=25, alpha=1.0, marker='o', edgecolors='none',
                       label=f'Critical ({len(self.critical_indices)})')

        ranges = np.ptp(all_c, axis=0)
        ranges[ranges == 0.0] = 1.0
        ax.set_box_aspect(ranges)

        ax.view_init(elev=90, azim=-90)
        ax.axis('off')
        ax.set_title(f'GeoPINONet — ActiveCSV | {name} (Epoch {epoch})', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f'squad_{name.lower()}_epoch{epoch}.pdf')

        plt.tight_layout()
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.close()

        print(f"   [IMG] Squad plot saved: {path}")

    

# ============================================================
# ACTIVE SUPERVISION BATCH SAMPLER
# ============================================================

def sample_active_supervision(active_obj: ActiveCSV, n_total: int = SUP_BATCH,
                               critical_frac: float = CRITICAL_FRAC,
                               device: torch.device = None):
    """
    Return a supervision batch biased toward critical nodes.

    Parameters
    ----------
    active_obj    : ActiveCSV instance
    n_total       : total batch size
    critical_frac : fraction of batch drawn from critical squad
    device        : target device for output tensors

    Returns
    -------
    Tuple of (coords, u, sigma) tensors on device.
    """
    n_c = min(len(active_obj.critical_indices), int(n_total * critical_frac))
    n_d = min(len(active_obj.dynamic_indices),  n_total - n_c)

    idx = []
    if n_c > 0:
        idx.extend(random.sample(active_obj.critical_indices, n_c))
    if n_d > 0:
        idx.extend(random.sample(active_obj.dynamic_indices, n_d))
    if len(idx) == 0:
        n_fallback = min(n_total, active_obj.pool_size)
        idx = random.sample(range(active_obj.pool_size), n_fallback)

    idx = torch.tensor(idx, dtype=torch.long)
    return (
        active_obj.all_coords[idx].to(device),
        active_obj.all_u[idx].to(device),
        active_obj.all_sigma[idx].to(device),
    )