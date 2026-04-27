# ============================================================
# train.py
# GeoPINONet — Training loop, checkpoint management, post-training evaluation
# ============================================================

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import trimesh

from .config import (
    device, use_amp, amp_dtype,
    MODEL_SAVE_PATH, LOG_CSV_PATH, SCAN_LOG_PATH,
    ADAM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    DATA_SCALAR, SIGMA_SCALAR, CHECKPOINT_FREQUENCY,
    EPS_U_COMP, EPS_U_LAT, EPS_UZ_COMP, EPS_UZ_LAT, EPS_SIG_COMP, EPS_SIG_LAT,
    ACTIVE_BUDGET, ACTIVE_N_CRITICAL, ACTIVE_MIN_RADIUS, ACTIVE_CHECK_FREQUENCY,
    SUP_BATCH, CRITICAL_FRAC,
    VM_LOSS_WEIGHT, VM_WEIGHT_BOOST, VM_EPS,
)
from .domain import BASE_PATH, GEOMETRIES, GEOMETRIES_VAL
from .model import GeoPINONet
from .physics import (
    compute_physics_unified, get_physics_weight,
    epsilon_insensitive_loss, von_mises_from_sigma,
)
from .dataset import load_ansys_csv, compute_geometry_scales
from .active_sampler import ActiveCSV, sample_active_supervision
from .metrics import compute_global_metrics

scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# ============================================================
# HELPERS
# ============================================================

def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _predict_full(sampler: ActiveCSV, latent: torch.Tensor,
                  model: GeoPINONet, mode: str,
                  batch_size: int = 12000):
    """Predict u and sigma over the full pool in batches (no grad)."""
    u_list, s_list = [], []
    with torch.no_grad():
        for i in range(0, sampler.pool_size, batch_size):
            batch = sampler.all_coords[i:i + batch_size].to(device)
            if mode == 'comp':
                u, s = model.forward_comp(
                    latent, batch[:, 0:1], batch[:, 1:2], batch[:, 2:3],
                    sampler.z_mid, sampler.z_half)
            else:
                u, s = model.forward_lat(
                    latent, batch[:, 0:1], batch[:, 1:2], batch[:, 2:3],
                    sampler.z_mid, sampler.z_half)
            u_list.append(u.cpu())
            s_list.append(s.cpu())
    return torch.cat(u_list), torch.cat(s_list)


def _refresh_latents(model, pointclouds, latents_comp, latents_lat):
    with torch.no_grad():
        for i, pc in enumerate(pointclouds):
            latents_comp[i] = model.encode_geometry_comp(pc).detach()
            latents_lat[i]  = model.encode_geometry_lat(pc).detach()


# ============================================================
# MAIN
# ============================================================

def main():
    _set_seed(42)
    print(f"[INIT] Seed: 42 | Device: {device}")

    loss_history = []
    scan_history = []

    try:
        # ── Compute per-geometry scales ────────────────────────────────────────
        compute_geometry_scales(GEOMETRIES + GEOMETRIES_VAL, BASE_PATH)

        # ── Load meshes and point clouds ───────────────────────────────────────
        print("[INIT] Loading all training geometries...")
        all_interior_pts = []
        all_load_pts     = []
        all_load_normals = []
        all_fixed_pts    = []
        all_pointclouds  = []

        for g in GEOMETRIES:
            mv = trimesh.load(os.path.join(BASE_PATH, g['vol']))
            ml = trimesh.load(os.path.join(BASE_PATH, g['load']))
            mf = trimesh.load(os.path.join(BASE_PATH, g['fixed']))

            int_pts          = torch.from_numpy(mv.sample(300000)).float() * 1e-3 / g['L_char']
            lpts, lface_idx  = trimesh.sample.sample_surface(ml, 25000)
            load_pts         = torch.from_numpy(lpts).float() * 1e-3 / g['L_char']
            fix_pts          = torch.from_numpy(mf.sample(25000)).float() * 1e-3 / g['L_char']
            lnormals         = torch.from_numpy(ml.face_normals[lface_idx]).float()
            pc               = torch.cat([int_pts, load_pts, fix_pts], dim=0)

            all_interior_pts.append(int_pts.to(device))
            all_load_pts.append(load_pts.to(device))
            all_load_normals.append(lnormals.to(device))
            all_fixed_pts.append(fix_pts.to(device))
            all_pointclouds.append(pc.to(device))

        print(f"[INIT] {len(GEOMETRIES)} geometries loaded.")

        # ── Load FEM data & create active samplers ─────────────────────────────
        all_active_comp, all_active_lat = [], []
        for g in GEOMETRIES:
            cd = load_ansys_csv(os.path.join(BASE_PATH, g['comp']),
                                g=g, mode='comp', name=g['comp'])
            ld = load_ansys_csv(os.path.join(BASE_PATH, g['lat']),
                                g=g, mode='lat',  name=g['lat'])
            all_active_comp.append(ActiveCSV(
                cd, budget=ACTIVE_BUDGET, min_radius=ACTIVE_MIN_RADIUS,
                n_critical=ACTIVE_N_CRITICAL, load_mode='comp',
                z_mid=g['z_mid'], z_half=g['z_half']))
            all_active_lat.append(ActiveCSV(
                ld, budget=ACTIVE_BUDGET, min_radius=ACTIVE_MIN_RADIUS,
                n_critical=ACTIVE_N_CRITICAL, load_mode='lat',
                z_mid=g['z_mid'], z_half=g['z_half']))

        # ── Model, tractions, optimizer ────────────────────────────────────────
        print("[INIT] Initializing GeoPINONet...")
        model = GeoPINONet().to(device)

        traction_comp = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
        traction_lat  = torch.tensor([1.0,  0.0, 0.0], device=device, dtype=torch.float32)

        optimizer = torch.optim.Adam(
            list(model.geometric_encoder_comp.parameters()) +
            list(model.positional_encoder_comp.parameters()) +
            list(model.decoder_comp.parameters()) +
            list(model.geometric_encoder_lat.parameters()) +
            list(model.positional_encoder_lat.parameters()) +
            list(model.decoder_lat.parameters()) +
            [model.log_vars],
            lr=LEARNING_RATE)

        # ── Checkpoint resume ──────────────────────────────────────────────────
        start_epoch = 1
        if os.path.exists(MODEL_SAVE_PATH):
            print("[CKPT] Loading checkpoint...")
            ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_physics_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_physics_state_dict'])
            if 'log_vars' in ckpt:
                model.log_vars.data = ckpt['log_vars'].data.to(device)
            start_epoch = ckpt.get('epoch', 1) + 1
            print(f"[CKPT] Resuming from epoch {start_epoch}.")

        # ── Pre-compute geometry latents ───────────────────────────────────────
        n_geos           = len(GEOMETRIES)
        all_latents_comp = [None] * n_geos
        all_latents_lat  = [None] * n_geos
        _refresh_latents(model, all_pointclouds, all_latents_comp, all_latents_lat)

        SIGMA_WEIGHTS = torch.ones(6, dtype=torch.float32, device=device)

        # ══════════════════════════════════════════════════════════════════════
        # TRAINING LOOP
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n[TRAIN] Epochs {start_epoch}–{ADAM_EPOCHS} | {n_geos} geometries per step")
        time_measurements = []

        for epoch in range(start_epoch, ADAM_EPOCHS + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            model.train()

            # Refresh latents every 50 epochs
            if epoch % 50 == 0:
                _refresh_latents(model, all_pointclouds, all_latents_comp, all_latents_lat)

            # ── Scan & squad update ────────────────────────────────────────────
            if epoch >= ACTIVE_CHECK_FREQUENCY and epoch % ACTIVE_CHECK_FREQUENCY == 0:
                model.eval()
                eval_idx = (epoch // ACTIVE_CHECK_FREQUENCY) % n_geos
                geo_name = f"Geo{eval_idx + 1}"
                print(f"[SCAN] Evaluating {geo_name}")

                g_eval = GEOMETRIES[eval_idx]
                lat_c  = all_latents_comp[eval_idx]
                lat_l  = all_latents_lat[eval_idx]
                ac     = all_active_comp[eval_idx]
                al     = all_active_lat[eval_idx]

                err_rel_c = err_rel_l = None

                if ac is not None:
                    ac.update_pde_ranking(model, lat_c, device)
                    _, err_rel_c = ac.scan_and_update(model, lat_c, device, epoch, "COMP", BASE_PATH)
                    err_arr_c = err_rel_c.numpy() * 100
                    scan_history.append({
                        'epoch': epoch, 'geo': geo_name, 'mode': 'COMP',
                        'error_mean_pct' : float(err_arr_c.mean()),
                        'error_max_pct'  : float(err_arr_c.max()),
                        'p50_pct'        : float(np.percentile(err_arr_c, 50)),
                        'p90_pct'        : float(np.percentile(err_arr_c, 90)),
                        'p95_pct'        : float(np.percentile(err_arr_c, 95)),
                        'nodes_lt_5pct'  : float((err_arr_c < 5).mean()  * 100),
                        'nodes_lt_10pct' : float((err_arr_c < 10).mean() * 100),
                        'nodes_lt_50pct' : float((err_arr_c < 50).mean() * 100),
                    })

                if al is not None:
                    al.update_pde_ranking(model, lat_l, device)
                    _, err_rel_l = al.scan_and_update(model, lat_l, device, epoch, "LAT", BASE_PATH)
                    err_arr_l = err_rel_l.numpy() * 100
                    scan_history.append({
                        'epoch': epoch, 'geo': geo_name, 'mode': 'LAT',
                        'error_mean_pct' : float(err_arr_l.mean()),
                        'error_max_pct'  : float(err_arr_l.max()),
                        'p50_pct'        : float(np.percentile(err_arr_l, 50)),
                        'p90_pct'        : float(np.percentile(err_arr_l, 90)),
                        'p95_pct'        : float(np.percentile(err_arr_l, 95)),
                        'nodes_lt_5pct'  : float((err_arr_l < 5).mean()  * 100),
                        'nodes_lt_10pct' : float((err_arr_l < 10).mean() * 100),
                        'nodes_lt_50pct' : float((err_arr_l < 50).mean() * 100),
                    })


                pd.DataFrame(scan_history).to_csv(SCAN_LOG_PATH, index=False)
                print(f"[LOG] Scan log saved ({len(scan_history)} entries)")
                model.train()

            # ── Gradient accumulation over all geometries ──────────────────────
            optimizer.zero_grad()
            loss_total = 0.0
            log = {k: 0.0 for k in ['pde_c','pde_l','dir_c','dir_l',
                                     'dat_c','dat_l','sig_c','sig_l',
                                     'vm_c','vm_l','hoo_c','hoo_l']}
            physics_w = get_physics_weight(epoch)

            for geo_idx in range(n_geos):
                g      = GEOMETRIES[geo_idx]
                pc_sub = all_pointclouds[geo_idx]
                idx_pc = torch.randperm(len(pc_sub), device=device)[:8000]
                lc = model.encode_geometry_comp(pc_sub[idx_pc])
                ll = model.encode_geometry_lat(pc_sub[idx_pc])
                ac = all_active_comp[geo_idx]
                al = all_active_lat[geo_idx]

                int_pts = all_interior_pts[geo_idx]
                lp_pts  = all_load_pts[geo_idx]
                ln_pts  = all_load_normals[geo_idx]
                fx_pts  = all_fixed_pts[geo_idx]

                idx_i = torch.randperm(len(int_pts), device=device)[:BATCH_SIZE // 2]
                idx_l = torch.randperm(len(lp_pts),  device=device)[:BATCH_SIZE // 4]
                idx_f = torch.randperm(len(fx_pts),  device=device)[:BATCH_SIZE // 4]
                b_int = int_pts[idx_i]
                b_lp  = lp_pts[idx_l];  b_ln = ln_pts[idx_l]
                b_fx  = fx_pts[idx_f]

                # PDE + Hooke
                res_c = compute_physics_unified(
                    model, lc, b_int, mode='comp',
                    compute_pde=True, compute_hooke=True,
                    e_norm=g['E_norm_comp'], z_mid=g['z_mid'], z_half=g['z_half'])
                res_l = compute_physics_unified(
                    model, ll, b_int, mode='lat',
                    compute_pde=True, compute_hooke=True,
                    e_norm=g['E_norm_lat'], z_mid=g['z_mid'], z_half=g['z_half'])

                # Dirichlet BC
                u_fc, _ = model.forward_comp(lc, b_fx[:,0:1], b_fx[:,1:2], b_fx[:,2:3], g['z_mid'], g['z_half'])
                u_fl, _ = model.forward_lat(ll, b_fx[:,0:1], b_fx[:,1:2], b_fx[:,2:3], g['z_mid'], g['z_half'])
                loss_dir_c = nn.MSELoss()(u_fc, torch.zeros_like(u_fc))
                loss_dir_l = nn.MSELoss()(u_fl, torch.zeros_like(u_fl))

                # Neumann BC
                neu_c = compute_physics_unified(
                    model, lc, b_lp, mode='comp',
                    compute_pde=False, compute_hooke=False, compute_neumann=True,
                    normals_hat=b_ln, t_target_norm=traction_comp,
                    z_mid=g['z_mid'], z_half=g['z_half'])
                neu_l = compute_physics_unified(
                    model, ll, b_lp, mode='lat',
                    compute_pde=False, compute_hooke=False, compute_neumann=True,
                    normals_hat=b_ln, t_target_norm=traction_lat,
                    z_mid=g['z_mid'], z_half=g['z_half'])

                # Data supervision
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    # COMP
                    cc, uc, sc = sample_active_supervision(ac, n_total=SUP_BATCH,
                                                            critical_frac=CRITICAL_FRAC, device=device)
                    up_c, sp_c = model.forward_comp(lc, cc[:,0:1], cc[:,1:2], cc[:,2:3], g['z_mid'], g['z_half'])

                    loss_data_c = (
                        epsilon_insensitive_loss(up_c[:,0:1], uc[:,0:1], EPS_U_COMP)   * DATA_SCALAR +
                        epsilon_insensitive_loss(up_c[:,1:2], uc[:,1:2], EPS_U_COMP)   * DATA_SCALAR +
                        epsilon_insensitive_loss(up_c[:,2:3], uc[:,2:3], EPS_UZ_COMP)  * DATA_SCALAR * 0.01
                    )
                    loss_sigma_c = epsilon_insensitive_loss(
                        sp_c * SIGMA_WEIGHTS, sc * SIGMA_WEIGHTS, EPS_SIG_COMP) * SIGMA_SCALAR

                    sigma_scale_c = ac.sigma_scale.to(device)
                    vm_pred_c = von_mises_from_sigma(sp_c * sigma_scale_c) / ac.vm_global_max.to(device)
                    vm_true_c = von_mises_from_sigma(sc   * sigma_scale_c) / ac.vm_global_max.to(device)
                    vm_weight_c   = 1.0 + VM_WEIGHT_BOOST * vm_true_c.detach()
                    vm_deadzone_c = torch.clamp(torch.abs(vm_pred_c - vm_true_c) - VM_EPS, min=0.0)
                    loss_vm_c     = (vm_weight_c * vm_deadzone_c ** 2).mean() * VM_LOSS_WEIGHT
                    loss_data_sig_c = loss_sigma_c + loss_vm_c

                    # LAT
                    cl, ul, sl = sample_active_supervision(al, n_total=SUP_BATCH,
                                                            critical_frac=CRITICAL_FRAC, device=device)
                    up_l, sp_l = model.forward_lat(ll, cl[:,0:1], cl[:,1:2], cl[:,2:3], g['z_mid'], g['z_half'])

                    loss_data_l = (
                        epsilon_insensitive_loss(up_l[:,0:1], ul[:,0:1], EPS_U_LAT)   * DATA_SCALAR +
                        epsilon_insensitive_loss(up_l[:,1:2], ul[:,1:2], EPS_U_LAT)   * DATA_SCALAR +
                        epsilon_insensitive_loss(up_l[:,2:3], ul[:,2:3], EPS_UZ_LAT)  * DATA_SCALAR * 0.01
                    )
                    loss_sigma_l = epsilon_insensitive_loss(
                        sp_l * SIGMA_WEIGHTS, sl * SIGMA_WEIGHTS, EPS_SIG_LAT) * SIGMA_SCALAR

                    sigma_scale_l = al.sigma_scale.to(device)
                    vm_pred_l = von_mises_from_sigma(sp_l * sigma_scale_l) / al.vm_global_max.to(device)
                    vm_true_l = von_mises_from_sigma(sl   * sigma_scale_l) / al.vm_global_max.to(device)
                    vm_weight_l   = 1.0 + VM_WEIGHT_BOOST * vm_true_l.detach()
                    vm_deadzone_l = torch.clamp(torch.abs(vm_pred_l - vm_true_l) - VM_EPS, min=0.0)
                    loss_vm_l     = (vm_weight_l * vm_deadzone_l ** 2).mean() * VM_LOSS_WEIGHT
                    loss_data_sig_l = loss_sigma_l + loss_vm_l

                # Homoscedastic weighting
                pc_arr = [torch.exp(-model.log_vars[i]) for i in range(6)]
                pl_arr = [torch.exp(-model.log_vars[i]) for i in range(6, 12)]

                loss_geo = (
                    physics_w * (pc_arr[0] * res_c['pde_loss']    + model.log_vars[0]) +
                    physics_w * (pc_arr[1] * loss_dir_c            + model.log_vars[1]) +
                    physics_w * (pc_arr[2] * neu_c['neumann_loss'] + model.log_vars[2]) +
                               (pc_arr[3] * loss_data_c            + model.log_vars[3]) +
                               (pc_arr[4] * loss_data_sig_c        + model.log_vars[4]) +
                    physics_w * (pc_arr[5] * res_c['hooke_loss']   + model.log_vars[5]) +

                    physics_w * (pl_arr[0] * res_l['pde_loss']    + model.log_vars[6]) +
                    physics_w * (pl_arr[1] * loss_dir_l            + model.log_vars[7]) +
                    physics_w * (pl_arr[2] * neu_l['neumann_loss'] + model.log_vars[8]) +
                               (pl_arr[3] * loss_data_l            + model.log_vars[9]) +
                               (pl_arr[4] * loss_data_sig_l        + model.log_vars[10]) +
                    physics_w * (pl_arr[5] * res_l['hooke_loss']   + model.log_vars[11])
                )

                loss_geo_scaled = loss_geo / n_geos
                scaler.scale(loss_geo_scaled).backward()
                loss_total += loss_geo_scaled.detach().item()

                log['pde_c'] += res_c['pde_loss'].item()    / n_geos
                log['pde_l'] += res_l['pde_loss'].item()    / n_geos
                log['dir_c'] += loss_dir_c.item()           / n_geos
                log['dir_l'] += loss_dir_l.item()           / n_geos
                log['dat_c'] += loss_data_c.item()          / n_geos
                log['dat_l'] += loss_data_l.item()          / n_geos
                log['sig_c'] += loss_data_sig_c.item()      / n_geos
                log['sig_l'] += loss_data_sig_l.item()      / n_geos
                log['vm_c']  += loss_vm_c.item()            / n_geos
                log['vm_l']  += loss_vm_l.item()            / n_geos
                log['hoo_c'] += res_c['hooke_loss'].item()  / n_geos
                log['hoo_l'] += res_l['hooke_loss'].item()  / n_geos

            # ── Optimizer step ─────────────────────────────────────────────────
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.log_vars[3].clamp_(max=0.1)
                model.log_vars[4].clamp_(max=0.1)

            # ── Timing ─────────────────────────────────────────────────────────
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_measurements.append(time.time() - t0)

            # ── Logging every 100 epochs ───────────────────────────────────────
            if epoch % 100 == 0 or epoch == start_epoch:
                avg_t = sum(time_measurements) / len(time_measurements)
                time_measurements = []
                print(f"\n[EPOCH {epoch}/{ADAM_EPOCHS}] {avg_t:.2f}s/epoch")
                print(f"  Total loss  : {loss_total:.6f}")
                print(f"  PDE   C/L   : {log['pde_c']:.6f} / {log['pde_l']:.6f}")
                print(f"  Dir   C/L   : {log['dir_c']:.6f} / {log['dir_l']:.6f}")
                print(f"  Data u C/L  : {log['dat_c']:.6f} / {log['dat_l']:.6f}")
                print(f"  Data σ C/L  : {log['sig_c']:.6f} / {log['sig_l']:.6f}")
                print(f"  Hooke C/L   : {log['hoo_c']:.6f} / {log['hoo_l']:.6f}")
                print(f"  VM loss C/L : {log['vm_c']:.6f} / {log['vm_l']:.6f}")

                if epoch % CHECKPOINT_FREQUENCY == 0:
                    torch.save({
                        'epoch'                       : epoch,
                        'model_state_dict'            : model.state_dict(),
                        'optimizer_physics_state_dict': optimizer.state_dict(),
                        'log_vars'                    : model.log_vars.data.clone(),
                    }, MODEL_SAVE_PATH)

            loss_history.append({
                'epoch'          : epoch,
                'loss_total'     : loss_total,
                'pde_comp'       : log['pde_c'], 'pde_lat'        : log['pde_l'],
                'dirichlet_comp' : log['dir_c'], 'dirichlet_lat'  : log['dir_l'],
                'data_u_comp'    : log['dat_c'], 'data_u_lat'     : log['dat_l'],
                'data_sigma_comp': log['sig_c'], 'data_sigma_lat' : log['sig_l'],
                'hooke_comp'     : log['hoo_c'], 'hooke_lat'      : log['hoo_l'],
                'physics_weight' : physics_w,
                **{f'w_{i}': torch.exp(-model.log_vars[i]).item() for i in range(12)},
                'epoch_time_s'   : time_measurements[-1] if time_measurements else 0.0,
            })

            if epoch % 50 == 0:
                pd.DataFrame(loss_history).to_csv(LOG_CSV_PATH, index=False)

        # ══════════════════════════════════════════════════════════════════════
        # POST-TRAINING: METRICS & FIGURES
        # ══════════════════════════════════════════════════════════════════════
        print("\n[METRICS] Computing final metrics...")
        model.eval()

        for i in range(n_geos):
            u_pred_c, s_pred_c = _predict_full(all_active_comp[i], all_latents_comp[i], model, 'comp')
            u_pred_l, s_pred_l = _predict_full(all_active_lat[i],  all_latents_lat[i],  model, 'lat')

            for mode, up, sp, sampler in [
                    ('comp', u_pred_c, s_pred_c, all_active_comp[i]),
                    ('lat',  u_pred_l, s_pred_l, all_active_lat[i])]:
                m = compute_global_metrics(up, sampler.all_u, sp,
                                           sampler.all_sigma, sampler.sigma_scale)
                print(f"[METRICS] Geo {i+1} | {mode.upper()}")
                for k, v in m.items():
                    print(f"   {k:15s}: {v}")

        model.train()


    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()

    finally:
        try:
            if loss_history:
                pd.DataFrame(loss_history).to_csv(LOG_CSV_PATH, index=False)
                print(f"[LOG] Training log saved: {LOG_CSV_PATH}")
            if scan_history:
                pd.DataFrame(scan_history).to_csv(SCAN_LOG_PATH, index=False)
                print(f"[LOG] Scan log saved: {SCAN_LOG_PATH}")
        except Exception as e:
            print(f"[WARN] Could not save logs: {e}")


if __name__ == '__main__':
    main()