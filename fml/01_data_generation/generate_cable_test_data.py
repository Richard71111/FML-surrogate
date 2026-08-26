"""
Generate a long-burst evaluation dataset from a Cable simulation .mat file
(MATLAB v7.3 / HDF5 format).

Data layout in the .mat file (h5py axes are MATLAB axes reversed):
    t_save  : (1,  Nt)                sample times [ms]
    phi     : (Nt, Ncell)             membrane potentials [mV]  → ctrl
    Icleft  : (Njunction, 2, Nt)      cleft currents [µA]       → QoI

Pipeline
--------
1. Load the .mat file with h5py.
2. Extract one cell pair (junction j ↔ cells j and j+1).
3. PCHIP-resample to config.dt [ms].
4. Cut long bursts: length = NUM_MEMORY + 1 + N_PRED steps.
5. Save to fml/data with qoi (N, 2, L) and ctrl (N, 2, L+1).

This .npz is directly consumable by evaluate.py  (set TEST_NPZ to its path).

QoI and Ctrl begin at the same time; Ctrl contains the extra target-time
sample required by the model. No gating state is loaded or saved.

Usage
-----
    python generate_cable_test_data.py --memlen 80
    python generate_cable_test_data.py --memlen 80 --junc 12 --npred 4000 --nbursts 5

Required argument (comes from config parser):
    --memlen  INT    History window length n_M (same value used during training)

Optional arguments:
    --mat     PATH   Optional explicit .mat path. By default it is constructed
                    from config.BCL, GJ_COUPLING, and MESH_IDX.
    --junc    INT    Junction index, 0-based  (default: middle junction)
    --npred   INT    Prediction steps at config.dt  (default: 4000 = 400 ms)
    --nbursts INT    Number of long bursts to cut  (default: 5)
    --ignore_start_ms  FLOAT  Skip this many ms at trajectory start
                              to avoid initial transients  (default: 200 ms)
    --seed    INT    RNG seed for burst-start positions  (default: 42)
"""

import os
import sys
import argparse

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator

# ── import Config from the project root (two levels up from this file) ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..')))
from config import Config

_CABLE_BCL_ROOT = (
    "/fs/ess/PAS1622/RichardSui/MultiMesh_Cable_Cleft/VaryBCL"
)


def default_cable_mat(config):
    """Build the cable MAT path from Config without affecting model naming."""
    coupling = str(config.GJ_COUPLING).lower()
    if coupling == "strong":
        gj_tag = "StrongGJ"
    elif coupling == "weak":
        gj_tag = "WeakGJ"
    else:
        raise ValueError(
            f"GJ_COUPLING must be 'strong' or 'weak', got "
            f"{config.GJ_COUPLING!r}")
    return os.path.join(
        _CABLE_BCL_ROOT,
        gj_tag,
        f"BCL{config.BCL}",
        f"FEMDATA_{config.MESH_IDX}mat_{gj_tag}_BCL_{config.BCL}.mat",
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_cable_mat(mat_path):
    """Load Cable v7.3 .mat and return time-first arrays.

    Returns
    -------
    t   : (Nt,)             times [ms]
    phi : (Nt, Ncell)       membrane potentials [mV]
    Ic  : (Njunction, 2, Nt) cleft currents [µA]
    """
    with h5py.File(mat_path, 'r') as f:
        t   = np.asarray(f['t_save']).reshape(-1)   # (Nt,)
        phi = np.asarray(f['phi'])                   # (Nt, Ncell) — already time-first
        Ic  = np.asarray(f['Icleft'])               # (Njunction, 2, Nt)
    return t, phi, Ic


def extract_pair(phi, Ic, junction_idx):
    """Extract the ctrl and QoI arrays for one cell pair.

    For junction j the two adjacent cells are j and j+1.

    Parameters
    ----------
    phi          : (Nt, Ncell)
    Ic           : (Njunction, 2, Nt)
    junction_idx : int, 0-based

    Returns
    -------
    ctrl : (Nt, 2)   phi_axial of cells  [j, j+1]
    qoi  : (Nt, 2)   Icleft sides        [0, 1]  at junction j
    """
    ctrl = phi[:, junction_idx : junction_idx + 2]   # (Nt, 2)
    qoi  = Ic[junction_idx, :, :].T                  # (2, Nt) → (Nt, 2)
    return ctrl, qoi


# ── Resample ──────────────────────────────────────────────────────────────────

def resample_to_dt(t_old, t_new, data_tf):
    """PCHIP interpolation of time-first data (Nt, Nchan) → (Nt_new, Nchan)."""
    if not np.all(np.diff(t_old) > 0):
        raise ValueError("t_old must be strictly increasing.")
    interp = PchipInterpolator(t_old, data_tf, axis=0, extrapolate=False)
    out = interp(t_new)
    if not np.isfinite(out).all():
        raise ValueError("Interpolation produced NaN/Inf — t_new may exceed t_old range.")
    return out


# ── Burst cutting ─────────────────────────────────────────────────────────────

def cut_bursts(ctrl_res, qoi_res, n_mem, n_pred, n_bursts, ignore_start,
               seed):
    """Sample `n_bursts` non-overlapping-start bursts from the resampled trajectory.

    Each burst has length  L = n_mem + 1 + n_pred.
    Returns ctrl with shape (n_bursts, Nctrl, L+1) and QoI with shape
    (n_bursts, 2, L).
    """
    L        = n_mem + 1 + n_pred
    Nt       = ctrl_res.shape[0]
    max_start = Nt - L - 1

    if max_start < ignore_start:
        raise ValueError(
            f"Trajectory too short after ignore_start:\n"
            f"  Nt={Nt}, burst_len={L}, ignore_start={ignore_start}\n"
            f"  Need Nt >= {L + ignore_start + 1}.  "
            f"Try reducing --npred or --ignore_start_ms.")

    rng    = np.random.default_rng(seed)
    starts = rng.integers(ignore_start, max_start + 1, size=n_bursts)

    ctrl_bursts = np.stack([
        ctrl_res[s:s + L + 1, :].T
        for s in starts
    ], axis=0)
    qoi_bursts  = np.stack([qoi_res[s:s+L,  :].T for s in starts], axis=0)  # (N, 2, L)

    return ctrl_bursts, qoi_bursts, starts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── argument parser -------------------------------------------------------
    # Note: Config already parses --memlen from sys.argv (required).
    # Additional arguments specific to this script:
    # allow_abbrev=False: this parser and config.py's parser both scan the same
    # sys.argv independently -- abbreviation matching between them can silently
    # misassign one script's flag to the other's differently-named-but-prefix-
    # matching flag (this bit evaluate.py's --model vs config.py's --model_type).
    ap = argparse.ArgumentParser(
        description="Generate long-burst cable evaluation data.",
        add_help=True, allow_abbrev=False)
    ap.add_argument("--mat",
                    default=None,
                    help="Explicit Cable .mat path (HDF5/v7.3). If omitted, "
                         "construct it from config.BCL/GJ_COUPLING/MESH_IDX.")
    ap.add_argument("--junc", type=int, default=None,
                    help="Junction index 0-based (default: middle junction)")
    ap.add_argument("--npred", type=int, default=40000,
                    help="Prediction steps at config.dt "
                         "(default: 4000 ≈ 400 ms at dt=0.1 ms)")
    ap.add_argument("--nbursts", type=int, default=30,
                    help="Number of long bursts to cut (default: 5)")
    ap.add_argument("--ignore_start_ms", type=float, default=200.0,
                    help="Skip this many ms at trajectory start "
                         "to avoid initial transients (default: 200 ms)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for burst-start positions (default: 42)")
    # parse_known_args so that --memlen (consumed by Config) is ignored here
    args, _ = ap.parse_known_args()

    # ── load config -----------------------------------------------------------
    cfg   = Config()
    n_mem = cfg.NUM_MEMORY
    dt    = cfg.dt           # [ms]
    mat_path = args.mat if args.mat is not None else default_cable_mat(cfg)

    # ── 1. Load .mat ----------------------------------------------------------
    if not os.path.exists(mat_path):
        sys.exit(
            f"[ERROR] Cable .mat file not found for BCL={cfg.BCL}:\n"
            f"  {mat_path}\n"
            f"Set --bcl to an available value or override with --mat PATH.")

    print("=" * 60)
    print("Cable → FML evaluation data generator")
    print("=" * 60)
    print(f"  BCL        : {cfg.BCL} ms")
    print(f"  .mat file  : {mat_path}")
    print(f"  config dt  : {dt} ms   n_M : {n_mem}")
    print("  Ctrl contract: aligned, one sample longer than QoI")

    t, phi, Ic = load_cable_mat(mat_path)
    Ncell     = phi.shape[1]
    Njunction = Ic.shape[0]
    dt_save   = float(t[1] - t[0])
    print(f"\nLoaded: Nt={len(t)}, Ncell={Ncell}, Njunction={Njunction}")
    print(f"  t range : {t[0]:.4f} → {t[-1]:.4f} ms   dt_save={dt_save:.4f} ms")

    # ── 2. Pick junction ------------------------------------------------------
    j = args.junc if args.junc is not None else Njunction // 2
    if not (0 <= j < Njunction):
        sys.exit(f"[ERROR] --junc={j} out of range [0, {Njunction-1}]")
    print(f"\nExtracting junction {j}  (cells {j} ↔ {j+1})")

    # ── Resolve output path early; skip heavy work if file already exists ──────
    out_name = (f"cable_test_FEM{cfg.MESH_IDX}_BCL{cfg.BCL}"
                f"_nm{n_mem}_nr{cfg.NUM_RECURRENT}_dt{cfg._DT_TAG}"
                f"_j{j}.npz")
    out_path = os.path.join(cfg.DATA_DIR, out_name)
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"\n[skip] file already exists:\n  {out_path}")
        print(f"OUT_PATH={out_path}", flush=True)
        return
    # ──────────────────────────────────────────────────────────────────────────

    ctrl_tf, qoi_tf = extract_pair(phi, Ic, j)   # (Nt, 2) each
    print(f"  voltage range: [{ctrl_tf.min():.2f}, {ctrl_tf.max():.2f}] mV")
    print(f"  ctrl shape : {ctrl_tf.shape}")
    print(f"  qoi range  : [{qoi_tf.min():.5f}, {qoi_tf.max():.5f}] µA")

    # ── 3. Resample to config dt ----------------------------------------------
    t_start = t[0]
    t_end   = t[-1]
    t_new   = np.arange(t_start, t_end + 1e-9, dt)
    t_new   = t_new[t_new <= t_end]                # guard against floating-point overshoot
    print(f"\nResampling {len(t)} → {len(t_new)} points (dt={dt} ms)")

    ctrl_res = resample_to_dt(t, t_new, ctrl_tf)   # (Nt_new, 2)
    qoi_res  = resample_to_dt(t, t_new, qoi_tf)    # (Nt_new, 2)

    # ── 4. Cut long bursts ----------------------------------------------------
    ignore_start = max(0, int(np.ceil(args.ignore_start_ms / dt)))
    burst_len    = n_mem + 1 + args.npred
    pred_ms      = args.npred * dt

    print(f"\nBurst config:")
    print(f"  n_mem           = {n_mem}  ({(n_mem+1)*dt:.1f} ms history window)")
    print(f"  n_pred          = {args.npred}  ({pred_ms:.1f} ms prediction)")
    print(f"  burst_len       = {burst_len} steps")
    print(f"  ignore_start    = {ignore_start} steps ({args.ignore_start_ms:.0f} ms)")
    print(f"  n_bursts        = {args.nbursts}")
    print(f"  Nt after resamp = {len(t_new)}")
    print(f"  max valid start = {len(t_new) - burst_len - 1}")

    ctrl_b, qoi_b, starts = cut_bursts(
        ctrl_res, qoi_res,
        n_mem=n_mem, n_pred=args.npred,
        n_bursts=args.nbursts,
        ignore_start=ignore_start,
        seed=args.seed,
    )

    print(f"\nBurst shapes:  qoi {qoi_b.shape}   ctrl {ctrl_b.shape}")
    print(f"Burst start times [ms]: "
          + "  ".join(f"{s*dt + t_new[0]:.1f}" for s in starts))

    # ── 5. Save --------------------------------------------------------------
    # out_path already computed above (cable_test_FEM{idx}_nm{nm}_nr{nr}_j{j}.npz)
    np.savez(out_path,
             qoi=qoi_b.astype(np.float64),
             ctrl=ctrl_b.astype(np.float64),
             dt_ms=np.array(dt, dtype=np.float64))

    print(f"\nSaved → {out_path}")
    print(f"OUT_PATH={out_path}", flush=True)
    print("=" * 60)


if __name__ == "__main__":
    main()
