"""
Training, validation, and test data generation for the ID-cleft surrogate.

Reads the flat FOLDER of per-case voltage-clamp .mat files (one two-cell
junction per file), interpolates each trajectory onto a fixed dt, and creates
train/validation windows. For the ``hybrid_one_cell`` dataset, 20 complete,
randomly selected raw trajectories are held out for test.

    QoI    = Icleft     -> npz key 'qoi'   (shape [N, 2, burst_len])
    source = phi_axial  -> npz key 'ctrl'  (shape [N, 2, burst_len + 1])

QoI and Ctrl start at the same time. Ctrl always has one extra sample, which
is the known target-time input used by the flow map. Gating states are not
part of the dataset.

Run:
    python generate_train_val_test_data.py --memlen 5
"""

import os
import sys
import time
import numpy as np

import data_processing as dp

# Optional burst-interval plot (skipped if plot.py is absent).
try:
    import plot
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, PROJECT_ROOT)

from config import Config


cfg = Config()

# ---- parameters from config ----
DT              = cfg.dt                      # resample step [ms]
IGNORE_START    = cfg.IGNORE_START
NUM_MEMORY      = cfg.NUM_MEMORY
BURST_LEN       = cfg.BURST_LEN               # train burst length
VAL_BURST_LEN   = cfg.VAL_BURST_LEN            # fixed validation rollout length
TEST_FULL_TRAJECTORY = bool(
    getattr(cfg, "TEST_FULL_TRAJECTORY", False)
)
TEST_BURST_LEN = (
    None
    if TEST_FULL_TRAJECTORY
    else NUM_MEMORY + 1 + int(cfg.T_TEST / cfg.dt)
)
TEST_NREC = (
    None
    if TEST_FULL_TRAJECTORY
    else TEST_BURST_LEN - NUM_MEMORY - 1
)
NUM_BURSTS      = cfg.NUM_BURSTS              # train bursts per trajectory
NUM_BURSTS_VAL  = cfg.NUM_VAL                 # val bursts per trajectory
NUM_BURSTS_TEST = cfg.NUM_TEST                # test bursts per trajectory
NUM_TEST_TRAJECTORIES = cfg.NUM_TEST_TRAJECTORIES
GJ_coupling     = cfg.GJ_COUPLING
source_dir      = cfg.SOURCE_DIR
Save_data       = True
max_show        = 100
SCALE_DATA      = getattr(cfg, "SCALE_DATA", True)
QOI_SCALE_FACTOR = np.float64(getattr(cfg, "QOI_SCALE_FACTOR", 1000.0))
SCALING_INFO_FILE = getattr(
    cfg,
    "SCALING_INFO_FILE",
    os.path.join(
        cfg.DATA_DIR,
        f"scaling_info_{cfg.MODEL_NAME_BASE}.npz"
    ),
)

WINDOW_SPLITS = [
    ("train", BURST_LEN,      NUM_BURSTS),
    ("val",   VAL_BURST_LEN,  NUM_BURSTS_VAL),
]
if not TEST_FULL_TRAJECTORY:
    WINDOW_SPLITS.append(("test", TEST_BURST_LEN, NUM_BURSTS_TEST))
SPLIT_NAMES = ("train", "val", "test")


def main():
    print("=" * 60)
    print("Generating train/val/test data (ID-cleft voltage-clamp folder)...")
    t_data_start = time.time()

    files = dp.list_case_files(source_dir, cfg.FILE_GLOB)
    num_files = len(files)

    print("Config summary:")
    print(f"    source dir       : {source_dir}")
    print(f"    .mat files       : {num_files}")
    print(f"    resample dt      : {DT}")
    print(f"    burst len train  : {BURST_LEN} (nM={cfg.NUM_MEMORY}, nR={cfg.NUM_RECURRENT})")
    print(f"    burst len val    : {VAL_BURST_LEN} "
          f"(nR_val={cfg.NUM_VAL_RECURRENT}, "
          f"{cfg.NUM_VAL_RECURRENT * DT:g} ms)")
    if TEST_FULL_TRAJECTORY:
        print("    test contract    : complete held-out trajectories")
    else:
        print(f"    burst len test   : {TEST_BURST_LEN} "
              f"(nR_test={TEST_NREC}, {cfg.T_TEST} ms)")
    print(f"    bursts/traj      : train {NUM_BURSTS} | "
          f"val {NUM_BURSTS_VAL}")
    print(f"    test trajectories: {min(NUM_TEST_TRAJECTORIES, num_files)} selected files (seed={cfg.DATA_SEED})")
    print(f"    GJ coupling      : {GJ_coupling}")
    print(f"    ctrl channels    : {cfg.NUM_CTRL}")
    print("    Ctrl contract    : aligned, one sample longer than QoI")
    print(f"    scale data       : {SCALE_DATA}")
    if SCALE_DATA:
        print(f"    qoi scale factor : {QOI_SCALE_FACTOR:g}")
        print("    ctrl scale factor: 1 (unchanged)")
    print("=" * 60)

    # The voltage-clamp dataset contract is one trajectory per direct child
    # MAT file. The selection is random but reproducible. For the hybrid
    # dataset these files are held out from train/validation and retained in
    # their entirety for test.
    test_selection_rng = np.random.default_rng(cfg.DATA_SEED + 1)
    n_test_files = min(int(NUM_TEST_TRAJECTORIES), num_files)
    selected_test_file_indices = set(
        test_selection_rng.choice(num_files, size=n_test_files, replace=False).tolist()
    )

    # Master RNG -> a (train, val, test) seed triple per trajectory.
    master = np.random.default_rng(cfg.DATA_SEED)

    buckets = {name: {"phi": [], "I": []} for name in SPLIT_NAMES}
    skipped = {name: 0 for name in SPLIT_NAMES}
    last_idx = {}            # for the optional plot
    t0 = time.time()
    n_traj = 0
    n_test_traj = 0
    complete_test_burst_len = None

    for f_idx, fpath in enumerate(files):
        # ---- load + interpolate this file (one trajectory) ----
        try:
            t_old, phi_tf, I_tf = dp.load_case_mat(
                fpath, cfg.T_VAR, cfg.CTRL_VAR, cfg.QOI_VAR)
            # Uniform grid that stays within [t0, t_end] (no spurious extrapolation).
            n_grid = int(np.floor((t_old[-1] - t_old[0]) / DT + 1e-9)) + 1
            t_new = t_old[0] + DT * np.arange(n_grid)
            phi_i = dp.interpolate_data_matrix(t_old, t_new, phi_tf, orient=False)  # (L, Ncell)
            I_i = dp.interpolate_data_matrix(t_old, t_new, I_tf, orient=False)      # (L, 2, Nj)
        except Exception as e:
            print(f"  [skip file] {os.path.basename(fpath)}: {e}")
            continue

        L = len(t_new)
        num_j = I_i.shape[2]

        for j in range(num_j):
            n_traj += 1

            if TEST_FULL_TRAJECTORY and f_idx in selected_test_file_indices:
                # Keep the entire usable interval. The Ctrl-now contract
                # requires one more Ctrl sample than QoI, hence QoI uses
                # samples [0, L-2] and Ctrl uses [0, L-1].
                full_blen = L - 1
                if full_blen <= NUM_MEMORY + 1:
                    skipped["test"] += 1
                    continue
                if complete_test_burst_len is None:
                    complete_test_burst_len = full_blen
                elif full_blen != complete_test_burst_len:
                    raise ValueError(
                        "Complete test trajectories have inconsistent "
                        f"resampled lengths: {full_blen} vs "
                        f"{complete_test_burst_len}."
                    )
                idx = np.array([0], dtype=np.int64)
                I_b, phi_b = dp.extract_qoi_ctrl_bursts(
                    I_i, phi_i, idx, full_blen, j, channel_first=True
                )
                buckets["test"]["phi"].append(phi_b)
                buckets["test"]["I"].append(I_b)
                last_idx["test"] = idx
                n_test_traj += 1
                continue

            seeds = {
                name: int(seed)
                for (name, _, _), seed in zip(
                    WINDOW_SPLITS,
                    master.integers(0, 2**31, size=len(WINDOW_SPLITS))
                )
            }
            for name, blen, bnum in WINDOW_SPLITS:
                if name == "test" and (
                    f_idx not in selected_test_file_indices
                    or n_test_traj >= NUM_TEST_TRAJECTORIES
                ):
                    continue
                # ---- feasibility: enough samples for this burst length? ----
                source_len = blen + 1
                if L - source_len < IGNORE_START:
                    skipped[name] += 1
                    continue
                idx = dp.generate_burst_index(
                    L, source_len, bnum, seeds[name], IGNORE_START
                )
                I_b, phi_b = dp.extract_qoi_ctrl_bursts(
                    I_i, phi_i, idx, blen, j, channel_first=True
                )
                buckets[name]["phi"].append(phi_b)
                buckets[name]["I"].append(I_b)
                last_idx[name] = idx
                if name == "test":
                    n_test_traj += 1

        if (f_idx + 1) % 50 == 0 or f_idx == num_files - 1:
            print(f"  processed {f_idx + 1}/{num_files} files "
                  f"({time.time() - t0:.1f}s)")

    print(f"Loaded + bursted {n_traj} trajectories in {time.time() - t_data_start:.1f}s")
    print(f"Test split used {n_test_traj} trajectories (requested {NUM_TEST_TRAJECTORIES}).")
    if TEST_FULL_TRAJECTORY and n_test_traj != n_test_files:
        raise RuntimeError(
            f"Expected {n_test_files} complete test trajectories, produced "
            f"{n_test_traj}. Check unreadable MAT files or junction count."
        )
    for name in skipped:
        if skipped[name]:
            print(f"  NOTE: {skipped[name]} trajectories too short for '{name}' "
                  f"burst (dropped from that split).")

    # ---- concatenate, cast, save ----
    out = {}
    for name in SPLIT_NAMES:
        if not buckets[name]["I"]:
            raise ValueError(
                f"No '{name}' bursts produced — every trajectory was shorter than "
                f"its burst length. Lower NUM_MEMORY or the corresponding "
                f"recurrent-step count.")
        I = np.concatenate(buckets[name]["I"], axis=0).astype(np.float64)
        phi = np.concatenate(buckets[name]["phi"], axis=0).astype(np.float64)
        out[name] = (I, phi)
        print(f"  {name:5s}: qoi {I.shape}  ctrl {phi.shape}")

    if TEST_FULL_TRAJECTORY:
        TEST_BURST_LEN_ACTUAL = out["test"][0].shape[-1]
        TEST_NREC_ACTUAL = TEST_BURST_LEN_ACTUAL - NUM_MEMORY - 1
    else:
        TEST_BURST_LEN_ACTUAL = TEST_BURST_LEN
        TEST_NREC_ACTUAL = TEST_NREC
    if TEST_NREC_ACTUAL < 1:
        raise ValueError(
            "Test trajectory is too short for the requested NUM_MEMORY."
        )

    scaling_payload = {
        **dp.dataset_metadata(DT),
        "normalized": np.array(False),
        "normalization_mode": np.array("none"),
        "scaled": np.array(False),
        "scaling_mode": np.array("none"),
        "qoi_scale_factor": np.array(1.0, dtype=np.float64),
        "ctrl_scale_factor": np.array(1.0, dtype=np.float64),
    }
    if SCALE_DATA:
        for name, (I, phi) in out.items():
            out[name] = (
                (I * QOI_SCALE_FACTOR).astype(np.float64, copy=False),
                phi,
            )

        scaling_payload = {
            **dp.dataset_metadata(DT),
            "normalized": np.array(False),
            "normalization_mode": np.array("none"),
            "scaled": np.array(True),
            "scaling_mode": np.array("qoi_multiply_constant"),
            "qoi_scale_factor": np.array(QOI_SCALE_FACTOR, dtype=np.float64),
            "ctrl_scale_factor": np.array(1.0, dtype=np.float64),
        }
        print(f"Applied scaling: qoi *= {QOI_SCALE_FACTOR:g}; ctrl unchanged")

    if Save_data:
        np.savez(cfg.TRAIN_FILE, qoi=out["train"][0], ctrl=out["train"][1],
                 num_memory=np.array(cfg.NUM_MEMORY),
                 num_recurrent=np.array(cfg.NUM_RECURRENT),
                 **scaling_payload)
        np.savez(cfg.VAL_FILE,   qoi=out["val"][0],   ctrl=out["val"][1],
                 num_memory=np.array(cfg.NUM_MEMORY),
                 num_recurrent=np.array(cfg.NUM_VAL_RECURRENT),
                 **scaling_payload)
        np.savez(cfg.TEST_FILE,  qoi=out["test"][0],  ctrl=out["test"][1],
                 num_memory=np.array(cfg.NUM_MEMORY),
                 num_recurrent=np.array(TEST_NREC_ACTUAL),
                 num_source_trajectories=np.array(n_test_traj),
                 complete_trajectory=np.array(TEST_FULL_TRAJECTORY),
                 **scaling_payload)
        np.savez(SCALING_INFO_FILE, **scaling_payload)
        print("Saved:")
        print(f"  {cfg.TRAIN_FILE}")
        print(f"  {cfg.VAL_FILE}")
        print(f"  {cfg.TEST_FILE}")
        print(f"  {SCALING_INFO_FILE}")

    if _HAS_PLOT and last_idx.get("train") is not None:
        try:
            plot.plot_burst_intervals(
                burst_idx_train=last_idx.get("train"),
                burst_idx_val=last_idx.get("val"),
                burst_idx_test=last_idx.get("test"),
                train_len=BURST_LEN, val_len=VAL_BURST_LEN,
                test_len=TEST_BURST_LEN_ACTUAL,
                trajectory_length=len(t_new), max_show=max_show,
                title=f"burst interval check (last trajectory, first {max_show})",
                save_path=cfg.EVAL_DIR)
        except Exception as e:
            print(f"  [plot skipped] {e}")


if __name__ == "__main__":
    main()
