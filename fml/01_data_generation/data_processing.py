"""
Helper functions for data processing, including
    1. Data matrix orientation      -- orient_data_matrix
    2. Data matrix interpolation    -- interpolate_data_matrix
    3. Burst index generation       -- generate_burst_index
    4. Burst extraction             -- extract_bursts
    5. Aligned QoI/Ctrl extraction    -- extract_qoi_ctrl_bursts
    6. Case-file listing            -- list_case_files          (added)
    7. Per-case .mat loading        -- load_case_mat            (added)

(5) and (6) support the voltage-clamp dataset, where the source is a FOLDER of
many per-case .mat files (one 2-cell junction per file) written by
`run_2CellVoltage_clamp_dataset.m` in -v7 format. Each file holds:
    t          (Nt,)               sample times [ms]
    phi_axial  (Ncell, Nt)         axial potentials   -> source / ctrl
    Icleft     (2, Njunction, Nt)  cleft current      -> QoI
"""

# Import libraries
import os
import sys
import glob
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat


def orient_data_matrix(data_matrix):
    """
    Orient data matrix to time first format,
    Definition:
        Orient 2D matrix to shape (Time, Ncells) axial membrane potential matrix
        Orient 3D matrix to shape (Time, Nfeatures, Ncells - 1) cleft current matrix
    :param data_matrix: the data matrix to be oriented
    """

    data_matrix = np.asarray(data_matrix)
    if data_matrix.ndim == 2:
        # Orient 2D matrix to shape (Time, Ncells) axial membrane potential matrix
        if data_matrix.shape[0] < data_matrix.shape[1]:
            data_matrix = data_matrix.T

    elif data_matrix.ndim == 3:
        shape           = data_matrix.shape
        time_axis       = int(np.argmax(shape))
        feature_axis    = int(np.argmin(shape))
        axes            = [0, 1, 2]
        remaining_axis  = [ax for ax in axes if ax not in [time_axis, feature_axis]][0]
        data_matrix     = np.transpose(data_matrix, (time_axis, feature_axis, remaining_axis))
    else:
        raise ValueError("Data matrix must be 2D or 3D.")
    return data_matrix


def interpolate_data_matrix(t_old, t_new, data_matrix, orient=True):
    """
    Interpolate data matrix to new time points using PCHIP interpolation.

    After orientation:
    2D data should be (T, Ncell)
    3D data should be (T, 2, Ncell)

    :param orient: if True, call orient_data_matrix first (heuristic). Set to
        False when the data is already time-first and you want to keep the
        feature/trajectory axis order exactly as given (needed for the
        single-junction clamp data, where the heuristic would mis-order axes).

    return: interpolated data matrix with shape (len(t_new), Ncell) for 2D data,
            or (len(t_new), 2, Ncell) for 3D data
    """

    t_old           = np.asarray(t_old).squeeze()
    t_new           = np.asarray(t_new).squeeze()
    is_extrapolate  = False
    # ===== Check time vectors =====
    if not np.isfinite(t_old).all():
        raise ValueError(
            f"t_old contains NaN/Inf. "
            f"nan={np.isnan(t_old).sum()}, inf={np.isinf(t_old).sum()}"
        )

    if not np.isfinite(t_new).all():
        raise ValueError(
            f"t_new contains NaN/Inf. "
            f"nan={np.isnan(t_new).sum()}, inf={np.isinf(t_new).sum()}"
        )

    if t_old.ndim != 1:
        raise ValueError(f"t_old must be 1D after squeeze, got shape {t_old.shape}")

    if t_new.ndim != 1:
        raise ValueError(f"t_new must be 1D after squeeze, got shape {t_new.shape}")

    if len(t_old) < 2:
        raise ValueError(f"t_old must contain at least 2 points, got len(t_old)={len(t_old)}")

    if len(t_new) < 1:
        raise ValueError(f"t_new must contain at least 1 point, got len(t_new)={len(t_new)}")

    if not np.all(np.diff(t_old) > 0):
        raise ValueError("t_old must be strictly increasing for PCHIP interpolation.")

    # ===== Orient data =====
    if orient:
        data_matrix = orient_data_matrix(data_matrix)
    else:
        data_matrix = np.asarray(data_matrix)

    # ===== Check raw data =====
    if not np.isfinite(data_matrix).all():
        raise ValueError(
            f"data_matrix contains NaN/Inf before interpolation. "
            f"shape={data_matrix.shape}, "
            f"nan={np.isnan(data_matrix).sum()}, "
            f"inf={np.isinf(data_matrix).sum()}"
        )

    if data_matrix.shape[0] != len(t_old):
        raise ValueError(
            f"Time dimension mismatch: data_matrix.shape[0] = {data_matrix.shape[0]}, "
            f"but len(t_old) = {len(t_old)}"
        )

    # ===== Prevent extrapolation NaN =====
    t_old_min = float(t_old[0])
    t_old_max = float(t_old[-1])
    t_new_min = float(np.min(t_new))
    t_new_max = float(np.max(t_new))

    tol = 1e-12

    if t_new_min < t_old_min - tol or t_new_max > t_old_max + tol:
        print(f"t_new is outside interpolation range. "
            f"t_old range = [{t_old_min}, {t_old_max}], "
            f"t_new range = [{t_new_min}, {t_new_max}]. "
            f"This would produce NaN when extrapolate=False. We need to set extrapolate = True"
            )
        is_extrapolate = True

    # Clean tiny floating-point overshoot at the boundaries
    t_new = np.clip(t_new, t_old_min, t_old_max)

    # ===== Interpolate =====
    interpolator = PchipInterpolator(
        t_old,
        data_matrix,
        axis=0,
        extrapolate=is_extrapolate
    )

    interpolated_matrix = interpolator(t_new)

    # ===== Check interpolated result =====
    if not np.isfinite(interpolated_matrix).all():
        raise ValueError(
            f"interpolated_matrix contains NaN/Inf after interpolation. "
            f"shape={interpolated_matrix.shape}, "
            f"nan={np.isnan(interpolated_matrix).sum()}, "
            f"inf={np.isinf(interpolated_matrix).sum()}, "
            f"t_old range = [{t_old_min}, {t_old_max}], "
            f"t_new range = [{t_new_min}, {t_new_max}]"
        )

    return interpolated_matrix


def generate_burst_index(trajectory_length, burst_length, burst_number, seed, ignore_start):
    """
    Randomly generate valid burst starting indices for one trajectory.

    A start index is valid only when the half-open interval
        [start, start + burst_length)
    stays inside the trajectory. If the requested burst is longer than the
    trajectory, or if no valid start remains after IGNORE_START, return an
    empty array so the caller can skip this trajectory/split cleanly.
    """
    trajectory_length = int(trajectory_length)
    burst_length = int(burst_length)
    burst_number = int(burst_number)
    ignore_start = max(0, int(ignore_start))

    if burst_number <= 0:
        return np.empty(0, dtype=np.int64)
    if burst_length <= 0:
        raise ValueError(f"burst_length must be positive, got {burst_length}.")
    if trajectory_length <= 0 or burst_length > trajectory_length:
        return np.empty(0, dtype=np.int64)

    max_start = trajectory_length - burst_length
    if max_start < ignore_start:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    burst_idx = rng.integers(ignore_start, max_start + 1,
                             size=burst_number, dtype=np.int64)

    # Defensive filter: Python slices use an exclusive end index, so
    # start + burst_length == trajectory_length is still valid.
    burst_idx = burst_idx[burst_idx + burst_length <= trajectory_length]
    return burst_idx


def extract_bursts(data_mat, start_indices, burst_len, traj_idx, channel_first=True):
    """
    Extract bursts from either phi_mat or Icleft_mat.

    Parameters
    ----------
    data_mat : np.ndarray
        If 2D:
            shape = (T, Ncell)
            Example: phi_mat

        If 3D:
            shape = (T, 2, Ncell-1)
            Example: Icleft_mat

    start_indices : array-like
        Start indices of bursts.
        shape = (Nburst,)

    burst_len : int
        Length of each burst.

    traj_idx : int
        Pair trajectory index.
        For phi_mat:
            use columns traj_idx and traj_idx+1
        For Icleft_mat:
            use cleft index traj_idx

    channel_first : bool
        If True:
            output shape = (Nburst, 2, burst_len)
        If False:
            output shape = (Nburst, burst_len, 2)

    Returns
    -------
    bursts : np.ndarray
        Extracted bursts.
    """

    data_mat = np.asarray(data_mat)
    start_indices = np.asarray(start_indices, dtype=np.int64)

    offsets = np.arange(burst_len, dtype=np.int64)

    # shape = (Nburst, burst_len)
    time_idx = start_indices[:, None] + offsets[None, :]

    if time_idx.max() >= data_mat.shape[0]:
        raise ValueError(
            f"Some bursts exceed time dimension. "
            f"max requested index = {time_idx.max()}, "
            f"data time length = {data_mat.shape[0]}"
        )

    if data_mat.ndim == 2:
        # phi_mat: shape = (T, Ncell)
        # Take adjacent cell pair: traj_idx and traj_idx+1
        if traj_idx + 1 >= data_mat.shape[1]:
            raise ValueError(
                f"traj_idx={traj_idx} is invalid for 2D data with "
                f"shape {data_mat.shape}. Need traj_idx+1 < Ncell."
            )

        # result shape: (Nburst, burst_len, 2)
        bursts = data_mat[time_idx, traj_idx:traj_idx + 2]

    elif data_mat.ndim == 3:
        # Icleft_mat: shape = (T, 2, Ncell-1)
        if traj_idx >= data_mat.shape[2]:
            raise ValueError(
                f"traj_idx={traj_idx} is invalid for 3D data with "
                f"shape {data_mat.shape}. Need traj_idx < Ncell-1."
            )

        # result shape: (Nburst, burst_len, 2)
        bursts = data_mat[time_idx, :, traj_idx]

    else:
        raise ValueError("data_mat must be either 2D or 3D.")

    if channel_first:
        # (Nburst, burst_len, 2) -> (Nburst, 2, burst_len)
        bursts = bursts.transpose(0, 2, 1)

    return bursts


def extract_qoi_ctrl_bursts(qoi_mat, ctrl_mat, start_indices,
                            burst_len, traj_idx, channel_first=True):
    """Extract the pipeline's single Ctrl-now data contract.

    QoI and Ctrl begin at the same raw time index.  QoI contains
    ``burst_len`` samples and Ctrl contains ``burst_len + 1`` samples, so the
    known target-time control is available at every recurrent prediction.
    """
    starts = np.asarray(start_indices, dtype=np.int64)
    ctrl_len = int(burst_len) + 1
    if starts.size and np.max(starts + ctrl_len - 1) >= ctrl_mat.shape[0]:
        raise ValueError(
            f"Ctrl burst exceeds source: max Ctrl index="
            f"{np.max(starts + ctrl_len - 1)}, "
            f"Ctrl time length={ctrl_mat.shape[0]}"
        )
    qoi = extract_bursts(
        qoi_mat, starts, burst_len, traj_idx, channel_first=channel_first
    )
    ctrl = extract_bursts(
        ctrl_mat, starts, ctrl_len, traj_idx,
        channel_first=channel_first
    )
    return qoi, ctrl


def dataset_metadata(dt_ms):
    """Metadata shared by generated NPZ files (no contract/gating fields)."""
    return {"dt_ms": np.array(float(dt_ms), dtype=np.float64)}


# ======================================================================
# Added helpers for the per-case voltage-clamp dataset (folder of .mat)
# ======================================================================
def list_case_files(folder, pattern="*.mat"):
    """Return direct child MAT files only (non-recursive).

    Dataset folders may contain auxiliary subdirectories such as an obsolete
    ``train/`` folder.  They are deliberately ignored: only regular files
    matching ``folder/pattern`` are accepted.
    """
    files = sorted(
        path for path in glob.glob(os.path.join(folder, pattern))
        if os.path.isfile(path)
    )
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in:\n  {folder}\n"
            f"Set Config.STRONG_PATH / WEAK_PATH to the MATLAB target_folder.")
    return files


def _move_time_first(arr):
    """Move the time axis (the longest axis, last in the MATLAB save layout) to
    the front, keeping the order of the remaining axes.

    phi_axial (Ncell, Nt)         -> (Nt, Ncell)
    Icleft    (2, Njunction, Nt)  -> (Nt, 2, Njunction)
    This is deterministic (time = argmax of the shape) and, unlike the argmin
    heuristic in orient_data_matrix, preserves the (sides, junction) order even
    when Njunction == 1.
    """
    arr = np.asarray(arr, dtype=np.float64)
    time_axis = int(np.argmax(arr.shape))
    return np.moveaxis(arr, time_axis, 0)


def load_case_mat(file_path, t_key="t", ctrl_key="phi_axial", qoi_key="Icleft"):
    """Load one per-case .mat (-v7) and return time-first arrays.

    Returns
    -------
    t    : (Nt,)             sample times
    phi  : (Nt, Ncell)       source / ctrl   (from phi_axial)
    I    : (Nt, 2, Njunction) QoI            (from Icleft)
    """
    try:
        m = loadmat(file_path)
    except NotImplementedError as e:
        raise NotImplementedError(
            f"{os.path.basename(file_path)} looks like MATLAB v7.3 (HDF5). "
            f"The clamp driver writes -v7; load v7.3 via h5py instead. ({e})")
    required_keys = [t_key, ctrl_key, qoi_key]
    for k in required_keys:
        if k not in m:
            raise KeyError(f"{os.path.basename(file_path)}: missing variable '{k}'")

    t = np.asarray(m[t_key], dtype=np.float64).reshape(-1)

    phi = _move_time_first(m[ctrl_key])       # (Nt, Ncell)
    I = np.asarray(m[qoi_key], dtype=np.float64)
    if I.ndim == 2:                            # (2, Nt) -> (Nt, 2, 1)
        I = _move_time_first(I)[:, :, None]
    else:                                      # (2, Nj, Nt) -> (Nt, 2, Nj)
        I = _move_time_first(I)

    if phi.shape[0] != t.shape[0] or I.shape[0] != t.shape[0]:
        raise ValueError(
            f"{os.path.basename(file_path)}: time length mismatch "
            f"t={t.shape[0]}, phi={phi.shape[0]}, I={I.shape[0]}")
    return t, phi, I
