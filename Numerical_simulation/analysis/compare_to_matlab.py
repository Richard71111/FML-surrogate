#!/usr/bin/env python3
"""Compare one Numerical_simulation result with a MATLAB/FEM .mat file.

The simulation directory must contain ``summary.json`` and ``simulation.npz``.
Both trajectories are interpolated in time onto either the MATLAB output times
(the default, which avoids inventing extra reference information) or a user
requested uniform grid. Results are written into a new subdirectory of the
simulation directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-fml-compare")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def _load_reference_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load the trajectory fields from classic or MATLAB v7.3 MAT files."""
    with path.open("rb") as handle:
        header = handle.read(128)
    # MATLAB v7.3 commonly reserves a 512-byte HDF5 user block, so the HDF5
    # magic bytes are not necessarily at byte zero.  The MATLAB header is the
    # reliable discriminator in that case.
    is_hdf5 = (
        header.startswith(b"\x89HDF\r\n\x1a\n")
        or b"MATLAB 7.3 MAT-file" in header
    )

    if is_hdf5:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "MATLAB v7.3 reference files require h5py in the runtime environment."
            ) from exc

        with h5py.File(path, "r") as handle:
            aliases = {
                "time": ("time", "t_save"),
                "phi_axial": ("phi_axial", "phi"),
                "Icleft": ("Icleft",),
            }
            selected: dict[str, str] = {}
            for logical_name, candidates in aliases.items():
                selected_name = next(
                    (candidate for candidate in candidates if candidate in handle), None
                )
                if selected_name is None:
                    raise KeyError(
                        f"MATLAB v7.3 file is missing {logical_name}; "
                        f"accepted variables are {candidates}."
                    )
                selected[logical_name] = selected_name
            time = np.asarray(handle[selected["time"]], dtype=float)
            voltage = np.asarray(handle[selected["phi_axial"]], dtype=float)
            icleft = np.asarray(handle[selected["Icleft"]], dtype=float)
        return time, voltage, icleft, "matlab_v7.3_hdf5"

    mat = loadmat(path)
    aliases = {
        "time": ("time", "t_save"),
        "phi_axial": ("phi_axial", "phi"),
        "Icleft": ("Icleft",),
    }
    selected: dict[str, str] = {}
    for logical_name, candidates in aliases.items():
        selected_name = next((candidate for candidate in candidates if candidate in mat), None)
        if selected_name is None:
            raise KeyError(
                f"MATLAB file is missing {logical_name}; "
                f"accepted variables are {candidates}."
            )
        selected[logical_name] = selected_name
    return (
        np.asarray(mat[selected["time"]], dtype=float),
        np.asarray(mat[selected["phi_axial"]], dtype=float),
        np.asarray(mat[selected["Icleft"]], dtype=float),
        "classic_mat",
    )


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return number


def _parse_indices(value: str | None, size: int, count: int) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=int)
    if value is None:
        return np.unique(np.linspace(0, size - 1, min(count, size), dtype=int))
    try:
        indices = np.asarray([int(item.strip()) for item in value.split(",")], dtype=int)
    except ValueError as exc:
        raise ValueError("indices must be comma-separated integers") from exc
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= size):
        raise ValueError(f"indices must lie in [0, {size - 1}]")
    return np.unique(indices)


def _as_time_major(array: np.ndarray, n_time: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float).squeeze()
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional after squeeze; got {value.shape}")
    if value.shape[0] == n_time:
        return value
    if value.shape[1] == n_time:
        return value.T
    raise ValueError(f"{name} shape {value.shape} has no time axis of length {n_time}")


def _matlab_icleft_time_major(array: np.ndarray, n_time: int) -> np.ndarray:
    value = np.asarray(array, dtype=float).squeeze()
    if value.ndim != 3:
        raise ValueError(f"MATLAB Icleft must be 3-D; got {value.shape}")
    time_axes = [axis for axis, length in enumerate(value.shape) if length == n_time]
    side_axes = [axis for axis, length in enumerate(value.shape) if length == 2]
    if not time_axes or not side_axes:
        raise ValueError(
            f"Cannot identify time/side axes in MATLAB Icleft shape {value.shape}."
        )
    time_axis = time_axes[-1]
    side_axis = next((axis for axis in side_axes if axis != time_axis), None)
    if side_axis is None:
        raise ValueError(f"Cannot identify the two-side axis in Icleft shape {value.shape}.")
    junction_axis = next(axis for axis in range(3) if axis not in (time_axis, side_axis))
    return np.transpose(value, (time_axis, junction_axis, side_axis))


def _strictly_increasing(t: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t = np.asarray(t, dtype=float).reshape(-1)[order]
    y = np.asarray(y, dtype=float)[order]
    unique_t, unique_index = np.unique(t, return_index=True)
    y = y[unique_index]
    if unique_t.size < 2 or np.any(np.diff(unique_t) <= 0.0):
        raise ValueError(f"{name} needs at least two unique increasing time samples.")
    if not np.isfinite(unique_t).all() or not np.isfinite(y).all():
        raise ValueError(f"{name} contains NaN/Inf values.")
    return unique_t, y


def _interp(t_old: np.ndarray, y_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    flat = y_old.reshape(y_old.shape[0], -1)
    out = np.empty((t_new.size, flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(t_new, t_old, flat[:, column])
    return out.reshape((t_new.size,) + y_old.shape[1:])


def _map_indices(n_sim: int, n_ref: int, mode: str) -> np.ndarray:
    if mode == "index":
        if n_sim > n_ref:
            raise ValueError(
                f"index spatial matching needs reference size >= simulation size; "
                f"got {n_ref} < {n_sim}. Use --spatial-match normalized."
            )
        return np.arange(n_sim, dtype=int)
    if n_sim == 1:
        return np.zeros(1, dtype=int)
    return np.rint(np.linspace(0, n_ref - 1, n_sim)).astype(int)


def _metrics(sim: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    error = sim - ref
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
        "relative_l2": float(np.linalg.norm(error.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-30)),
    }


def _activation_times(
    grid: np.ndarray, voltage: np.ndarray, threshold_mv: float
) -> np.ndarray:
    """First-upstroke activation time [ms] per cell, or NaN if it never fires.

    The activation time is the first upward crossing of ``threshold_mv``,
    linearly interpolated between the two bracketing samples so the estimate is
    not limited to the comparison-grid spacing.
    """
    n_cells = voltage.shape[1]
    times = np.full(n_cells, np.nan, dtype=float)
    for cell in range(n_cells):
        trace = voltage[:, cell]
        above = trace >= threshold_mv
        if above[0] or not above.any():
            continue  # already above at t0 (no clean upstroke) or never fires
        index = int(np.argmax(above))  # first sample at/above threshold
        v0, v1 = trace[index - 1], trace[index]
        t0, t1 = grid[index - 1], grid[index]
        times[cell] = t1 if v1 == v0 else t0 + (threshold_mv - v0) * (t1 - t0) / (v1 - v0)
    return times


def _conduction_velocity(
    grid: np.ndarray,
    voltage: np.ndarray,
    cell_length_um: float,
    threshold_mv: float,
    start_cell: int,
    end_cell: int,
) -> dict[str, object]:
    """Two-point CV from first threshold crossings at selected cell indices.

    Cell indices are zero-based.  The crossing times are linearly interpolated
    between native bracketing samples, then

        CV = (end_cell-start_cell)*cell_length / (t_end-t_start).
    """
    n_cells = int(voltage.shape[1])
    if not (0 <= start_cell < end_cell < n_cells):
        raise ValueError(
            "CV cells must satisfy 0 <= start < end < n_cells; "
            f"got start={start_cell}, end={end_cell}, n_cells={n_cells}."
        )
    used_threshold = float(threshold_mv)
    activation = _activation_times(grid, voltage, used_threshold)
    start_time = activation[start_cell]
    end_time = activation[end_cell]
    distance_um = float(end_cell - start_cell) * float(cell_length_um)
    distance_cm = distance_um * 1.0e-4
    result: dict[str, object] = {
        "cv_cm_s": None,
        "method": "two_point_first_upward_threshold_crossing",
        "start_cell": int(start_cell),
        "end_cell": int(end_cell),
        "start_activation_time_ms": (
            None if not np.isfinite(start_time) else float(start_time)
        ),
        "end_activation_time_ms": (
            None if not np.isfinite(end_time) else float(end_time)
        ),
        "travel_time_ms": None,
        "distance_um": distance_um,
        "distance_cm": distance_cm,
        "n_cells_total": n_cells,
        "cell_length_um": float(cell_length_um),
        "threshold_mv": used_threshold,
        "threshold_kind": "absolute",
        "activation_time_ms": [
            None if not np.isfinite(t) else float(t) for t in activation
        ],
    }
    if not np.isfinite(start_time) or not np.isfinite(end_time):
        missing = []
        if not np.isfinite(start_time):
            missing.append(str(start_cell))
        if not np.isfinite(end_time):
            missing.append(str(end_cell))
        result["note"] = (
            f"cell(s) {','.join(missing)} never crossed {used_threshold:g} mV"
        )
        return result
    travel_time_ms = float(end_time - start_time)
    result["travel_time_ms"] = travel_time_ms
    if travel_time_ms <= 0.0:
        result["note"] = "non-positive travel time; propagation is not left-to-right"
        return result
    result["cv_cm_s"] = float(1.0e3 * distance_cm / travel_time_ms)
    return result


def _last_beat_trajectory(
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    bcl_ms: float,
    nbeats: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the final pacing interval on the trajectory's native grid.

    Exact points at the BCL boundaries are inserted by linear interpolation
    when they are absent from the saved trajectory.  Interior samples remain
    on the native simulation/MATLAB grid, so threshold-crossing precision is
    not reduced to the common plotting grid.
    """
    if bcl_ms <= 0.0 or nbeats <= 0:
        raise ValueError(
            f"{name} needs positive BCL/nbeats for last-beat CV; "
            f"got BCL={bcl_ms}, nbeats={nbeats}."
        )
    beat_start = float(nbeats - 1) * float(bcl_ms)
    beat_end = float(nbeats) * float(bcl_ms)
    tolerance = 1.0e-9
    if time_ms[0] > beat_start + tolerance or time_ms[-1] < beat_end - tolerance:
        raise ValueError(
            f"{name} does not cover its last beat [{beat_start:g}, "
            f"{beat_end:g}] ms; available range is "
            f"[{time_ms[0]:g}, {time_ms[-1]:g}] ms."
        )

    interior = (time_ms > beat_start) & (time_ms < beat_end)
    beat_time = np.concatenate(
        ([beat_start], time_ms[interior], [beat_end])
    )
    beat_voltage = _interp(time_ms, voltage_mv, beat_time)
    return beat_time, beat_voltage, beat_start, beat_end


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=250, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_overlays(
    grid: np.ndarray,
    sim: np.ndarray,
    ref: np.ndarray,
    selected: np.ndarray,
    mapped: np.ndarray,
    ylabel: str,
    title: str,
    path: Path,
    sim_prefix: str,
    ref_prefix: str,
) -> None:
    fig, axes = plt.subplots(
        len(selected), 1, figsize=(10.5, max(3.2, 2.15 * len(selected))),
        sharex=True, squeeze=False,
    )
    for row, sim_index in enumerate(selected):
        ax = axes[row, 0]
        ref_index = int(mapped[sim_index])
        ax.plot(grid, ref[:, sim_index], color="black", lw=1.25,
                label=f"{ref_prefix} {ref_index}")
        ax.plot(grid, sim[:, sim_index], color="#d95f02", lw=1.0, ls="--",
                label=f"{sim_prefix} {sim_index}")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_ylabel(ylabel)
    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel("time [ms]")
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_error_heatmap(
    grid: np.ndarray,
    error: np.ndarray,
    ylabel: str,
    color_label: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    limit = float(np.max(np.abs(error)))
    if limit == 0.0:
        limit = 1.0
    image = ax.imshow(
        error.T,
        origin="lower",
        aspect="auto",
        extent=(float(grid[0]), float(grid[-1]), 0, error.shape[1] - 1),
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set(xlabel="time [ms]", ylabel=ylabel, title=title)
    fig.colorbar(image, ax=ax, label=color_label)
    fig.tight_layout()
    _save_figure(fig, path)


def _filename_metadata(path: Path) -> dict[str, int]:
    match = re.search(r"_N(?P<n>\d+)_BCL(?P<bcl>\d+)_nb(?P<nb>\d+)", path.stem)
    if match:
        return {key: int(value) for key, value in match.groupdict().items()}
    # MultiMesh_Cable_Cleft names use, for example,
    # FEMDATA_10mat_StrongGJ_BCL_200.mat and do not encode the beat count.
    match = re.search(r"_BCL_?(?P<bcl>\d+)(?:_|$)", path.stem)
    return {"bcl": int(match.group("bcl"))} if match else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpolate and compare Numerical_simulation output with MATLAB/FEM data."
    )
    parser.add_argument("simulation_dir", type=Path,
                        help="Folder containing summary.json and simulation.npz.")
    parser.add_argument("reference_mat", type=Path,
                        help="MATLAB .mat file containing time, phi_axial, and Icleft.")
    parser.add_argument(
        "--output-name", default=None,
        help="Output subfolder name (default: comparison_to_<MAT-file-stem>).",
    )
    parser.add_argument(
        "--grid-dt-ms", type=_positive_float, default=None,
        help="Uniform comparison-grid spacing. Default uses MATLAB reference times.",
    )
    parser.add_argument("--t-start-ms", type=float, default=None)
    parser.add_argument("--t-end-ms", type=float, default=None)
    parser.add_argument(
        "--spatial-match", choices=("normalized", "index"), default="normalized",
        help="Map simulation to reference indices by normalized cable position "
             "(default) or identical integer index.",
    )
    parser.add_argument("--cells", default=None,
                        help="Simulation cell indices to plot, comma-separated.")
    parser.add_argument("--junctions", default=None,
                        help="Simulation junction indices to plot, comma-separated.")
    parser.add_argument("--plot-count", type=int, default=5,
                        help="Number of evenly spaced traces when indices are omitted.")
    parser.add_argument(
        "--cell-length-um", type=_positive_float, default=100.0,
        help="Cell pitch in microns for two-point CV (default 100 um; assumed "
             "identical for simulation and reference).",
    )
    parser.add_argument(
        "--cv-threshold-mv", type=float, default=10.0,
        help="Absolute voltage for first upward crossing (default 10 mV).",
    )
    parser.add_argument(
        "--cv-cell-start", type=int, default=10,
        help="Zero-based upstream cell index for two-point CV (default 10).",
    )
    parser.add_argument(
        "--cv-cell-end", type=int, default=40,
        help="Zero-based downstream cell index for two-point CV (default 40).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation_dir = args.simulation_dir.expanduser().resolve()
    reference_path = args.reference_mat.expanduser().resolve()
    summary_path = simulation_dir / "summary.json"
    simulation_path = simulation_dir / "simulation.npz"
    for path in (summary_path, simulation_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text())
    sim_npz = np.load(simulation_path)
    sim_t_v = np.asarray(sim_npz["time_ms"], dtype=float).reshape(-1)
    sim_v = np.asarray(sim_npz["voltage_mv"], dtype=float)
    sim_i = np.asarray(sim_npz["cleft_current_ua"], dtype=float)
    if "cleft_time_ms" in sim_npz.files and len(sim_npz["cleft_time_ms"]) == len(sim_i):
        sim_t_i = np.asarray(sim_npz["cleft_time_ms"], dtype=float).reshape(-1)
    elif "current_time_ms" in sim_npz.files and len(sim_npz["current_time_ms"]) == len(sim_i):
        sim_t_i = np.asarray(sim_npz["current_time_ms"], dtype=float).reshape(-1)
    elif len(sim_t_v) == len(sim_i):
        sim_t_i = sim_t_v.copy()
    else:
        raise ValueError("Cannot match cleft_current_ua to a simulation time array.")
    if sim_v.ndim != 2 or sim_i.ndim != 3 or sim_i.shape[2] != 2:
        raise ValueError(
            f"Unexpected simulation shapes: voltage={sim_v.shape}, Icleft={sim_i.shape}."
        )
    sim_t_v, sim_v = _strictly_increasing(sim_t_v, sim_v, "simulation voltage")
    sim_t_i, sim_i = _strictly_increasing(sim_t_i, sim_i, "simulation Icleft")

    ref_t_raw, ref_v_raw, ref_i_raw, reference_format = _load_reference_arrays(
        reference_path
    )
    ref_t = np.asarray(ref_t_raw, dtype=float).reshape(-1)
    ref_v = _as_time_major(ref_v_raw, ref_t.size, "phi_axial")
    ref_i = _matlab_icleft_time_major(ref_i_raw, ref_t.size)
    ref_t, ref_v = _strictly_increasing(ref_t, ref_v, "MATLAB voltage")
    ref_t_i, ref_i = _strictly_increasing(
        np.asarray(ref_t_raw, dtype=float).reshape(-1), ref_i, "MATLAB Icleft"
    )
    if not np.array_equal(ref_t, ref_t_i):
        raise ValueError("MATLAB voltage and Icleft time axes differ after cleanup.")

    start = max(float(sim_t_v[0]), float(sim_t_i[0]), float(ref_t[0]))
    end = min(float(sim_t_v[-1]), float(sim_t_i[-1]), float(ref_t[-1]))
    if args.t_start_ms is not None:
        start = max(start, float(args.t_start_ms))
    if args.t_end_ms is not None:
        end = min(end, float(args.t_end_ms))
    if end <= start:
        raise ValueError(f"No common time interval: start={start}, end={end} ms.")
    if args.grid_dt_ms is None:
        grid = ref_t[(ref_t >= start) & (ref_t <= end)]
        grid_kind = "matlab_reference_times"
    else:
        count = int(np.floor((end - start) / args.grid_dt_ms + 1.0e-10))
        grid = start + np.arange(count + 1, dtype=float) * args.grid_dt_ms
        if grid[-1] < end - 1.0e-9:
            grid = np.append(grid, end)
        grid_kind = "uniform"
    if grid.size < 2:
        raise ValueError("The common comparison grid has fewer than two samples.")

    n_sim_cells, n_ref_cells = sim_v.shape[1], ref_v.shape[1]
    n_sim_junc, n_ref_junc = sim_i.shape[1], ref_i.shape[1]
    cell_map = _map_indices(n_sim_cells, n_ref_cells, args.spatial_match)
    junc_map = _map_indices(n_sim_junc, n_ref_junc, args.spatial_match)
    cells = _parse_indices(args.cells, n_sim_cells, args.plot_count)
    junctions = _parse_indices(args.junctions, n_sim_junc, args.plot_count)

    sim_v_grid = _interp(sim_t_v, sim_v, grid)
    ref_v_grid_full = _interp(ref_t, ref_v, grid)
    ref_v_grid = ref_v_grid_full[:, cell_map]
    sim_i_grid = _interp(sim_t_i, sim_i, grid)
    ref_i_grid_full = _interp(ref_t, ref_i, grid)
    ref_i_grid = ref_i_grid_full[:, junc_map, :]

    warnings: list[str] = []
    metadata = _filename_metadata(reference_path)
    if n_sim_cells != n_ref_cells:
        warnings.append(
            f"Cell-count mismatch: simulation={n_sim_cells}, reference={n_ref_cells}; "
            f"used spatial_match={args.spatial_match}."
        )
    if n_sim_junc != n_ref_junc:
        warnings.append(
            f"Junction-count mismatch: simulation={n_sim_junc}, reference={n_ref_junc}; "
            f"used spatial_match={args.spatial_match}."
        )
    if "bcl" in metadata and "bcl_ms" in summary:
        if not np.isclose(float(summary["bcl_ms"]), metadata["bcl"]):
            warnings.append(
                f"BCL mismatch: simulation={summary['bcl_ms']} ms, "
                f"reference filename={metadata['bcl']} ms."
            )
    if "nb" in metadata and "nbeats" in summary:
        if int(summary["nbeats"]) != metadata["nb"]:
            warnings.append(
                f"Beat-count mismatch: simulation={summary['nbeats']}, "
                f"reference filename={metadata['nb']}."
            )

    output_name = args.output_name or f"comparison_to_{reference_path.stem}"
    output_dir = simulation_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sim_index, ref_index in enumerate(cell_map):
        rows.append({
            "signal": "phi_axial", "simulation_index": sim_index,
            "reference_index": int(ref_index), "side": "",
            **_metrics(sim_v_grid[:, sim_index], ref_v_grid[:, sim_index]),
        })
    side_names = ("left", "right")
    for sim_index, ref_index in enumerate(junc_map):
        for side, side_name in enumerate(side_names):
            rows.append({
                "signal": "Icleft", "simulation_index": sim_index,
                "reference_index": int(ref_index), "side": side_name,
                **_metrics(sim_i_grid[:, sim_index, side], ref_i_grid[:, sim_index, side]),
            })
        rows.append({
            "signal": "Icleft_net", "simulation_index": sim_index,
            "reference_index": int(ref_index), "side": "sum",
            **_metrics(sim_i_grid[:, sim_index].sum(axis=1),
                       ref_i_grid[:, sim_index].sum(axis=1)),
        })

    aggregate = {
        "phi_axial": _metrics(sim_v_grid, ref_v_grid),
        "Icleft_left": _metrics(sim_i_grid[:, :, 0], ref_i_grid[:, :, 0]),
        "Icleft_right": _metrics(sim_i_grid[:, :, 1], ref_i_grid[:, :, 1]),
        "Icleft_net": _metrics(sim_i_grid.sum(axis=2), ref_i_grid.sum(axis=2)),
    }
    # CV is measured only on the final beat of each trajectory.  Use each
    # native time grid rather than the common plotting grid; only exact BCL
    # endpoints are interpolated when absent from the saved data.
    sim_bcl = float(summary["bcl_ms"])
    sim_nbeats = int(summary["nbeats"])
    ref_bcl = float(metadata.get("bcl", sim_bcl))
    ref_nbeats = int(metadata.get("nb", sim_nbeats))
    sim_cv_t, sim_cv_v, sim_cv_start, sim_cv_end = _last_beat_trajectory(
        sim_t_v, sim_v, sim_bcl, sim_nbeats, "simulation"
    )
    ref_cv_t, ref_cv_v, ref_cv_start, ref_cv_end = _last_beat_trajectory(
        ref_t, ref_v, ref_bcl, ref_nbeats, "MATLAB reference"
    )
    conduction_velocity = {
        "simulation": _conduction_velocity(
            sim_cv_t, sim_cv_v, args.cell_length_um,
            args.cv_threshold_mv, args.cv_cell_start, args.cv_cell_end,
        ),
        "reference": _conduction_velocity(
            ref_cv_t, ref_cv_v, args.cell_length_um,
            args.cv_threshold_mv, args.cv_cell_start, args.cv_cell_end,
        ),
    }
    conduction_velocity["simulation"].update({
        "beat_number_one_based": sim_nbeats,
        "search_window_start_ms": sim_cv_start,
        "search_window_end_ms": sim_cv_end,
    })
    conduction_velocity["reference"].update({
        "beat_number_one_based": ref_nbeats,
        "search_window_start_ms": ref_cv_start,
        "search_window_end_ms": ref_cv_end,
    })
    conduction_velocity["last_beat_only"] = True
    sim_cv = conduction_velocity["simulation"]["cv_cm_s"]
    ref_cv = conduction_velocity["reference"]["cv_cm_s"]
    if isinstance(sim_cv, float) and isinstance(ref_cv, float) and ref_cv != 0.0:
        conduction_velocity["cv_relative_error"] = (sim_cv - ref_cv) / ref_cv
    report = {
        "simulation_directory": str(simulation_dir),
        "simulation_npz": str(simulation_path),
        "simulation_summary": str(summary_path),
        "simulation_checkpoint": summary.get("checkpoint"),
        "simulation_checkpoint_sha256": summary.get("checkpoint_sha256"),
        "reference_mat": str(reference_path),
        "reference_format": reference_format,
        "common_time_start_ms": float(grid[0]),
        "common_time_end_ms": float(grid[-1]),
        "comparison_grid": grid_kind,
        "grid_dt_ms": args.grid_dt_ms,
        "grid_samples": int(grid.size),
        "spatial_match": args.spatial_match,
        "cell_map_sim_to_reference": cell_map.tolist(),
        "junction_map_sim_to_reference": junc_map.tolist(),
        "warnings": warnings,
        "aggregate_metrics": aggregate,
        "conduction_velocity_cm_s": conduction_velocity,
    }
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output_dir / "per_trace_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "interpolated_comparison.npz",
        time_ms=grid,
        simulation_voltage_mv=sim_v_grid,
        reference_voltage_mv=ref_v_grid,
        voltage_error_mv=sim_v_grid - ref_v_grid,
        simulation_icleft_ua=sim_i_grid,
        reference_icleft_ua=ref_i_grid,
        icleft_error_ua=sim_i_grid - ref_i_grid,
        cell_map_sim_to_reference=cell_map,
        junction_map_sim_to_reference=junc_map,
    )

    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    _plot_overlays(
        grid, sim_v_grid, ref_v_grid, cells, cell_map,
        "phi_ax [mV]", "Axial potential: numerical simulation vs MATLAB/FEM",
        output_dir / "voltage_overlay.png", "simulation cell", "reference cell",
    )
    for side, side_name in enumerate(side_names):
        _plot_overlays(
            grid, sim_i_grid[:, :, side], ref_i_grid[:, :, side], junctions,
            junc_map, "Icleft [uA]",
            f"Icleft {side_name}: numerical simulation vs MATLAB/FEM",
            output_dir / f"icleft_{side_name}_overlay.png",
            "simulation junction", "reference junction",
        )
    _plot_overlays(
        grid, sim_i_grid.sum(axis=2), ref_i_grid.sum(axis=2), junctions,
        junc_map, "Icleft left+right [uA]",
        "Net Icleft: numerical simulation vs MATLAB/FEM",
        output_dir / "icleft_net_overlay.png",
        "simulation junction", "reference junction",
    )
    _plot_error_heatmap(
        grid, sim_v_grid - ref_v_grid, "simulation cell index",
        "simulation - reference [mV]", "Axial-potential error",
        output_dir / "voltage_error_heatmap.png",
    )
    _plot_error_heatmap(
        grid, (sim_i_grid - ref_i_grid).sum(axis=2), "simulation junction index",
        "simulation - reference [uA]", "Net-Icleft error",
        output_dir / "icleft_net_error_heatmap.png",
    )

    print(f"Simulation : {simulation_dir}")
    print(f"Reference  : {reference_path}")
    print(f"Grid       : {grid_kind}, {grid.size} samples, {grid[0]:g}..{grid[-1]:g} ms")
    for warning in warnings:
        print(f"WARNING    : {warning}")
    print("Aggregate errors:")
    for signal, values in aggregate.items():
        print(
            f"  {signal:14s} RMSE={values['rmse']:.6g}, "
            f"MAE={values['mae']:.6g}, max={values['max_abs']:.6g}, "
            f"relL2={100.0 * values['relative_l2']:.3f}%"
        )
    print("Conduction velocity (cm/s):")
    for label in ("simulation", "reference"):
        cv = conduction_velocity[label]
        value = cv["cv_cm_s"]
        if isinstance(value, float):
            print(
                f"  {label:11s} {value:8.2f}  "
                f"(beat {cv['beat_number_one_based']}, "
                f"window {cv['search_window_start_ms']:g}.."
                f"{cv['search_window_end_ms']:g} ms; "
                f"cell {cv['start_cell']} @ {cv['start_activation_time_ms']:.6f} ms "
                f"-> cell {cv['end_cell']} @ {cv['end_activation_time_ms']:.6f} ms, "
                f"dx={cv['distance_cm']:.4f} cm, thr={cv['threshold_mv']:.2f} mV)"
            )
        else:
            print(f"  {label:11s} undefined ({cv.get('note', 'no fit')})")
    rel = conduction_velocity.get("cv_relative_error")
    if isinstance(rel, float):
        print(f"  sim/ref rel. error = {100.0 * rel:+.2f}%")
    print(f"Saved comparison: {output_dir}")


if __name__ == "__main__":
    main()
