#!/usr/bin/env python3
"""Build event-balanced fine-tuning bursts from a real cable run.

Every split contains a fixed number of windows per junction. A configurable
fraction is sampled from high-activity regions measured from the future
Icleft/voltage increments; the remainder provides uniform background coverage.
This prevents rare upstrokes from being overwhelmed by quiet rest-state data
and makes the training budget comparable across DT_FML values.

Each QoI training sample has the standard FML length

    NUM_MEMORY + 1 + NUM_RECURRENT

Validation has the fixed length

    NUM_MEMORY + 1 + NUM_VAL_RECURRENT

independent of DT_FML. Ctrl starts at the same time and has one additional
sample. Gating states are not loaded or saved.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


CABLE_ROOT = Path(
    "/fs/ess/PAS1622/RichardSui/MultiMesh_Cable_Cleft/VaryBCL"
)
DEFAULT_REST_VOLTAGE_MV = -87.9989146999539


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def default_mat_path(config: Config) -> Path:
    coupling = str(config.GJ_COUPLING).lower()
    if coupling not in {"strong", "weak"}:
        raise ValueError(f"Unsupported GJ_COUPLING={config.GJ_COUPLING!r}")
    tag = "StrongGJ" if coupling == "strong" else "WeakGJ"
    return (
        CABLE_ROOT / tag / f"BCL{config.BCL}"
        / f"FEMDATA_{config.MESH_IDX}mat_{tag}_BCL_{config.BCL}.mat"
    )


def uniform_relative_grid(window_ms: float, dt_ms: float) -> np.ndarray:
    n_steps = int(round(window_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, window_ms, atol=1.0e-10, rtol=0.0):
        raise ValueError(
            f"window {window_ms} ms is not an integer multiple of dt={dt_ms} ms"
        )
    return np.arange(n_steps + 1, dtype=np.float64) * dt_ms


def interpolate_beat(
    interpolator: PchipInterpolator,
    beat_index: int,
    bcl_ms: float,
    relative_grid: np.ndarray,
) -> np.ndarray:
    absolute_time = beat_index * bcl_ms + relative_grid
    values = interpolator(absolute_time)
    if not np.isfinite(values).all():
        raise ValueError(
            f"Interpolation failed for beat {beat_index}: requested "
            f"[{absolute_time[0]}, {absolute_time[-1]}] ms"
        )
    return values


def adjacent_ctrl(phi: np.ndarray) -> np.ndarray:
    """Return adjacent voltages with shape (time, junction, 2)."""
    return np.stack((phi[:, :-1], phi[:, 1:]), axis=2)


def _scaled_increment_activity(qoi: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """Dimensionless local activity used only to choose burst starts."""
    ctrl = ctrl[:qoi.shape[0]]
    dq = np.max(np.abs(np.diff(qoi, axis=0)), axis=2)
    dc = np.max(np.abs(np.diff(ctrl, axis=0)), axis=2)

    def scale(values: np.ndarray) -> np.ndarray:
        denom = np.quantile(values, 0.95, axis=0)
        denom = np.maximum(denom, np.finfo(np.float64).eps)
        return values / denom[None, :]

    activity = np.zeros(qoi.shape[:2], dtype=np.float64)
    activity[1:] = scale(dq) + 0.25 * scale(dc)
    return activity


def event_balanced_bursts(
    qoi: np.ndarray,
    ctrl: np.ndarray,
    burst_len: int,
    history_len: int,
    bursts_per_junction: int,
    event_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return paired QoI/Ctrl bursts using identical event-balanced starts."""
    if not 0.0 <= event_fraction <= 1.0:
        raise ValueError("event_fraction must lie in [0, 1]")
    n_time, n_junction, n_qoi = qoi.shape
    if ctrl.shape[:2] != (n_time + 1, n_junction):
        raise ValueError(
            f"Ctrl must have one more time sample than QoI; qoi={qoi.shape}, "
            f"ctrl={ctrl.shape}"
        )
    n_starts = n_time - burst_len + 1
    if n_starts <= 0:
        raise ValueError(
            f"trajectory has {n_time} samples, shorter than burst_len={burst_len}"
        )
    if bursts_per_junction <= 0 or bursts_per_junction > n_starts:
        raise ValueError(
            f"bursts_per_junction={bursts_per_junction} must be in "
            f"[1, {n_starts}]"
        )

    future_len = burst_len - history_len
    if future_len <= 0:
        raise ValueError("burst must contain at least one future target")
    activity = _scaled_increment_activity(qoi, ctrl)
    activity_windows = np.lib.stride_tricks.sliding_window_view(
        activity, window_shape=future_len, axis=0
    )
    scores = activity_windows[
        history_len:history_len + n_starts
    ].max(axis=2)

    starts = np.empty((n_junction, bursts_per_junction), dtype=np.int64)
    event_mask = np.zeros_like(starts, dtype=bool)
    n_event = int(round(bursts_per_junction * event_fraction))
    all_starts = np.arange(n_starts)
    for junction in range(n_junction):
        selected_event = np.empty(0, dtype=np.int64)
        if n_event:
            threshold = np.quantile(scores[:, junction], 0.85)
            candidates = all_starts[scores[:, junction] >= threshold]
            if candidates.size < n_event:
                candidates = np.argsort(scores[:, junction])[-n_event:]
            selected_event = rng.choice(candidates, n_event, replace=False)
        remaining = np.setdiff1d(all_starts, selected_event, assume_unique=False)
        n_background = bursts_per_junction - selected_event.size
        selected_background = rng.choice(
            remaining, n_background, replace=False
        )
        starts[junction] = np.concatenate(
            (selected_event, selected_background)
        )
        event_mask[junction, :selected_event.size] = True

    qoi_bursts = np.empty(
        (n_junction * bursts_per_junction, n_qoi, burst_len),
        dtype=np.float64,
    )
    ctrl_bursts = np.empty(
        (n_junction * bursts_per_junction, ctrl.shape[2], burst_len + 1),
        dtype=np.float64,
    )
    row = 0
    for junction in range(n_junction):
        for start in starts[junction]:
            qoi_bursts[row] = qoi[start:start + burst_len, junction].T
            ctrl_bursts[row] = ctrl[start:start + burst_len + 1, junction].T
            row += 1
    return qoi_bursts, ctrl_bursts, starts, event_mask


def scaling_metadata(config: Config) -> dict[str, np.ndarray]:
    scaled = bool(getattr(config, "SCALE_DATA", False))
    factor = float(getattr(config, "QOI_SCALE_FACTOR", 1000.0)) if scaled else 1.0
    return {
        "normalized": np.array(False),
        "normalization_mode": np.array("none"),
        "scaled": np.array(scaled),
        "scaling_mode": np.array(
            "qoi_multiply_constant" if scaled else "none"
        ),
        "qoi_scale_factor": np.array(factor, dtype=np.float64),
        "ctrl_scale_factor": np.array(1.0, dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate event-balanced cable fine-tuning data.",
        allow_abbrev=False,
    )
    parser.add_argument("--mat", type=Path, default=None)
    parser.add_argument("--train-beat", type=int, default=0)
    parser.add_argument("--val-beat", type=int, default=1)
    parser.add_argument("--test-beat", type=int, default=2)
    parser.add_argument("--train-window-ms", type=float, default=400.0)
    parser.add_argument("--eval-window-ms", type=float, default=400.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument(
        "--train-bursts-per-junction", type=int,
        default=int(os.environ.get("TRAIN_BURSTS_PER_JUNCTION", "512")),
    )
    parser.add_argument(
        "--val-bursts-per-junction", type=int,
        default=int(os.environ.get("VAL_BURSTS_PER_JUNCTION", "64")),
    )
    parser.add_argument(
        "--test-bursts-per-junction", type=int,
        default=int(os.environ.get("TEST_BURSTS_PER_JUNCTION", "64")),
    )
    parser.add_argument(
        "--event-fraction", type=float,
        default=float(os.environ.get("EVENT_FRACTION", "0.5")),
    )

    # Accepted here because Config independently reads the same argv.
    parser.add_argument("--memlen", type=int, required=True)
    parser.add_argument("--dataset", default="sine_one_cell")
    parser.add_argument("--bcl", type=int, default=None)
    parser.add_argument("--nrec", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--mesh-idx", "--mesh_idx", dest="mesh_idx",
                        type=int, default=1)
    parser.add_argument("--gj-coupling", "--gj_coupling",
                        dest="gj_coupling", choices=("strong", "weak"),
                        default="strong")
    parser.add_argument("--hidden_layers", type=str, default=None)
    parser.add_argument("--model_type", choices=("FNN", "ResNet", "LSTM"),
                        default="FNN")
    args = parser.parse_args()

    config = Config()
    dt_ms = float(config.dt)
    bcl_ms = float(config.BCL)
    train_len = int(config.BURST_LEN)
    val_len = int(config.VAL_BURST_LEN)
    test_len = config.NUM_MEMORY + 1 + int(round(config.T_TEST / dt_ms))
    test_nrec = test_len - config.NUM_MEMORY - 1

    if args.train_window_ms > bcl_ms or args.eval_window_ms > bcl_ms:
        raise ValueError(
            "train/eval windows must not cross a beat boundary: "
            f"BCL={bcl_ms:g}, train={args.train_window_ms:g}, "
            f"eval={args.eval_window_ms:g} ms"
        )

    beats = (args.train_beat, args.val_beat, args.test_beat)
    if len(set(beats)) != 3 or min(beats) < 0:
        raise ValueError("train/val/test beats must be three distinct nonnegative indices")

    mat_path = (args.mat or default_mat_path(config)).expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)

    output_dir = (args.output_dir or Path(config.DATA_DIR)).expanduser().resolve()
    prefix = args.output_prefix or (
        f"finetune_cable_FEM{config.MESH_IDX}_GJ{str(config.GJ_COUPLING).lower()}_"
        f"BCL{config.BCL}_nm{config.NUM_MEMORY}_nr{config.NUM_RECURRENT}"
        f"_dt{config._DT_TAG}"
    )
    paths = {
        split: output_dir / f"{prefix}_{split}.npz"
        for split in ("train", "val", "test")
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output exists; pass --overwrite true to replace it:\n  "
            + "\n  ".join(map(str, existing))
        )

    print("=" * 72)
    print("Cable FEM -> FML fine-tuning dataset")
    print(f"source              : {mat_path}")
    print(f"dt / BCL            : {dt_ms} / {bcl_ms} ms")
    print(f"train/val/test beats: {beats}")
    print(f"train interval       : [0, {args.train_window_ms:g}] ms")
    print(f"eval intervals       : [0, {args.eval_window_ms:g}] ms")
    print(f"nM / nR / train L    : {config.NUM_MEMORY} / "
          f"{config.NUM_RECURRENT} / {train_len}")
    print(f"validation nR / L    : {config.NUM_VAL_RECURRENT} / {val_len}")
    print(f"test nR / L          : {test_nrec} / {test_len}")
    print("Ctrl contract        : aligned, one sample longer than QoI")
    print(f"ctrl channels        : {config.NUM_CTRL}")
    print("=" * 72)

    with h5py.File(mat_path, "r") as fem:
        native_time = np.asarray(fem["t_save"]).reshape(-1).astype(np.float64)
        phi = np.asarray(fem["phi"], dtype=np.float64)
        cleft = np.moveaxis(
            np.asarray(fem["Icleft"], dtype=np.float64), -1, 0
        )
        if phi.shape[0] != native_time.size or cleft.shape[0] != native_time.size:
            raise ValueError("time dimension mismatch in source MAT")
        n_cell = phi.shape[1]
        n_junction = cleft.shape[1]
        if n_junction != n_cell - 1 or cleft.shape[2] != 2:
            raise ValueError(
                f"expected a 1D cable: phi={phi.shape}, Icleft={cleft.shape}"
            )

    native_steps = np.diff(native_time)
    native_dt_min = float(np.min(native_steps))
    native_dt_median = float(np.median(native_steps))
    if dt_ms < native_dt_min - 1.0e-12:
        print(
            "WARNING: DT_FML is finer than the native MATLAB save interval "
            f"({dt_ms:g} < {native_dt_min:g} ms). PCHIP only interpolates the "
            "saved curve and cannot recover unresolved Icleft upstroke dynamics. "
            "For production training regenerate MATLAB data with native save "
            "spacing <= DT_FML."
        )


    # The legacy cable file's first saved point is t=0.04 ms. Add the known
    # t=0 resting state so the requested training interval truly begins at 0.
    if native_time[0] > 0.0:
        native_time = np.concatenate(([0.0], native_time))
        phi = np.concatenate((
            np.full((1, n_cell), DEFAULT_REST_VOLTAGE_MV), phi
        ), axis=0)
        cleft = np.concatenate((np.zeros((1, n_junction, 2)), cleft), axis=0)

    max_requested_time = max(
        args.train_beat * bcl_ms + args.train_window_ms,
        args.val_beat * bcl_ms + args.eval_window_ms,
        args.test_beat * bcl_ms + args.eval_window_ms,
    )
    if max_requested_time > native_time[-1] + 1.0e-9:
        raise ValueError(
            f"requested data through {max_requested_time} ms, but source ends at "
            f"{native_time[-1]} ms"
        )

    phi_interp = PchipInterpolator(native_time, phi, axis=0, extrapolate=False)
    cleft_interp = PchipInterpolator(native_time, cleft, axis=0, extrapolate=False)

    train_grid = uniform_relative_grid(args.train_window_ms, dt_ms)
    eval_grid = uniform_relative_grid(args.eval_window_ms, dt_ms)

    def beat_arrays(beat: int, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beat_phi = interpolate_beat(phi_interp, beat, bcl_ms, grid)
        beat_cleft = interpolate_beat(cleft_interp, beat, bcl_ms, grid)
        return adjacent_ctrl(beat_phi), beat_cleft

    train_ctrl_tf, train_qoi_tf = beat_arrays(args.train_beat, train_grid)
    val_ctrl_tf, val_qoi_tf = beat_arrays(args.val_beat, eval_grid)
    test_ctrl_tf, test_qoi_tf = beat_arrays(args.test_beat, eval_grid)

    # Drop the final QoI only when choosing starts. This yields the same set of
    # valid starts as Ctrl windows of length QoI length + 1.
    train_qoi_source = train_qoi_tf[:-1]
    train_ctrl_source = train_ctrl_tf
    val_qoi_source = val_qoi_tf[:-1]
    val_ctrl_source = val_ctrl_tf
    test_qoi_source = test_qoi_tf[:-1]
    test_ctrl_source = test_ctrl_tf

    rng = np.random.default_rng(
        config.DATA_SEED if args.seed is None else args.seed
    )
    history_len = config.NUM_MEMORY + 1
    train_qoi, train_ctrl, train_starts, train_event_mask = event_balanced_bursts(
        train_qoi_source, train_ctrl_source, train_len, history_len,
        args.train_bursts_per_junction, args.event_fraction, rng,
    )
    val_qoi, val_ctrl, val_starts, val_event_mask = event_balanced_bursts(
        val_qoi_source, val_ctrl_source, val_len, history_len,
        args.val_bursts_per_junction, args.event_fraction, rng,
    )
    test_qoi, test_ctrl, test_starts, test_event_mask = event_balanced_bursts(
        test_qoi_source, test_ctrl_source, test_len, history_len,
        args.test_bursts_per_junction, args.event_fraction, rng,
    )

    metadata = scaling_metadata(config)
    qoi_factor = float(metadata["qoi_scale_factor"])
    if qoi_factor != 1.0:
        train_qoi *= qoi_factor
        val_qoi *= qoi_factor
        test_qoi *= qoi_factor

    common = {
        **metadata,
        "source_mat": np.array(str(mat_path)),
        "dt_ms": np.array(dt_ms),
        "bcl_ms": np.array(bcl_ms),
        "num_memory": np.array(config.NUM_MEMORY),
        "junction_count": np.array(n_junction),
        "mesh_idx": np.array(config.MESH_IDX),
        "gj_coupling": np.array(str(config.GJ_COUPLING).lower()),
        "sampling_mode": np.array("event_balanced_fixed_count"),
        "event_fraction": np.array(args.event_fraction),
        "native_dt_min_ms": np.array(native_dt_min),
        "native_dt_median_ms": np.array(native_dt_median),
        "native_upsample_ratio": np.array(native_dt_min / dt_ms),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        paths["train"], qoi=train_qoi, ctrl=train_ctrl,
        beat_index=np.array(args.train_beat),
        interval_ms=np.array([0.0, args.train_window_ms]),
        starts_steps=train_starts, starts_ms=train_starts * dt_ms,
        event_selected=train_event_mask,
        bursts_per_junction=np.array(args.train_bursts_per_junction),
        num_recurrent=np.array(config.NUM_RECURRENT), **common,
    )
    np.savez(
        paths["val"], qoi=val_qoi, ctrl=val_ctrl,
        beat_index=np.array(args.val_beat),
        interval_ms=np.array([0.0, args.eval_window_ms]),
        starts_steps=val_starts, starts_ms=val_starts * dt_ms,
        event_selected=val_event_mask,
        bursts_per_junction=np.array(args.val_bursts_per_junction),
        num_recurrent=np.array(config.NUM_VAL_RECURRENT), **common,
    )
    np.savez(
        paths["test"], qoi=test_qoi, ctrl=test_ctrl,
        beat_index=np.array(args.test_beat),
        interval_ms=np.array([0.0, args.eval_window_ms]),
        starts_steps=test_starts, starts_ms=test_starts * dt_ms,
        event_selected=test_event_mask,
        bursts_per_junction=np.array(args.test_bursts_per_junction),
        num_recurrent=np.array(test_nrec), **common,
    )

    print(f"train: qoi {train_qoi.shape}  ctrl {train_ctrl.shape}")
    print(f"val  : qoi {val_qoi.shape}  ctrl {val_ctrl.shape}")
    print(f"test : qoi {test_qoi.shape}  ctrl {test_ctrl.shape}")
    print(f"train windows/junction: {train_qoi.shape[0] // n_junction}")
    for split, path in paths.items():
        print(f"{split:5s}: {path}")
        print(f"OUT_{split.upper()}={path}")


if __name__ == "__main__":
    main()
