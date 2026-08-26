#!/usr/bin/env python3
"""Run an ORd11 cable coupled to an FML junction-current surrogate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Numerical_simulation.config_numerical import NUMERICAL_CONFIG
from Numerical_simulation.fml_model import FMLJunctionBatch
from Numerical_simulation.ionic_model.adapter import (
    CellGeometry,
    ORd11CableIonicModel,
    configure_ord11_compilation,
)
from Numerical_simulation.ionic_model.localization import (
    build_axial_f_i_tensor,
    build_boundary_f_i_tensor,
)
from Numerical_simulation.solvers import (
    simulate_macro_endpoint_linear,
)


NUMERICAL_METHOD = "macro_endpoint_linear"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def infer_model_type(name: str) -> str | None:
    """Infer the checkpoint architecture without rebuilding its filename."""
    if "_ResNet_" in name:
        return "ResNet"
    if "_FNN_" in name or "directpred" in name:
        return "FNN"
    return None


def _model_dt_from_name(name: str) -> float | None:
    """Return a conventional dt/dt_ filename tag when one is present."""
    match = re.search(r"(?:^|_)dt_?([0-9]+(?:p[0-9]+)?)(?:_|$)", name)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ORd11 cable with raw two-sided FML currents and implicit "
            "macro-endpoint coupling."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--n-cells", type=int, default=50)
    parser.add_argument("--bcl", type=int, default=200)
    parser.add_argument("--nbeats", type=int, default=5)
    parser.add_argument("--stim-cell", type=int, default=0)
    parser.add_argument("--stim-amp", type=float, default=50.0)
    parser.add_argument("--stim-duration-ms", type=float, default=2.0)
    parser.add_argument("--fast-run", type=parse_bool, default=False)
    parser.add_argument(
        "--record-last-beat-only",
        type=parse_bool,
        default=False,
        help=(
            "Run every beat normally but retain trajectory arrays only from "
            "the final beat; numerical dynamics are unchanged."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--model-name-base", required=True)
    parser.add_argument(
        "--checkpoint-kind", choices=("best_val", "best_train"), default="best_val"
    )
    parser.add_argument("--finetune", type=parse_bool, default=False)
    parser.add_argument("--finetune-bcl", type=int, default=None)
    parser.add_argument(
        "--finetune-model", choices=("best_val", "best_train"), default="best_val"
    )
    parser.add_argument("--memlen", type=int, default=10)
    parser.add_argument("--nrec", type=int, default=3)
    parser.add_argument("--dt-fml", type=float, default=0.1)
    parser.add_argument("--dataset", default="sine_one_cell")
    parser.add_argument("--mesh-idx", type=int, default=1)
    parser.add_argument(
        "--gj-coupling", choices=("strong", "weak"), default="strong"
    )
    parser.add_argument(
        "--model-type", "--model_type", dest="model_type",
        choices=("auto", "FNN", "ResNet"), default="auto",
    )
    parser.add_argument(
        "--hidden-layers", "--hidden_layers", dest="hidden_layers", default=None,
        help="Comma-separated widths; omitted means training-config defaults.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_cells < 2:
        raise ValueError("--n-cells must be at least 2.")
    if args.bcl <= 0 or args.nbeats <= 0:
        raise ValueError("--bcl and --nbeats must be positive.")
    if not 0 <= args.stim_cell < args.n_cells:
        raise ValueError("--stim-cell lies outside the cable.")
    if args.stim_duration_ms < 0.0:
        raise ValueError("--stim-duration-ms cannot be negative.")
    if args.memlen < 0 or args.nrec <= 0 or args.dt_fml <= 0.0:
        raise ValueError("MEMLEN >= 0, NREC > 0, and DT_FML > 0 are required.")
    if args.mesh_idx <= 0:
        raise ValueError("--mesh-idx must be a positive integer.")
    total_time_ms = float(args.bcl) * int(args.nbeats)
    macro_count = total_time_ms / float(args.dt_fml)
    if not np.isclose(macro_count, round(macro_count), rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "BCL*NBEATS must be an integer multiple of DT_FML so every "
            "history commit remains on the trained clock."
        )
    tagged_dt = _model_dt_from_name(args.model_name_base)
    if tagged_dt is not None and not np.isclose(
        tagged_dt, args.dt_fml, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            f"MODEL_NAME_BASE encodes dt={tagged_dt:g} ms but "
            f"DT_FML={args.dt_fml:g} ms."
        )
    expected_prefix = f"FEM{args.mesh_idx}_GJ{args.gj_coupling}_VC_"
    if not args.model_name_base.startswith(expected_prefix):
        raise ValueError(
            "MODEL_NAME_BASE does not match --mesh-idx/--gj-coupling: "
            f"expected prefix {expected_prefix!r}, got "
            f"{args.model_name_base!r}."
        )


def resolve_training_config(args: argparse.Namespace):
    """Instantiate the shared training Config with explicit FML metadata."""
    inferred = infer_model_type(args.model_name_base)
    model_type = inferred if args.model_type == "auto" else args.model_type
    if model_type is None:
        raise ValueError(
            "Cannot infer FNN/ResNet from MODEL_NAME_BASE; set MODEL_TYPE."
        )
    sys.argv.extend(
        (
            "--memlen", str(args.memlen),
            "--nrec", str(args.nrec),
            "--dt", str(args.dt_fml),
            "--dataset", str(args.dataset),
            "--model_type", model_type,
            "--mesh-idx", str(args.mesh_idx),
            "--gj-coupling", str(args.gj_coupling),
        )
    )
    if args.hidden_layers:
        sys.argv.extend(("--hidden_layers", args.hidden_layers))
    from config import Config

    config = Config()
    config.MODEL_TYPE = model_type
    config.DEVICE = select_device(NUMERICAL_CONFIG.device)
    config.DATA_TYPE = torch.float64
    return config


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config_numerical.py requests CUDA, but it is unavailable.")
    return torch.device(name)


def resolve_checkpoint(args: argparse.Namespace, model_dir: str | Path) -> Path:
    if args.finetune:
        finetune_bcl = args.bcl if args.finetune_bcl is None else args.finetune_bcl
        filename = (
            f"{args.model_name_base}_finetune_BCL{finetune_bcl}_"
            f"{args.finetune_model}.pth"
        )
    else:
        filename = f"{args.model_name_base}_{args.checkpoint_kind}.pth"
    checkpoint = Path(model_dir) / filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FML checkpoint not found: {checkpoint}")
    return checkpoint.resolve()


def estimate_micro_steps(args: argparse.Namespace, numerical) -> int:
    total_time_ms = float(args.bcl) * int(args.nbeats)
    if not numerical.adaptive_dt:
        return int(np.ceil(total_time_ms / numerical.dt_fine_ms))
    fine_per_beat = min(float(numerical.fine_window_ms), float(args.bcl))
    return int(args.nbeats) * (
        int(np.ceil(fine_per_beat / numerical.dt_fine_ms))
        + int(np.ceil(
            (float(args.bcl) - fine_per_beat) / numerical.dt_coarse_ms
        ))
    )


def save_result(
    result, args, numerical, config, checkpoint, ionic, junction,
    axial_f_i, boundary_f_i, compile_ord, compile_mode,
    boundary_area_source, estimated_steps, program_start,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "simulation.npz"
    np.savez(
        data_path,
        time_ms=result.time_ms,
        voltage_mv=result.voltage_mv,
        boundary_voltage_mv=result.boundary_voltage_mv,
        current_time_ms=result.current_time_ms,
        time_step_ms=result.time_step_ms,
        ionic_current_ua=result.ionic_current_ua,
        cleft_current_ua=result.cleft_current_ua,
        cell_cleft_current_ua=result.cell_cleft_current_ua,
        boundary_current_ua=result.boundary_current_ua,
        boundary_ionic_current_ua=result.boundary_ionic_current_ua,
        stimulus_current_ua=result.stimulus_current_ua,
        fml_commit_time_ms=result.fml_commit_time_ms,
        axial_f_i=axial_f_i.cpu().numpy(),
        boundary_f_i=boundary_f_i.cpu().numpy(),
        sigma=result.sigma,
        dt_ms=np.array(result.dt_ms),
        dt_fml_ms=np.array(result.dt_ms),
        dt_fine_ms=np.array(result.dt_fine_ms),
        dt_coarse_ms=np.array(result.dt_coarse_ms),
        fine_window_ms=np.array(result.fine_window_ms),
        capacitance_uf=np.array(result.capacitance_uf),
        boundary_capacitance_uf=np.array(result.boundary_capacitance_uf),
        boundary_area_um2=np.array(result.boundary_area_um2),
        boundary_area_source=np.array(str(boundary_area_source)),
        boundary_area_formula=np.array(
            "sum(FEM_data.partition_surface)"
        ),
        gmyo_ms=np.array(result.gmyo_ms),
        bcl_ms=np.array(result.bcl_ms),
        nbeats=np.array(result.nbeats),
        iteration_wall_time_s=result.iteration_wall_time_s,
        simulation_wall_time_s=np.array(result.simulation_wall_time_s),
        coupling_iterations=result.coupling_iterations,
        coupling_converged=result.coupling_converged,
        coupling_residual_mv=result.coupling_residual_mv,
        numerical_method=np.array(NUMERICAL_METHOD),
        macro_reconstruction=np.array(numerical.macro_reconstruction),
        model_type=np.array(config.MODEL_TYPE),
        fem_mesh=np.array(args.mesh_idx),
        gj_coupling=np.array(args.gj_coupling),
    )
    if numerical.make_plots:
        from Numerical_simulation.visualization import plot_result
        plot_paths = plot_result(result, args.output_dir)
    else:
        plot_paths = []

    iteration_time = result.iteration_wall_time_s
    summary = {
        "n_cells": args.n_cells,
        "n_junctions": args.n_cells - 1,
        "bcl_ms": args.bcl,
        "nbeats": args.nbeats,
        "total_time_ms": args.bcl * args.nbeats,
        "numerical_method": NUMERICAL_METHOD,
        "macro_reconstruction": numerical.macro_reconstruction,
        "fml_dt_ms": result.dt_ms,
        "dt_fine_ms": result.dt_fine_ms,
        "dt_coarse_ms": result.dt_coarse_ms,
        "fine_window_ms": result.fine_window_ms,
        "adaptive_dt": numerical.adaptive_dt,
        "fml_history_clock_strict": True,
        "fml_history_commits": int(result.fml_commit_time_ms.size),
        "scheme": "macro_endpoint_quasi_newton_with_rollback_line_search",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_name_base": args.model_name_base,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_finetuned": bool(args.finetune),
        "finetune_bcl": args.finetune_bcl,
        "finetune_model": args.finetune_model,
        "model_type": config.MODEL_TYPE,
        "dataset": args.dataset,
        "fem_mesh": args.mesh_idx,
        "gj_coupling": args.gj_coupling,
        "memlen": args.memlen,
        "nrec": args.nrec,
        "hidden_layers": config.HIDDEN_LAYER_SIZES,
        "fml_ctrl_channels": config.NUM_CTRL,
        "ctrl_contract": "aligned_history_plus_target_time_phi",
        "rest_voltage_mv": ionic.rest_voltage_mv,
        "fml_rest_current_bias_ua": junction.rest_current_bias.cpu().tolist(),
        "subtract_rest_bias": numerical.subtract_rest_bias,
        "current_mapping": "raw_owner_side",
        "owner_side_kcl": True,
        "icleft_multiplier": 1.0,
        "f_i_mode": "matlab_axial_1_minus_loc",
        "axial_f_i": axial_f_i.cpu().tolist(),
        "boundary_f_i": boundary_f_i.cpu().tolist(),
        "stimulus_cell": args.stim_cell,
        "stimulus_amplitude": args.stim_amp,
        "stimulus_duration_ms": args.stim_duration_ms,
        "stimulus_multiplied_by_f_i": False,
        "capacitance_mode": "axial_membrane",
        "capacitance_uf": result.capacitance_uf,
        "terminal_boundary": numerical.terminal_boundary,
        "boundary_model": "matlab_active_terminal_disc_ord11",
        "boundary_area_um2": result.boundary_area_um2,
        "boundary_area_source": str(boundary_area_source),
        "boundary_area_formula": "sum(FEM_data.partition_surface)",
        "boundary_area_mesh_idx": args.mesh_idx,
        "boundary_capacitance_uf": result.boundary_capacitance_uf,
        "rho_myo_kohm_um": numerical.rho_myo_kohm_um,
        "gmyo_ms": result.gmyo_ms,
        "min_voltage_mv": float(result.voltage_mv.min()),
        "max_voltage_mv": float(result.voltage_mv.max()),
        "simulation_wall_time_s": result.simulation_wall_time_s,
        "mean_iteration_wall_time_s": float(iteration_time.mean()),
        "min_iteration_wall_time_s": float(iteration_time.min()),
        "max_iteration_wall_time_s": float(iteration_time.max()),
        "coupling_iters_max_allowed": numerical.coupling_iters,
        "coupling_iters_mean": float(result.coupling_iterations.mean()),
        "coupling_iters_max_used": int(result.coupling_iterations.max()),
        "coupling_steps_not_converged": int(
            (~result.coupling_converged).sum()
        ),
        "coupling_residual_max_mv": float(result.coupling_residual_mv.max()),
        "coupling_residual_p99_mv": float(
            np.quantile(result.coupling_residual_mv, 0.99)
        ),
        "fast_run": bool(args.fast_run),
        "record_fml_only": result.record_fml_only,
        "record_last_beat_only": result.record_last_beat_only,
        "make_plots": numerical.make_plots,
        "device": str(config.DEVICE),
        "torch_threads": torch.get_num_threads(),
        "compile_ord11": compile_ord,
        "compile_ord11_mode": compile_mode,
        "estimated_electrical_steps": estimated_steps,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved data    : {data_path}")
    for path in plot_paths:
        print(f"Saved plot    : {path}")
    print(f"Saved summary : {summary_path}")
    print(f"Total program wall time: {time.perf_counter() - program_start:.9f} s")


def main() -> None:
    program_start = time.perf_counter()
    args = build_parser().parse_args()
    validate_args(args)
    numerical = (
        NUMERICAL_CONFIG.for_fml_dt(args.dt_fml)
        .for_fast_run(args.fast_run)
        .for_last_beat_only(args.record_last_beat_only)
    )
    boundary_area_um2, boundary_area_source = (
        numerical.load_boundary_area_um2(args.mesh_idx)
    )
    config = resolve_training_config(args)
    checkpoint = resolve_checkpoint(args, config.MODEL_DIR)
    torch.set_default_dtype(torch.float64)
    if numerical.torch_threads <= 0:
        raise ValueError("torch_threads in config_numerical.py must be positive.")
    torch.set_num_threads(numerical.torch_threads)
    estimated_steps = estimate_micro_steps(args, numerical)
    compile_ord = numerical.compile_ord11 == "true" or (
        numerical.compile_ord11 == "auto"
        and estimated_steps >= numerical.compile_threshold_steps
    )
    compile_ord, compile_mode = configure_ord11_compilation(
        compile_ord, device=config.DEVICE
    )
    geometry = CellGeometry(
        length_um=numerical.length_um,
        radius_um=numerical.radius_um,
        cm_density_uf_per_um2=numerical.cm_density_uf_per_um2,
    )
    axial_f_i = build_axial_f_i_tensor(
        geometry, device=config.DEVICE, dtype=config.DATA_TYPE
    )
    boundary_f_i = build_boundary_f_i_tensor(
        geometry, cell_port_count=2,
        device=config.DEVICE, dtype=config.DATA_TYPE,
    )
    ionic = ORd11CableIonicModel(
        n_cells=args.n_cells,
        dt_ms=numerical.dt_fine_ms,
        device=config.DEVICE,
        dtype=config.DATA_TYPE,
        geometry=geometry,
        stimulus_cell=args.stim_cell,
        stimulus_amplitude=args.stim_amp,
        stimulus_duration_ms=args.stim_duration_ms,
        stimulus_bcl_ms=args.bcl,
        f_i=axial_f_i,
    )
    boundary_ionic = ORd11CableIonicModel(
        n_cells=2,
        dt_ms=numerical.dt_fine_ms,
        device=config.DEVICE,
        dtype=config.DATA_TYPE,
        geometry=geometry,
        stimulus_cell=0,
        stimulus_amplitude=0.0,
        stimulus_duration_ms=0.0,
        stimulus_bcl_ms=args.bcl,
        f_i=boundary_f_i,
    )
    junction = FMLJunctionBatch(
        config=config,
        n_cells=args.n_cells,
        rest_voltage_mv=ionic.rest_voltage_mv,
        checkpoint=checkpoint,
        subtract_rest_bias=numerical.subtract_rest_bias,
    )
    print("=" * 72)
    print("ORd11 + FML cable simulation")
    print(f"method              : {NUMERICAL_METHOD}")
    print(f"current policy      : {numerical.macro_reconstruction}")
    print(f"cells / junctions   : {args.n_cells} / {args.n_cells - 1}")
    print(f"BCL / nbeats        : {args.bcl} ms / {args.nbeats}")
    print(f"FML dt              : {args.dt_fml} ms (strict history clock)")
    print(
        "electrical dt       : "
        f"{numerical.dt_fine_ms}/{numerical.dt_coarse_ms} ms"
    )
    print(f"model               : {config.MODEL_TYPE}, {checkpoint.name}")
    print("Ctrl contract       : per-channel M+2 (target-time Ctrl included)")
    print("raw owner currents  : true (no antisymmetry projection)")
    print(f"terminal boundary   : {numerical.terminal_boundary}")
    print(
        "boundary area       : "
        f"{boundary_area_um2:.15g} um^2 "
        "(sum(FEM_data.partition_surface))"
    )
    print(f"boundary mesh file  : {boundary_area_source}")
    print(f"fast output mode    : {args.fast_run}")
    print(f"last-beat output    : {args.record_last_beat_only}")
    print(f"device / dtype      : {config.DEVICE} / {config.DATA_TYPE}")
    print("=" * 72)
    common_solver_args = dict(
        bcl_ms=args.bcl,
        nbeats=args.nbeats,
        progress_every=numerical.progress_every,
        timing_interval_ms=numerical.timing_interval_ms,
        dt_fine_ms=numerical.dt_fine_ms,
        dt_coarse_ms=numerical.dt_coarse_ms,
        fine_window_ms=numerical.fine_window_ms,
        adaptive_dt=numerical.adaptive_dt,
        fml_dt_ms=args.dt_fml,
        rho_myo_kohm_um=numerical.rho_myo_kohm_um,
        terminal_boundary=numerical.terminal_boundary,
        boundary_area_um2=boundary_area_um2,
        record_fml_only=numerical.record_fml_only,
        record_last_beat_only=numerical.record_last_beat_only,
    )
    result = simulate_macro_endpoint_linear(
        ionic,
        boundary_ionic,
        junction,
        coupling_iters=numerical.coupling_iters,
        coupling_tol=numerical.coupling_tol_mv,
        coupling_relax=numerical.coupling_relax,
        coupling_min_relax=numerical.coupling_min_relax,
        fail_on_nonconvergence=numerical.fail_on_nonconvergence,
        **common_solver_args,
    )
    save_result(
        result, args, numerical, config, checkpoint, ionic, junction,
        axial_f_i, boundary_f_i, compile_ord, compile_mode,
        boundary_area_source, estimated_steps, program_start,
    )


if __name__ == "__main__":
    main()
