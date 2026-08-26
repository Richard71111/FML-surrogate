#!/usr/bin/env python3
"""Calculate last-beat CV for a BCL sweep and plot BCL versus CV.

The input manifest is written by ``submit_bcl_cv_sweep.sh`` and identifies the
exact Slurm job/output directory for every BCL.  This avoids accidentally
mixing a new sweep with older output folders.  The CV definition is shared
with the MATLAB comparison code: first upward 10 mV crossings at zero-based
cells 10 and 40, linearly localized on the native saved time grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-fml-bcl-cv")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Numerical_simulation.analysis.compare_to_matlab import (
    _conduction_velocity,
    _last_beat_trajectory,
    _strictly_increasing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot last-beat conduction velocity versus BCL."
    )
    parser.add_argument("manifest", type=Path, help="Sweep manifest CSV.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output folder (default: manifest folder).",
    )
    parser.add_argument("--threshold-mv", type=float, default=10.0)
    parser.add_argument("--start-cell", type=int, default=10)
    parser.add_argument("--end-cell", type=int, default=40)
    parser.add_argument("--cell-length-um", type=float, default=100.0)
    return parser


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"bcl_ms", "job_id", "output_directory"}
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def _analyze_run(
    row: dict[str, str], threshold_mv: float, start_cell: int,
    end_cell: int, cell_length_um: float,
) -> dict[str, object]:
    bcl = float(row["bcl_ms"])
    job_id = row["job_id"]
    run_dir = Path(row["output_directory"]).expanduser().resolve()
    record: dict[str, object] = {
        "bcl_ms": bcl,
        "job_id": job_id,
        "status": "missing",
        "cv_cm_s": None,
        "start_activation_time_ms": None,
        "end_activation_time_ms": None,
        "travel_time_ms": None,
        "last_beat_start_ms": None,
        "last_beat_end_ms": None,
        "nbeats": None,
        "output_directory": str(run_dir),
        "message": "",
    }
    summary_path = run_dir / "summary.json"
    data_path = run_dir / "simulation.npz"
    missing = [str(path.name) for path in (summary_path, data_path) if not path.is_file()]
    if missing:
        record["message"] = "missing " + ", ".join(missing)
        return record

    try:
        summary = json.loads(summary_path.read_text())
        saved_bcl = float(summary["bcl_ms"])
        nbeats = int(summary["nbeats"])
        if not np.isclose(saved_bcl, bcl, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"manifest BCL={bcl:g} differs from summary BCL={saved_bcl:g}"
            )
        with np.load(data_path) as data:
            time_ms = np.asarray(data["time_ms"], dtype=float).reshape(-1)
            voltage_mv = np.asarray(data["voltage_mv"], dtype=float)
        if voltage_mv.ndim != 2:
            raise ValueError(f"voltage_mv must be 2-D; got {voltage_mv.shape}")
        time_ms, voltage_mv = _strictly_increasing(
            time_ms, voltage_mv, f"BCL={bcl:g} simulation voltage"
        )
        beat_t, beat_v, beat_start, beat_end = _last_beat_trajectory(
            time_ms, voltage_mv, saved_bcl, nbeats, f"BCL={bcl:g} simulation"
        )
        cv = _conduction_velocity(
            beat_t, beat_v, cell_length_um, threshold_mv, start_cell, end_cell
        )
        record.update({
            "status": "ok" if cv["cv_cm_s"] is not None else "no_cv",
            "cv_cm_s": cv["cv_cm_s"],
            "start_activation_time_ms": cv["start_activation_time_ms"],
            "end_activation_time_ms": cv["end_activation_time_ms"],
            "travel_time_ms": cv["travel_time_ms"],
            "last_beat_start_ms": beat_start,
            "last_beat_end_ms": beat_end,
            "nbeats": nbeats,
            "message": cv.get("note", ""),
        })
    except Exception as exc:  # Preserve all BCLs in the final audit table.
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
    return record


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "bcl_ms", "cv_cm_s", "status", "start_activation_time_ms",
        "end_activation_time_ms", "travel_time_ms", "last_beat_start_ms",
        "last_beat_end_ms", "nbeats", "job_id", "output_directory", "message",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot(path: Path, records: list[dict[str, object]]) -> None:
    if not any(record["cv_cm_s"] is not None for record in records):
        raise RuntimeError("No completed BCL has a valid last-beat CV.")
    # Retain every requested BCL. Undefined CV values become NaN so the line
    # visibly breaks instead of connecting across a failed/non-conducted case.
    bcl = np.asarray([record["bcl_ms"] for record in records], dtype=float)
    cv = np.asarray([
        np.nan if record["cv_cm_s"] is None else record["cv_cm_s"]
        for record in records
    ], dtype=float)
    order = np.argsort(bcl)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(bcl[order], cv[order], "o-", color="#1f77b4", linewidth=1.8,
              markersize=4.5)
    axis.set_xlabel("BCL [ms]")
    axis.set_ylabel("Conduction velocity [cm/s]")
    axis.set_title("BCL–CV relationship (last beat)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None else manifest.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest)
    records = [
        _analyze_run(
            row, args.threshold_mv, args.start_cell,
            args.end_cell, args.cell_length_um,
        )
        for row in rows
    ]
    records.sort(key=lambda record: float(record["bcl_ms"]))

    csv_path = output_dir / "bcl_cv_metrics.csv"
    json_path = output_dir / "bcl_cv_metrics.json"
    plot_path = output_dir / "bcl_vs_cv.png"
    _write_csv(csv_path, records)
    report = {
        "manifest": str(manifest),
        "last_beat_only": True,
        "cv_definition": {
            "method": "two_point_first_upward_threshold_crossing",
            "threshold_mv": args.threshold_mv,
            "start_cell_zero_based": args.start_cell,
            "end_cell_zero_based": args.end_cell,
            "cell_length_um": args.cell_length_um,
        },
        "records": records,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    _plot(plot_path, records)

    print("BCL [ms]   CV [cm/s]   status")
    for record in records:
        cv_text = "--" if record["cv_cm_s"] is None else f"{record['cv_cm_s']:.6f}"
        print(f"{record['bcl_ms']:8g}   {cv_text:>9}   {record['status']}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved PDF : {plot_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
