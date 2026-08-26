"""Plotting utilities for the FML cable simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Numerical_simulation.solvers.core import CableResult


def _selected_cells(n_cells: int, count: int = 6) -> np.ndarray:
    return np.unique(
        np.linspace(0, n_cells - 1, min(count, n_cells), dtype=int)
    )


def _decimate_preserve_extrema(
    t: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample (t, y) for rendering, keeping the largest-magnitude sample
    in every bin so sharp single-sample spikes are not aliased away or
    picked inconsistently between otherwise-identical events (a plain
    stride ``[::k]`` can silently drop or misalign narrow peaks).
    """
    n = len(t)
    if n <= max_points:
        return t, y
    stride = int(np.ceil(n / max_points))
    n_bins = int(np.ceil(n / stride))
    pad = n_bins * stride - n
    if pad:
        t = np.concatenate((t, np.full(pad, t[-1])))
        y = np.concatenate((y, np.full(pad, y[-1])))
    t_bins = t.reshape(n_bins, stride)
    y_bins = y.reshape(n_bins, stride)
    pick = np.argmax(np.abs(y_bins), axis=1)
    row = np.arange(n_bins)
    return t_bins[row, pick], y_bins[row, pick]


def plot_result(result: CableResult, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cells = _selected_cells(result.voltage_mv.shape[1])
    beat_times = np.arange(result.nbeats, dtype=float) * result.bcl_ms

    # Exact 2-D axial-voltage view requested by the simulation workflow:
    # one plot(t, phi_ax) curve per cell, with no cell subsampling.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.time_ms, result.voltage_mv, lw=0.8)
    for beat_time in beat_times:
        ax.axvline(beat_time, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.set(
        xlabel=r"$t\;[\mathrm{ms}]$",
        ylabel=r"$\phi_{\mathrm{ax}}\;[\mathrm{mV}]$",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "phi_ax_2d.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # 3-D axial-voltage surface. Downsample only the rendered time mesh so a
    # multi-beat run remains practical; simulation.npz retains every sample.
    max_surface_times = 2000
    stride = max(1, int(np.ceil(len(result.time_ms) / max_surface_times)))
    time_plot = result.time_ms[::stride]
    voltage_plot = result.voltage_mv[::stride]
    time_mesh, cell_mesh = np.meshgrid(
        time_plot, np.arange(result.voltage_mv.shape[1])
    )
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        time_mesh,
        cell_mesh,
        voltage_plot.T,
        cmap="turbo",
        linewidth=0,
        antialiased=True,
    )
    ax.set(
        xlabel=r"$t\;[\mathrm{ms}]$",
        ylabel=r"$i_{\mathrm{cell}}$",
        zlabel=r"$\phi_{\mathrm{ax}}\;[\mathrm{mV}]$",
    )
    fig.colorbar(
        surface, ax=ax, shrink=0.65, pad=0.1,
        label=r"$\phi_{\mathrm{ax}}\;[\mathrm{mV}]$",
    )
    fig.tight_layout()
    path = output_dir / "phi_ax_3d.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # Icleft(t): separate panels for the two sides of every junction.
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    side_labels = (
        r"$I_{\mathrm{cleft}}^{L}\;[\mu\mathrm{A}]$",
        r"$I_{\mathrm{cleft}}^{R}\;[\mu\mathrm{A}]$",
    )
    n_junctions = result.cleft_current_ua.shape[1]
    plotted_junctions = _selected_cells(n_junctions, count=5)
    colors = plt.cm.viridis(
        np.linspace(0.0, 1.0, len(plotted_junctions))
    )
    # Decimate rendering only (NPZ retains every time sample) by keeping the
    # largest-magnitude sample per bin, so narrow spikes stay a thin, exact
    # line instead of aliasing into an inconsistent or filled-looking trace.
    for side, ax in enumerate(axes):
        for junction, color in zip(plotted_junctions, colors):
            plot_time, plot_current = _decimate_preserve_extrema(
                result.current_time_ms,
                result.cleft_current_ua[:, junction, side],
                max_points=4000,
            )
            ax.plot(
                plot_time,
                plot_current,
                color=color,
                lw=0.7,
                label=rf"$j={junction}$",
            )
        for beat_time in beat_times:
            ax.axvline(beat_time, color="black", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylabel(side_labels[side])
        ax.grid(alpha=0.25)
        ax.legend(ncol=min(5, len(plotted_junctions)), fontsize=8)
    axes[-1].set_xlabel(r"$t\;[\mathrm{ms}]$")
    path = output_dir / "icleft.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    for cell in cells:
        ax.plot(
            result.time_ms,
            result.voltage_mv[:, cell],
            lw=1.1,
            label=rf"$\mathrm{{cell}}\ {cell}$",
        )
    ax.set(
        xlabel=r"$t\;[\mathrm{ms}]$",
        ylabel=r"$\phi_{\mathrm{ax}}\;[\mathrm{mV}]$",
    )
    for beat_time in beat_times:
        ax.axvline(beat_time, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output_dir / "voltage_traces.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        result.voltage_mv.T,
        origin="lower",
        aspect="auto",
        extent=(
            result.time_ms[0],
            result.time_ms[-1],
            0,
            result.voltage_mv.shape[1] - 1,
        ),
        cmap="turbo",
    )
    ax.set(xlabel=r"$t\;[\mathrm{ms}]$", ylabel=r"$i_{\mathrm{cell}}$")
    for beat_time in beat_times:
        ax.axvline(beat_time, color="white", ls=":", lw=0.8, alpha=0.65)
    fig.colorbar(
        image, ax=ax, label=r"$\phi_{\mathrm{ax}}\;[\mathrm{mV}]$"
    )
    fig.tight_layout()
    path = output_dir / "voltage_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for cell in cells:
        axes[0].plot(
            result.current_time_ms,
            result.ionic_current_ua[:, cell],
            lw=0.9,
            label=rf"$\mathrm{{cell}}\ {cell}$",
        )
    junctions = _selected_cells(result.cleft_current_ua.shape[1], count=4)
    for junction in junctions:
        axes[1].plot(
            result.current_time_ms,
            result.cleft_current_ua[:, junction, 0],
            lw=0.9,
            label=rf"$j={junction},\ L$",
        )
        axes[1].plot(
            result.current_time_ms,
            result.cleft_current_ua[:, junction, 1],
            "--",
            lw=0.9,
            label=rf"$j={junction},\ R$",
        )
    axes[0].set_ylabel(r"$I_{\mathrm{ion}}\;[\mu\mathrm{A}]$")
    axes[1].set_ylabel(r"$I_{\mathrm{cleft}}\;[\mu\mathrm{A}]$")
    axes[1].set_xlabel(r"$t\;[\mathrm{ms}]$")
    for ax in axes:
        for beat_time in beat_times:
            ax.axvline(beat_time, color="black", ls=":", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = output_dir / "current_traces.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 4))
    stimulated_cells = np.flatnonzero(
        np.any(result.stimulus_current_ua != 0.0, axis=0)
    )
    for cell in stimulated_cells:
        ax.step(
            result.current_time_ms,
            result.stimulus_current_ua[:, cell],
            where="post",
            label=rf"$\mathrm{{cell}}\ {cell}$",
        )
    ax.set(
        xlabel=r"$t\;[\mathrm{ms}]$",
        ylabel=r"$I_{\mathrm{stim}}\;[\mu\mathrm{A}]$",
    )
    ax.grid(alpha=0.25)
    if stimulated_cells.size:
        ax.legend()
    fig.tight_layout()
    path = output_dir / "stimulus.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # MATLAB terminal boundary-disc state at the two cable ends.
    if result.boundary_voltage_mv.size:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        labels = ("left terminal", "right terminal")
        for side, label in enumerate(labels):
            axes[0].plot(
                result.time_ms,
                result.boundary_voltage_mv[:, side],
                lw=1.0,
                label=label,
            )
            axes[1].plot(
                result.current_time_ms,
                result.boundary_current_ua[:, side],
                lw=1.0,
                label=label,
            )
        axes[0].set_ylabel(r"$\phi_{\mathrm{boundary}}\;[\mathrm{mV}]$")
        axes[1].set_ylabel(r"$I_{\mathrm{boundary}}\;[\mu\mathrm{A}]$")
        axes[1].set_xlabel(r"$t\;[\mathrm{ms}]$")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        path = output_dir / "terminal_boundary.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths
