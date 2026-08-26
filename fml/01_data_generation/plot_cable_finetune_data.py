#!/usr/bin/env python3
"""Create compact review plots for cable fine-tuning NPZ datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reconstruct_stride_one(windows: np.ndarray) -> np.ndarray:
    """Reconstruct (time, junction, channel) from junction-major windows."""
    n_junction = 49
    if windows.shape[0] % n_junction:
        raise ValueError("training sample count is not divisible by 49 junctions")
    n_window = windows.shape[0] // n_junction
    grouped = windows.reshape(n_junction, n_window, windows.shape[1], windows.shape[2])
    # First complete window followed by the new final point from every later window.
    reconstructed = np.concatenate(
        (grouped[:, 0], grouped[:, 1:, :, -1].transpose(0, 2, 1)), axis=2
    )
    return reconstructed.transpose(2, 0, 1)


def style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 140,
        "savefig.dpi": 200,
    })


def plot_training(path: Path, output: Path) -> None:
    data = np.load(path)
    qoi = reconstruct_stride_one(data["qoi"])
    ctrl = reconstruct_stride_one(data["ctrl"])
    dt = float(data["dt_ms"])
    time = np.arange(qoi.shape[0]) * dt
    ctrl_time = np.arange(ctrl.shape[0]) * dt
    shown = (0, qoi.shape[1] // 2, qoi.shape[1] - 1)
    colors = ("#2563eb", "#16a34a", "#dc2626")

    style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_v, ax_i, ax_vh, ax_ih = axes.ravel()
    for junction, color in zip(shown, colors):
        ax_v.plot(ctrl_time, ctrl[:, junction, 0], color=color, lw=1.2,
                  label=fr"$J={junction}$, left")
        ax_v.plot(ctrl_time, ctrl[:, junction, 1], color=color, lw=1.0, ls="--",
                  label=fr"$J={junction}$, right")
        ax_i.plot(time, qoi[:, junction, 0], color=color, lw=1.2,
                  label=fr"$J={junction}$, left")
        ax_i.plot(time, qoi[:, junction, 1], color=color, lw=1.0, ls="--",
                  label=fr"$J={junction}$, right")

    ax_v.set(title="Training control trajectories", xlabel=r"$t$ [ms]",
             ylabel=r"$\phi_{\mathrm{axial}}$ [mV]", xlim=(0, ctrl_time[-1]))
    ax_i.set(title="Training cleft-current trajectories", xlabel=r"$t$ [ms]",
             ylabel=r"$I_{\mathrm{cleft}}$ [$\mu$A]", xlim=(0, time[-1]))
    ax_v.legend(ncol=2, fontsize=7)
    ax_i.legend(ncol=2, fontsize=7)

    im_v = ax_vh.imshow(ctrl[:, :, 0].T, aspect="auto", origin="lower",
                        extent=(0, ctrl_time[-1], 0, qoi.shape[1] - 1), cmap="turbo")
    ax_vh.set(title="All junctions: left-cell voltage", xlabel=r"$t$ [ms]",
              ylabel="Junction index")
    fig.colorbar(im_v, ax=ax_vh, label=r"$\phi$ [mV]")

    magnitude = np.max(np.abs(qoi), axis=2).T
    im_i = ax_ih.imshow(magnitude, aspect="auto", origin="lower",
                        extent=(0, time[-1], 0, qoi.shape[1] - 1), cmap="magma")
    ax_ih.set(title=r"All junctions: $\max(|I^L|,|I^R|)$", xlabel=r"$t$ [ms]",
              ylabel="Junction index")
    fig.colorbar(im_i, ax=ax_ih, label=r"$|I_{\mathrm{cleft}}|$ [$\mu$A]")

    fig.suptitle(
        f"Fine-tune training data: {qoi.shape[1]} junctions, "
        f"{qoi.shape[0]} samples at $\Delta t={dt:g}$ ms"
    )
    fig.savefig(output)
    plt.close(fig)


def plot_test(path: Path, output: Path) -> None:
    data = np.load(path)
    qoi = data["qoi"]
    ctrl = data["ctrl"]
    starts_ms = data["starts_ms"]
    dt = float(data["dt_ms"])
    n_per_junction = starts_ms.shape[1]
    junction_for_row = np.repeat(np.arange(starts_ms.shape[0]), n_per_junction)

    # Show the most dynamically informative bursts rather than accidentally
    # selecting only resting segments.
    activity = np.ptp(ctrl[:, :2], axis=2).max(axis=1)
    selected = np.argsort(activity)[-6:][::-1]
    relative_time = np.arange(qoi.shape[2]) * dt
    ctrl_relative_time = np.arange(ctrl.shape[2]) * dt
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(selected)))

    style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_v, ax_i, ax_s, ax_h = axes.ravel()
    for row, color in zip(selected, colors):
        junction = junction_for_row[row]
        local_index = row % n_per_junction
        start = starts_ms[junction, local_index]
        label = fr"$J={junction}$, start={start:g} ms"
        ax_v.plot(ctrl_relative_time, ctrl[row, 0], color=color, lw=1.2, label=label)
        ax_v.plot(ctrl_relative_time, ctrl[row, 1], color=color, lw=1.0, ls="--")
        ax_i.plot(relative_time, qoi[row, 0], color=color, lw=1.2, label=label)
        ax_i.plot(relative_time, qoi[row, 1], color=color, lw=1.0, ls="--")

    ax_v.set(title="Six most dynamic test bursts (solid left, dashed right)",
             xlabel=r"Burst-relative $t$ [ms]",
             ylabel=r"$\phi_{\mathrm{axial}}$ [mV]")
    ax_i.set(title="Corresponding cleft-current bursts",
             xlabel=r"Burst-relative $t$ [ms]",
             ylabel=r"$I_{\mathrm{cleft}}$ [$\mu$A]")
    ax_v.legend(fontsize=7)
    ax_i.legend(fontsize=7)

    for junction in range(starts_ms.shape[0]):
        ax_s.scatter(starts_ms[junction], np.full(n_per_junction, junction),
                     s=12, color="#2563eb", alpha=0.75)
    ax_s.set(title="Random test-burst starts", xlabel="Beat-relative start [ms]",
             ylabel="Junction index", xlim=(0, float(data["interval_ms"][-1])))
    ax_h.hist(starts_ms.ravel(), bins=20, color="#2563eb", alpha=0.85)
    ax_h.set(title="Start-time distribution", xlabel="Beat-relative start [ms]",
             ylabel="Number of bursts")

    fig.suptitle(
        f"Fine-tune test data: beat {int(data['beat_index'])}, "
        f"{len(qoi)} bursts, length={relative_time[-1]:g} ms"
    )
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/fml_finetune_review"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_plot = args.output_dir / "training_data_review.png"
    test_plot = args.output_dir / "test_data_review.png"
    plot_training(args.train, train_plot)
    plot_test(args.test, test_plot)
    print(train_plot)
    print(test_plot)


if __name__ == "__main__":
    main()
