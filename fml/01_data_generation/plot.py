"""
Plot helper functions for visualizing training results, such as loss curves and prediction vs. true values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re



def plot_burst_intervals(
    burst_idx_train,
    burst_idx_val,
    burst_idx_test,
    train_len,
    val_len,
    test_len,
    trajectory_length=None,
    max_show=None,
    title="Burst intervals",
    save_path = None
):
    burst_idx_train = np.asarray(burst_idx_train)
    burst_idx_val = np.asarray(burst_idx_val)
    burst_idx_test = np.asarray(burst_idx_test)

    if max_show is not None:
        burst_idx_train = burst_idx_train[:max_show]
        burst_idx_val = burst_idx_val[:max_show]
        burst_idx_test = burst_idx_test[:max_show]

    train_ranges = [(int(s), int(train_len)) for s in burst_idx_train]
    val_ranges = [(int(s), int(val_len)) for s in burst_idx_val]
    test_ranges = [(int(s), int(test_len)) for s in burst_idx_test]

    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.broken_barh(
        train_ranges,
        (30, 8),
        facecolors="tab:blue",
        edgecolors="black",
        alpha=0.8,
        label="Train"
    )

    ax.broken_barh(
        val_ranges,
        (18, 8),
        facecolors="tab:orange",
        edgecolors="black",
        alpha=0.8,
        label="Validation"
    )

    ax.broken_barh(
        test_ranges,
        (6, 8),
        facecolors="tab:green",
        edgecolors="black",
        alpha=0.8,
        label="Test"
    )

    ax.set_yticks([10, 22, 34])
    ax.set_yticklabels(["Test", "Validation", "Train"])

    ax.set_xlabel("Time index")
    ax.set_title(title)

    if trajectory_length is not None:
        ax.set_xlim(0, trajectory_length)

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    if save_path is not None:
        safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        save_file = os.path.join(save_path, f"{safe_title}.png")
        fig.savefig(save_file)
        print(f"Saved burst interval plot to {save_file}")