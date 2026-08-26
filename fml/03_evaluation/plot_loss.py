"""
Plot train / val loss vs epoch from a training log file.

Usage
-----
    python plot_loss.py training_log_20260625_103437
    python plot_loss.py training_log_20260625_103437.txt   # extension optional

The script reads MODEL_NAME_BASE from line 5 of the log, parses all
"Epoch [N/M] | Train: … | Val: …" lines, and saves a semi-log loss
curve to  fml/eval/loss_plot/<MODEL_NAME_BASE>.png
"""

import os
import re
import sys
import argparse

import matplotlib
matplotlib.use("Agg")          # headless: no display needed on cluster
import matplotlib.pyplot as plt

# ── Paths (relative to this file: fml/03_evaluation/) ────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.normpath(os.path.join(_HERE, "..", "logs"))
SAVE_DIR = os.path.normpath(os.path.join(_HERE, "..", "eval", "loss_plot"))


# ── Parser ────────────────────────────────────────────────────────────────────
_RE_MODEL = re.compile(r"MODEL_NAME_BASE:\s*(.+)$")
_RE_EPOCH = re.compile(
    r"Epoch \[(\d+)/\d+\]"
    r" \| Train:\s*([\d.e+\-]+)"
    r" \| Val:\s*([\d.e+\-]+|N/A)"
)


def parse_log(log_path):
    """Return (model_name, epochs, train_losses, val_losses).

    val_losses entries are float or None when Val was N/A.
    """
    model_name  = None
    epochs      = []
    train_losses = []
    val_losses   = []

    with open(log_path, "r") as fh:
        for line in fh:
            line = line.strip()

            if model_name is None:
                m = _RE_MODEL.search(line)
                if m:
                    model_name = m.group(1).strip()
                    continue

            m = _RE_EPOCH.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_str = m.group(3)
                val_losses.append(None if val_str == "N/A" else float(val_str))

    return model_name, epochs, train_losses, val_losses


# ── Plotter ───────────────────────────────────────────────────────────────────
def plot_loss(log_name: str):
    # normalise: strip directory / extension
    log_name = os.path.basename(log_name)
    if not log_name.endswith(".txt"):
        log_name += ".txt"

    log_path = os.path.join(LOG_DIR, log_name)
    if not os.path.exists(log_path):
        sys.exit(f"[ERROR] Log file not found:\n  {log_path}")

    model_name, epochs, train_losses, val_losses = parse_log(log_path)

    if model_name is None:
        sys.exit("[ERROR] MODEL_NAME_BASE not found in log file.")
    if not epochs:
        sys.exit("[ERROR] No epoch lines found in log file.")

    has_val    = any(v is not None for v in val_losses)
    val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
    val_clean  = [v for v in val_losses if v is not None]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(epochs, train_losses,
                color="steelblue", linewidth=1.5, label="Train")
    if has_val:
        ax.semilogy(val_epochs, val_clean,
                    color="tomato", linewidth=1.5, label="Val")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss  (MSE, log scale)", fontsize=12)
    ax.set_title(model_name, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{model_name}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Model   : {model_name}")
    print(f"Epochs  : {epochs[0]} → {epochs[-1]}  ({len(epochs)} logged)")
    print(f"Train   : {train_losses[0]:.3e}  →  {train_losses[-1]:.3e}")
    if has_val and val_clean:
        print(f"Val     : {val_clean[0]:.3e}  →  {val_clean[-1]:.3e}")
    print(f"Saved   → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training loss curve from a log file.")
    parser.add_argument(
        "log_name",
        help="Log filename (with or without .txt extension), "
             "e.g.  training_log_20260625_103437")
    args = parser.parse_args()
    plot_loss(args.log_name)
