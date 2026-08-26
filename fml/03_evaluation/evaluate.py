"""
FML Evaluation for Case 3: Van der Pol Oscillator.

Loads trained model, runs autoregressive prediction on test data,
computes metrics, and generates publication-quality plots.

Test NPZ files use aligned Ctrl-now bursts (Ctrl is one sample longer than
QoI). ``MODEL_TYPE`` selects direct FNN prediction or residual ResNet.

Three plot types per test sample:
    *_prediction.png — QoI reference vs FML prediction
    *_control.png    — control signal as piecewise-constant step function
    *_error.png      — point-wise error (semilogy)

Usage:
    python evaluate.py                   # Use best_train model (default)
    python evaluate.py --model best_val  # Use best_val model
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_training'))
from config import Config
from model import FNN


# =========================================================================
# Case-specific settings
# =========================================================================

QOI_LABELS = ['$x_1$ (position)']
CTRL_LABELS = ['$u$ (force)']
CHECKPOINT_KIND_DEFAULT = 'best_train'


# =========================================================================
# Data loading
# =========================================================================

def load_test_data(config):
    """Load test data from .npz file."""
    data = np.load(config.TEST_NPZ)
    required = {"qoi", "ctrl", "dt_ms"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"Unsupported test NPZ; missing {sorted(missing)}. Regenerate it "
            "with the current generator."
        )
    data_dt = float(np.asarray(data["dt_ms"]).item())
    if not np.isclose(data_dt, config.dt, rtol=0.0, atol=1e-12):
        raise ValueError(f"Test NPZ dt={data_dt} does not match config dt={config.dt}")
    qoi_all = torch.tensor(data['qoi'], dtype=config.DATA_TYPE)

    total_len = qoi_all.shape[2]
    n_rec = total_len - config.NUM_MEMORY - 1
    print(f"Test data: {qoi_all.shape[0]} samples, {total_len} steps, n_rec={n_rec}")

    qoi_hist = qoi_all[:, :, :config.NUM_MEMORY + 1]
    future_qoi = qoi_all[:, :, config.NUM_MEMORY + 1:]

    result = {
        'qoi_hist': qoi_hist,
        'ctrl_hist': None,
        'future_tgt': future_qoi,
        'n_rec': n_rec,
        'params': None,
        'Mu': data.get('Mu', None),
    }

    num_ctrl = getattr(config, 'NUM_CTRL', 0)
    if num_ctrl > 0 and 'ctrl' in data:
        ctrl_all = torch.tensor(data['ctrl'], dtype=config.DATA_TYPE)
        if ctrl_all.shape[1] != num_ctrl:
            raise ValueError(
                f"Test NPZ has {ctrl_all.shape[1]} Ctrl channels; "
                f"config expects {num_ctrl}"
            )
        if (ctrl_all.shape[0] != qoi_all.shape[0]
                or ctrl_all.shape[2] != total_len + 1):
            raise ValueError(
                f"Ctrl must be one sample longer than QoI; got "
                f"qoi={tuple(qoi_all.shape)}, ctrl={tuple(ctrl_all.shape)}"
            )
        result['ctrl_hist'] = ctrl_all[:, :, :config.NUM_MEMORY + 2]
        future_ctrl = ctrl_all[:, :, config.NUM_MEMORY + 2:]
        result['future_tgt'] = torch.cat([future_qoi, future_ctrl], dim=1)

    return result


# =========================================================================
# Prediction
# =========================================================================

def predict_long_term(model, initial_qoi_hist, n_steps, config,
                      initial_ctrl_hist=None, future_targets=None,
                      params_input=None):
    """Autoregressive prediction using aligned M+2 Ctrl histories."""
    model.eval()
    current_qoi = initial_qoi_hist.clone().to(config.DEVICE)
    current_ctrl = initial_ctrl_hist.clone().to(config.DEVICE) if initial_ctrl_hist is not None else None
    params_dev = params_input.to(config.DEVICE) if params_input is not None else None

    predictions = []
    with torch.no_grad():
        for t in range(n_steps):
            pred = model(
                current_qoi, current_ctrl,
                params_input=params_dev,
            )
            predictions.append(pred.squeeze(-1).cpu().numpy()[0])

            current_qoi = torch.cat([current_qoi[:, :, 1:], pred], dim=2)

            if current_ctrl is not None:
                next_ctrl = future_targets[:, config.NUM_QOI:, t].to(config.DEVICE)
                current_ctrl = torch.cat(
                    [current_ctrl[:, :, 1:], next_ctrl.unsqueeze(-1)], dim=2
                )

    return np.array(predictions).T


# =========================================================================
# Metrics and plotting
# =========================================================================

def evaluate_prediction(pred, reference):
    """Compute prediction quality metrics."""
    mse = np.mean((pred - reference) ** 2)
    mae = np.mean(np.abs(pred - reference))
    norm_ref = np.linalg.norm(reference)
    rel_l2 = np.linalg.norm(pred - reference) / norm_ref if norm_ref > 0 else float('inf')
    return {'MSE': mse, 'MAE': mae, 'Relative_L2': rel_l2}


def plot_results(t_future, reference, prediction, test_idx, config,
                 ctrl=None, qoi_hist=None, ctrl_hist=None,
                 param_label="", save_dir=None):
    """Generate 3 separate plots: prediction, control, error.

    Shows the initial history window (nM steps) as ground truth,
    followed by the FML prediction region. A vertical dashed line
    marks the boundary where autoregressive prediction begins.
    """
    if save_dir is None:
        save_dir = config.EVAL_DIR
    dataset_tag = getattr(config, 'DATASET', 'unknown')
    if getattr(config, 'IF_TEST_REAL', False):
        save_dir = os.path.join(save_dir, "Cable_dataset", dataset_tag,
                                f"FEM_{config.MESH_IDX}", f"GJ_{config.GJ_COUPLING}",
                                f"BCL_{config.BCL}",
                                f"MEM_LEN_{config.NUM_MEMORY}")
    else:
        save_dir = os.path.join(save_dir, "Test_dataset", dataset_tag,
                                f"FEM_{config.MESH_IDX}", f"GJ_{config.GJ_COUPLING}",
                                f"MEM_LEN_{config.NUM_MEMORY}")
    os.makedirs(save_dir, exist_ok=True)

    num_qoi = config.NUM_QOI
    nM = config.NUM_MEMORY
    dt = config.dt
    labels = QOI_LABELS if len(QOI_LABELS) == num_qoi else [f'QoI {i+1}' for i in range(num_qoi)]
    title_base = f'Test {test_idx + 1}'
    if param_label:
        title_base = f' ({param_label})' + title_base

    # Build time axis with t=0 at the prediction boundary.
    # History:  t in [-(nM)*dt, 0]   (nM+1 points)
    # Future:   t in [dt, n_pred*dt] (n_pred points)
    t_hist = (np.arange(nM + 1) - nM) * dt   # [-nM*dt, ..., 0]
    t_boundary = 0.0                           # prediction starts at t=0
    # t_future is passed in as absolute time; shift so boundary = 0
    t_future_rel = t_future - t_future[0] + dt  # [dt, 2*dt, ..., n_pred*dt]

    # Concatenate history + future into continuous arrays
    if qoi_hist is not None:
        t_full = np.concatenate([t_hist, t_future_rel])
        qoi_ref_full = np.concatenate([qoi_hist, reference], axis=1)
        qoi_pred_full = np.concatenate([qoi_hist, prediction], axis=1)
    else:
        t_full = t_future_rel
        qoi_ref_full = reference
        qoi_pred_full = prediction

    if ctrl_hist is not None and ctrl is not None:
        # Ctrl history includes the first target-time sample (dt), and the
        # recurrent Ctrl tail extends one sample beyond the QoI reference.
        t_ctrl_future = np.arange(1, ctrl.shape[1] + 2) * dt
        t_ctrl_full = np.concatenate([t_hist, t_ctrl_future])
        ctrl_full = np.concatenate([ctrl_hist, ctrl], axis=1)
    elif ctrl is not None:
        t_ctrl_full = t_future_rel
        ctrl_full = ctrl
    else:
        t_ctrl_full = None
        ctrl_full = None

    # --- Plot 1: QoI prediction ---
    fig, axes = plt.subplots(num_qoi, 1, figsize=(12, 3.5 * num_qoi), squeeze=False)
    for q in range(num_qoi):
        ax = axes[q, 0]
        ax.plot(t_full, qoi_ref_full[q, :], 'b-', label='Reference', linewidth=1)
        ax.plot(t_full, qoi_pred_full[q, :], 'r--', label='FML Prediction', linewidth=1)
        ax.axvline(t_boundary, color='black', ls='--', lw=1, alpha=0.8)
        ax.set_ylabel(labels[q])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title(f'$n_M={nM}$  |  history: [{-nM*dt:.1f}, 0] ms  |  prediction: [0, {t_future_rel[-1]:.1f}] ms',
                         fontsize=10, loc='left')
    axes[-1, 0].set_xlabel('Time relative to prediction start (ms)')
    fig.suptitle(f'{title_base} — QoI Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{param_label}_test_{test_idx+1}_prediction.png'), dpi=300)
    plt.close()

    # --- Plot 2: Control signal (step function) ---
    if ctrl is not None:
        num_ctrl = ctrl.shape[0]
        c_labels = CTRL_LABELS if len(CTRL_LABELS) == num_ctrl else [f'$u_{{{c+1}}}$' for c in range(num_ctrl)]
        fig, axes = plt.subplots(num_ctrl, 1, figsize=(12, 3 * num_ctrl), squeeze=False)
        for c in range(num_ctrl):
            ax = axes[c, 0]
            ax.step(t_ctrl_full, ctrl_full[c, :], where='post', color='green', linewidth=1)
            ax.axvline(t_boundary, color='black', ls='--', lw=1, alpha=0.8)
            ax.set_ylabel(c_labels[c])
            ax.grid(True, alpha=0.3)
            if hasattr(config, 'CTRL_BOUNDS'):
                lo, hi = config.CTRL_BOUNDS
                ax.axhline(lo, color='gray', ls='--', lw=0.8, alpha=0.5)
                ax.axhline(hi, color='gray', ls='--', lw=0.8, alpha=0.5)
        axes[0, 0].set_title(f'$n_M={nM}$  |  history: [{-nM*dt:.1f}, 0] ms  |  prediction: [0, {t_future_rel[-1]:.1f}] ms',
                             fontsize=10, loc='left')
        axes[-1, 0].set_xlabel('Time relative to prediction start (ms)')
        fig.suptitle(f'{title_base} — Control Signal', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{param_label}_test_{test_idx+1}_control.png'), dpi=300)
        plt.close()

    # --- Plot 3: Error ---
    fig, axes = plt.subplots(num_qoi, 1, figsize=(12, 3 * num_qoi), squeeze=False)
    for q in range(num_qoi):
        ax = axes[q, 0]
        error = np.abs(prediction[q, :] - reference[q, :])
        ax.semilogy(t_future_rel, error, 'k-', linewidth=0.8)
        ax.set_ylabel(f'|Error| {labels[q]}')
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time relative to prediction start (ms)')
    fig.suptitle(f'{title_base} — Point-wise Error', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{param_label}_test_{test_idx+1}_error.png'), dpi=300)
    plt.close()


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    # allow_abbrev=False: this parser and config.py's parser both scan the same
    # sys.argv independently; with abbreviation matching on, config.py's
    # --model_type can swallow this parser's --model as a "prefix abbreviation"
    # (that's the bug that caused "--model best_val" to error as an invalid
    # --model_type choice). Keep this off here too.
    parser = argparse.ArgumentParser(description='Evaluate FML model on test data',
                                      allow_abbrev=False)
    parser.add_argument('--model', choices=['best_train', 'best_val'],
                        default=CHECKPOINT_KIND_DEFAULT,
                        help='Which saved model to load (default: best_train)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to cable .npz file. If set, overrides config.TEST_NPZ '
                             'and saves plots to Cable_dataset/ subfolder.')
    parser.add_argument('--if_test_real', type=lambda x: x.lower() == 'true', default=False,
                        help='True = use cable data (--test_file required); '
                             'False = use config.TEST_NPZ (default)')
    args, _ = parser.parse_known_args()   # parse_known_args so --memlen passes to Config
    checkpoint_kind = args.model
    config = Config()

    # Fine-tune mode: load the _finetune_BCL<bcl> checkpoint (not the base) and,
    # unless the caller pointed --test_file elsewhere, evaluate on the held-out
    # fine-tune test bursts.
    finetune = getattr(config, 'FINETUNE', False)
    model_name_base = (config.FINETUNE_MODEL_NAME if finetune
                       else config.MODEL_NAME_BASE)

    config.IF_TEST_REAL = args.if_test_real
    if args.if_test_real and args.test_file is not None:
        config.TEST_NPZ = args.test_file
        print(f"[cable mode] TEST_NPZ overridden → {config.TEST_NPZ}")
    elif finetune and args.test_file is None:
        config.TEST_NPZ = config.FINETUNE_TEST_NPZ
        print(f"[fine-tune mode] TEST_NPZ → {config.TEST_NPZ}")

    test_data = load_test_data(config)
    n_test = min(20, test_data['qoi_hist'].shape[0])
    n_rec = test_data['n_rec']

    # Dispatch on config.MODEL_TYPE, mirroring utils.build_model() in
    # 02_training (kept separate here since evaluate.py doesn't import utils.py).
    net_family = getattr(config, 'MODEL_TYPE', 'FNN')
    if net_family in {'FNN', 'ResNet'}:
        model = FNN(config)
    elif net_family == 'LSTM':
        raise NotImplementedError(
            "config.MODEL_TYPE='LSTM' is reserved for a future recurrent "
            "flow-map model; only 'FNN' is implemented so far."
        )
    else:
        raise ValueError(f"Unknown config.MODEL_TYPE: {net_family!r}")
    model = model.to(config.DEVICE)

    model_path = os.path.join(
        config.MODEL_DIR,
        f'{model_name_base}_{checkpoint_kind}.pth'
    )
    try:
        state_dict = torch.load(model_path, map_location=config.DEVICE,
                                weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model: {model_path}")
    print(f"Model type: {net_family}  |  Ctrl contract: aligned M+2")

    dt = config.dt
    nM = config.NUM_MEMORY
    t_future = (nM + 1 + np.arange(n_rec)) * dt  # absolute time
    total_prediction_time = 0.0
    for i in range(n_test):
        print(f"\n--- Test {i+1}/{n_test} ---")
        start_time = time.perf_counter()
        qoi_hist_i = test_data['qoi_hist'][i:i+1]
        future_tgt_i = test_data['future_tgt'][i:i+1]
        ctrl_hist_i = test_data['ctrl_hist'][i:i+1] if test_data['ctrl_hist'] is not None else None

        pred = predict_long_term(
            model, qoi_hist_i, n_rec, config,
            initial_ctrl_hist=ctrl_hist_i,
            future_targets=future_tgt_i
        )

        ref = test_data['future_tgt'][i, :config.NUM_QOI, :].numpy()

        # Extract control for plotting
        ctrl_i = None
        if test_data['future_tgt'].shape[1] > config.NUM_QOI:
            ctrl_i = test_data['future_tgt'][i, config.NUM_QOI:, :].numpy()

        # Extract history for plotting
        qoi_hist_np = test_data['qoi_hist'][i, :config.NUM_QOI, :].numpy()
        ctrl_hist_np = test_data['ctrl_hist'][i].numpy() if test_data['ctrl_hist'] is not None else None

        metrics = evaluate_prediction(pred, ref)

        elapsed = time.perf_counter() - start_time
        total_prediction_time += elapsed
        print(f"  MSE: {metrics['MSE']:.6e}, MAE: {metrics['MAE']:.6e}, "
              f"Rel L2: {metrics['Relative_L2']:.6e}"
              f"time {elapsed:.6f} s")
        bcl_tag = f"_BCL_{config.BCL}" if config.IF_TEST_REAL else ""
        param_label = (f"FEM_{config.MESH_IDX}_GJ_{config.GJ_COUPLING}"
                       f"{bcl_tag}_nm{nM}_nr{config.NUM_RECURRENT}_"
                       f"dt{config._DT_TAG}_{net_family}")
        plot_results(t_future, ref, pred, i, config, ctrl=ctrl_i,
                     qoi_hist=qoi_hist_np, ctrl_hist=ctrl_hist_np,param_label=param_label)

    print("\n=== Evaluation Complete ===")
    print(f"\nTotal prediction time: {total_prediction_time:.6f} seconds")
    print(f"Average prediction time: {total_prediction_time / n_test:.6f} seconds/test")
