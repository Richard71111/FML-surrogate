"""
Training utilities for Flow Map Learning.

Supports all four FML input combinations (determined by config):
  (i)   QoI only            NUM_CTRL=0, NUM_PARAMS=0
  (ii)  QoI + Parameters    NUM_CTRL=0, NUM_PARAMS>0
  (iii) QoI + Ctrl          NUM_CTRL>0, NUM_PARAMS=0
  (iv)  QoI + Ctrl + Params NUM_CTRL>0, NUM_PARAMS>0

Every NPZ uses aligned Ctrl-now bursts: Ctrl has one more sample than QoI.

Contains: logging, data loading, DataLoader creation, optimizer/scheduler
builders, loss functions (MSE / WeightedMSE), recurrent training step,
train/validate epoch, checkpoint saving.
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import FNN


# ===== Loss functions =====

class WeightedMSELoss(nn.Module):
    """Per-component weighted MSE loss for multi-QoI training."""
    def __init__(self, weights, dtype=torch.float64):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        weighted = diff_sq * self.weights.unsqueeze(0)
        return weighted.mean()


def build_criterion(config: Config) -> nn.Module:
    """Factory: return WeightedMSELoss if LOSS_WEIGHTS is set, else MSELoss."""
    weights = getattr(config, 'LOSS_WEIGHTS', None)
    if weights is not None:
        criterion = WeightedMSELoss(weights, dtype=config.DATA_TYPE).to(config.DEVICE)
        print(f"Using WeightedMSELoss with weights={weights}")
        return criterion
    return nn.MSELoss()


# ===== Logging =====

def setup_logging(config: Config) -> str:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # MODEL_TAG must be part of the stem: replicate runs of one shape start
    # within the same second, and write_log appends, so without it several
    # runs interleave into a single unreadable file.
    model_tag = getattr(config, "MODEL_TAG", "")
    shape_tag = (f"{config.DATASET}_nm{config.NUM_MEMORY}_"
                 f"nr{config.NUM_RECURRENT}_dt{config._DT_TAG}"
                 + (f"_{model_tag}" if model_tag else ""))
    run_kind = (
        f"fine_tune_BCL{config.BCL}"
        if getattr(config, "FINETUNE", False)
        else "training"
    )
    log_filepath = os.path.join(
        config.LOG_DIR,
        f'{run_kind}_log_{shape_tag}_{config.MODEL_TYPE}_{timestamp}.txt'
    )
    print(f"Logging to: {log_filepath}")
    return log_filepath


def write_log(log_filepath: str, message: str, console_print: bool = False):
    entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        with open(log_filepath, 'a') as f:
            f.write(entry + '\n')
        if console_print:
            print(entry)
    except Exception as e:
        print(f"Log write error ({log_filepath}): {e}")
        if console_print:
            print(entry)


def log_configuration(log_filepath: str, config: Config):
    if getattr(config, "FINETUNE", False):
        write_log(log_filepath, "=== FINE-TUNE TRAINING LOG ===")
        write_log(log_filepath, "--- Fine-tuning Configuration ---")
        write_log(log_filepath,
                  f"  RUN_TYPE: fine_tune")
        write_log(log_filepath,
                  f"  FINETUNE_BASE_CHECKPOINT: "
                  f"{config.FINETUNE_BASE_CHECKPOINT}")
        write_log(log_filepath,
                  f"  FINETUNE_MODEL_NAME: {config.FINETUNE_MODEL_NAME}")
    else:
        write_log(log_filepath, "=== BASE TRAINING LOG ===")
        write_log(log_filepath, "--- Training Configuration ---")
    for key, value in vars(config).items():
        if not key.startswith('_') and not isinstance(value, (torch.Tensor, np.ndarray)):
            write_log(log_filepath, f"  {key}: {value}")
    write_log(log_filepath, f"  MODEL_NAME_BASE: {config.MODEL_NAME_BASE}")
    write_log(log_filepath, f"  MESH_IDX: {config.MESH_IDX}")
    write_log(log_filepath, f"  GJ_COUPLING: {config.GJ_COUPLING}")
    write_log(log_filepath, f"  NUM_MEMORY: {config.NUM_MEMORY}")
    write_log(log_filepath, f"  NUM_RECURRENT: {config.NUM_RECURRENT}")
    write_log(log_filepath, f"  NUM_VAL_RECURRENT: {config.NUM_VAL_RECURRENT}")
    write_log(log_filepath, f"  NUM_QOI: {config.NUM_QOI}")
    write_log(log_filepath, f"  NUM_CTRL: {config.NUM_CTRL}")
    write_log(log_filepath, f"  DATASET: {config.DATASET}")
    write_log(log_filepath, f"  DT_FML: {config.dt}")
    write_log(log_filepath, f"  MODEL_TYPE: {config.MODEL_TYPE}")
    write_log(log_filepath, f"  HIDDEN_LAYER_SIZES: {config.HIDDEN_LAYER_SIZES}")
    write_log(log_filepath, "  CTRL_CONTRACT: aligned Ctrl length = QoI length + 1")
    write_log(log_filepath, "------------------------------")
    write_log(log_filepath, "--- Training Start ---")


# ===== Data loading =====

def load_data(config: Config, log_filepath: str,
              is_validation: bool = False):
    """Load the generated NPZ directly for both training and validation."""
    path = config.VAL_NPZ if is_validation else config.TRAIN_FILE
    if is_validation and not config.VAL_DATA_AVAILABLE:
        return None
    try:
        write_log(log_filepath, f"Loading {'validation' if is_validation else 'training'} NPZ: {path}")
        return load_npz_bursts(path, config, as_training=not is_validation)
    except FileNotFoundError:
        if is_validation:
            config.VAL_DATA_AVAILABLE = False
            write_log(log_filepath, f"Warning: validation file not found: {path}")
            return None
        raise


# ===== Fine-tuning helpers =====

def resolve_model_name(config: Config) -> str:
    """Checkpoint name stem: the fine-tune name when FINETUNE, else the base.

    Every stage (save here, load in evaluate.py) routes its checkpoint path
    through this one function so the fine-tuned model lives in its own files
    (MODEL_NAME_BASE + _finetune_BCL<bcl>) and never overwrites the base.
    """
    if getattr(config, 'FINETUNE', False):
        return config.FINETUNE_MODEL_NAME
    return config.MODEL_NAME_BASE


def load_npz_bursts(npz_path: str, config: Config, as_training: bool):
    """Load an NPZ burst file (QoI [N,Q,L], Ctrl [N,C,L+1]) into the
    same dict shape `load_data` returns.

    The fine-tune generator writes every split (train/val/test) in this one
    format: a stack of equal-length bursts, history in [:, :, :nM+1] and the
    future/recurrent target in the remainder. This is exactly the layout the
    validation branch of `load_data` already consumes, so fine-tune *training*
    reads the same way -- only the source NPZ differs.

    as_training=True  -> exactly config.NUM_RECURRENT target steps.
    as_training=False -> exactly config.NUM_VAL_RECURRENT target steps.
    """
    num_ctrl = getattr(config, 'NUM_CTRL', 0)
    data = np.load(npz_path)
    required = {"qoi", "ctrl", "dt_ms"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(
            f"{npz_path} is an old/unsupported NPZ; missing {sorted(missing)}. "
            "Regenerate it with the current data generator."
        )
    data_dt = float(np.asarray(data["dt_ms"]).item())
    if not np.isclose(data_dt, config.dt, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"NPZ dt={data_dt} ms does not match configured dt={config.dt} ms"
        )

    qoi_all = torch.tensor(data['qoi'], dtype=config.DATA_TYPE)

    total_len = qoi_all.shape[2]
    n_rec = total_len - config.NUM_MEMORY - 1
    if n_rec <= 0:
        raise ValueError(
            f"burst length {total_len} too short for memory {config.NUM_MEMORY} "
            f"in {npz_path}")
    expected_n_rec = (
        config.NUM_RECURRENT if as_training else config.NUM_VAL_RECURRENT
    )
    if n_rec != expected_n_rec:
        split = "training" if as_training else "validation"
        raise ValueError(
            f"{split} NPZ gives {n_rec} recurrent steps but expected "
            f"{expected_n_rec}; regenerate {npz_path} with the current "
            f"fixed-step dataset configuration")
    if "num_memory" in data.files:
        data_nm = int(np.asarray(data["num_memory"]).item())
        if data_nm != config.NUM_MEMORY:
            raise ValueError(
                f"NPZ num_memory={data_nm} does not match "
                f"NUM_MEMORY={config.NUM_MEMORY}"
            )
    if "mesh_idx" in data.files:
        data_mesh = int(np.asarray(data["mesh_idx"]).item())
        if data_mesh != config.MESH_IDX:
            raise ValueError(
                f"NPZ mesh_idx={data_mesh} does not match MESH_IDX="
                f"{config.MESH_IDX}: {npz_path}"
            )
    if "gj_coupling" in data.files:
        data_gj = str(np.asarray(data["gj_coupling"]).item()).lower()
        if data_gj != str(config.GJ_COUPLING).lower():
            raise ValueError(
                f"NPZ gj_coupling={data_gj!r} does not match GJ_COUPLING="
                f"{config.GJ_COUPLING!r}: {npz_path}"
            )
    if "num_recurrent" in data.files:
        data_nrec = int(np.asarray(data["num_recurrent"]).item())
        if data_nrec != n_rec:
            raise ValueError(
                f"NPZ metadata num_recurrent={data_nrec} does not match "
                f"its burst shape ({n_rec} recurrent steps)"
            )

    qoi_hist   = qoi_all[:, :, :config.NUM_MEMORY + 1]
    future_qoi = qoi_all[:, :, config.NUM_MEMORY + 1:
                                config.NUM_MEMORY + 1 + n_rec]

    ctrl_hist = None
    if num_ctrl > 0:
        ctrl_all = torch.tensor(data['ctrl'], dtype=config.DATA_TYPE)
        if ctrl_all.shape[1] != num_ctrl:
            raise ValueError(
                f"NPZ has {ctrl_all.shape[1]} Ctrl channels; config expects {num_ctrl}"
            )
        if (ctrl_all.shape[0] != qoi_all.shape[0]
                or ctrl_all.shape[2] != total_len + 1):
            raise ValueError(
                f"Ctrl must be one sample longer than QoI; got "
                f"qoi={tuple(qoi_all.shape)}, ctrl={tuple(ctrl_all.shape)}"
            )
        ctrl_hist = ctrl_all[:, :, :config.NUM_MEMORY + 2]
        future_ctrl = ctrl_all[:, :, config.NUM_MEMORY + 2:
                                      config.NUM_MEMORY + 2 + n_rec]
        future_tgt = torch.cat([future_qoi, future_ctrl], dim=1)
    else:
        future_tgt = future_qoi

    return {
        'qoi_hist': qoi_hist,
        'ctrl_hist': ctrl_hist,
        'future_tgt': future_tgt,
        'n_rec': None if as_training else n_rec,
        'params': None,
    }


def load_finetune_data(config: Config, log_filepath: str,
                       is_validation: bool = False):
    """Fine-tune counterpart of `load_data`: real-AP cable bursts from NPZ.

    Training reads FINETUNE_TRAIN_NPZ (dense sliding AP windows); validation
    reads FINETUNE_VAL_NPZ (longer random bursts from a different beat).
    """
    if is_validation:
        path = config.FINETUNE_VAL_NPZ
        if not os.path.isfile(path):
            write_log(log_filepath,
                      f"Fine-tune validation file not found: {path}")
            config.VAL_DATA_AVAILABLE = False
            return None
        write_log(log_filepath, f"Fine-tune validation data: {path}")
        return load_npz_bursts(path, config, as_training=False)

    path = config.FINETUNE_TRAIN_NPZ
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fine-tune training data not found: {path}\n"
            f"Generate it first with generate_cable_finetune_data.py "
            f"(same --memlen/--dataset/--bcl/--dt values).")
    write_log(log_filepath, f"Fine-tune training data: {path}")
    return load_npz_bursts(path, config, as_training=True)


def subset_burst_data(data, max_samples: int, seed: int):
    """Return a deterministic tensor subset without changing burst layout."""
    if data is None or max_samples <= 0:
        return data
    n_samples = int(data['qoi_hist'].shape[0])
    if n_samples <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    index = torch.as_tensor(
        rng.choice(n_samples, max_samples, replace=False), dtype=torch.long
    )
    return {
        key: (value[index] if isinstance(value, torch.Tensor) else value)
        for key, value in data.items()
    }


def mix_replay_data(primary, replay, replay_fraction: float, seed: int):
    """Mix real-AP fine-tune bursts with synthetic solver replay bursts."""
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("FINETUNE_REPLAY_FRACTION must lie in [0, 1)")
    if replay_fraction == 0.0:
        return primary, 0
    n_primary = int(primary['qoi_hist'].shape[0])
    n_replay_available = int(replay['qoi_hist'].shape[0])
    n_replay = int(round(n_primary * replay_fraction / (1.0 - replay_fraction)))
    rng = np.random.default_rng(seed)
    index = torch.as_tensor(
        rng.choice(n_replay_available, n_replay,
                   replace=n_replay > n_replay_available),
        dtype=torch.long,
    )
    mixed = {}
    for key in primary:
        left = primary[key]
        right = replay.get(key)
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor):
                raise ValueError(f"replay data is missing tensor field {key!r}")
            mixed[key] = torch.cat((left, right[index]), dim=0)
        else:
            mixed[key] = left
    return mixed, n_replay


def qoi_target_energy(data, config: Config) -> float:
    """Mean squared QoI magnitude used to normalize validation losses."""
    target = data['future_tgt'][:, :config.NUM_QOI]
    return max(float(torch.mean(target * target).item()),
               np.finfo(np.float64).tiny)


def load_base_checkpoint(model: nn.Module, config: Config,
                         log_filepath: str) -> None:
    """Load the pretrained base weights into `model` before fine-tuning.

    The base and fine-tune models share the identical architecture (same
    MODEL_NAME_BASE), so this is a plain state_dict load -- no remapping.
    """
    path = config.FINETUNE_BASE_CHECKPOINT
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fine-tune base checkpoint not found: {path}\n"
            f"Train the base model first (or set --finetune_base to an "
            f"existing checkpoint).")
    try:
        state_dict = torch.load(path, map_location=config.DEVICE,
                                weights_only=True)
    except TypeError:  # older PyTorch without weights_only
        state_dict = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    write_log(log_filepath, f"Loaded base checkpoint for fine-tuning: {path}")


def create_dataloader(qoi_history: torch.Tensor,
                      future_targets: torch.Tensor,
                      config: Config,
                      ctrl_history: torch.Tensor = None,
                      params: torch.Tensor = None) -> DataLoader:
    """Create DataLoader with consistent packing order.

    Packing order: [qoi, ctrl (if any), target, params (if any)]
    """
    tensors = [qoi_history]
    if ctrl_history is not None:
        tensors.append(ctrl_history)
    tensors.append(future_targets)
    if params is not None:
        tensors.append(params)

    dataset = TensorDataset(*tensors)
    is_cuda = config.DEVICE.type == 'cuda'
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=is_cuda,
        num_workers=config.NUM_WORKERS if is_cuda else 0,
    )


# ===== Model / optimizer builders =====

def build_model(config: Config, log_filepath: str) -> nn.Module:
    """Single dispatch point for config.MODEL_TYPE.

    FNN and ResNet share the same MLP implementation and differ only in the
    output skip connection. Add future families (for example LSTM) here.
    """
    model_type = getattr(config, 'MODEL_TYPE', 'FNN')
    if model_type in {'FNN', 'ResNet'}:
        model = FNN(config)
    elif model_type == 'LSTM':
        raise NotImplementedError(
            "config.MODEL_TYPE='LSTM' is reserved for a future recurrent "
            "flow-map model; only 'FNN' is implemented so far."
        )
    else:
        raise ValueError(f"Unknown config.MODEL_TYPE: {model_type!r}")

    model = model.to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(log_filepath, f"Model type: {model_type}")
    write_log(log_filepath, f"Model initialized -- {n_params} trainable parameters.")
    write_log(log_filepath, f"Model architecture:\n{model}")
    # print(model)
    return model


def build_optimizer_scheduler(model: nn.Module, config: Config,
                              lr: float = None, gamma: float = None):
    """Adam + exponential LR decay.

    `lr`/`gamma` default to the from-scratch config values; the fine-tune
    entry point passes the smaller FINETUNE_LEARNING_RATE / FINETUNE_LR_GAMMA.
    """
    lr = config.LEARNING_RATE if lr is None else lr
    gamma = config.LR_GAMMA if gamma is None else gamma
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return optimizer, scheduler


def load_latest_training_checkpoint(model: nn.Module, optimizer,
                                    scheduler, config: Config,
                                    log_filepath: str):
    """Resume from the newest periodic checkpoint when explicitly enabled."""
    defaults = (0, {'train': float('inf'), 'val': float('inf')}, [], [])
    if not getattr(config, "RESUME_TRAINING", False):
        return defaults
    model_name = resolve_model_name(config)
    pattern = re.compile(rf"^{re.escape(model_name)}_epoch(\d+)\.pth$")
    candidates = []
    for filename in os.listdir(config.MODEL_DIR):
        match = pattern.match(filename)
        if match:
            candidates.append((int(match.group(1)), filename))
    if not candidates:
        write_log(log_filepath, "Resume requested; no periodic checkpoint found.")
        return defaults
    _, filename = max(candidates)
    path = os.path.join(config.MODEL_DIR, filename)
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = int(checkpoint['epoch'])
    best_losses = {
        'train': float(checkpoint.get('best_train_loss', float('inf'))),
        'val': float(checkpoint.get('best_val_loss', float('inf'))),
    }
    train_history = list(checkpoint.get('train_loss_history', []))
    val_history = list(checkpoint.get('val_loss_history', []))
    write_log(log_filepath, f"Resumed training at epoch {start_epoch}: {path}")
    return start_epoch, best_losses, train_history, val_history


# ===== Training logic =====

def run_recurrent_steps(model: nn.Module,
                        initial_qoi_hist: torch.Tensor,
                        future_targets: torch.Tensor,
                        num_recurrent_steps: int,
                        criterion: nn.Module,
                        config: Config,
                        is_training: bool,
                        initial_ctrl_hist: torch.Tensor = None,
                        params_input: torch.Tensor = None) -> torch.Tensor:
    """Recurrent prediction loop.

    For non-autonomous systems (NUM_CTRL>0):
      - future_targets has shape [B, Q+C, T] (QoI + ctrl stacked)
      - ctrl history is updated with GROUND TRUTH future ctrl values
      - qoi history is updated with model PREDICTIONS
      - Ctrl history has M+2 entries; its last entry is target-time Ctrl

    Parameters are time-independent and passed unchanged at every step.
    """
    num_ctrl = getattr(config, 'NUM_CTRL', 0)
    current_qoi_hist = initial_qoi_hist.clone()
    current_ctrl_hist = initial_ctrl_hist.clone() if initial_ctrl_hist is not None else None
    step_losses = []

    for t in range(num_recurrent_steps):
        pred = model(
            current_qoi_hist, current_ctrl_hist,
            params_input=params_input,
        )  # [B, Q, 1]
        target_qoi = future_targets[:, :config.NUM_QOI, t]               # [B, Q]
        step_losses.append(criterion(pred.squeeze(-1), target_qoi))

        # --- Update History for Next Step ---
        pred_for_hist = pred if is_training else pred.detach()
        current_qoi_hist = torch.cat(
            [current_qoi_hist[:, :, 1:], pred_for_hist], dim=2
        )

        if num_ctrl > 0 and current_ctrl_hist is not None:
            next_ctrl = future_targets[:, config.NUM_QOI:, t]
            current_ctrl_hist = torch.cat(
                [current_ctrl_hist[:, :, 1:], next_ctrl.unsqueeze(-1)], dim=2
            )

    return torch.stack(step_losses).mean()


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: optim.Optimizer, criterion: nn.Module,
                config: Config) -> float:
    """Run a single training epoch."""
    model.train()
    num_ctrl = getattr(config, 'NUM_CTRL', 0)
    num_params = getattr(config, 'NUM_PARAMS', 0)
    running_loss = 0.0

    for batch_data in loader:
        # Unpack in same order as create_dataloader packing:
        #   [qoi, ctrl (if any), target, params (if any)]
        idx = 0
        qoi_batch = batch_data[idx].to(config.DEVICE); idx += 1

        ctrl_batch = None
        if num_ctrl > 0:
            ctrl_batch = batch_data[idx].to(config.DEVICE); idx += 1

        tgt_batch = batch_data[idx].to(config.DEVICE); idx += 1

        params_batch = None
        if num_params > 0 and idx < len(batch_data):
            params_batch = batch_data[idx].to(config.DEVICE); idx += 1

        optimizer.zero_grad()
        loss = run_recurrent_steps(
            model, qoi_batch, tgt_batch,
            config.NUM_RECURRENT, criterion, config, is_training=True,
            initial_ctrl_hist=ctrl_batch, params_input=params_batch
        )
        loss.backward()
        if config.CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.CLIP_GRAD_NORM
            )
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def validate_epoch(model: nn.Module, val_data, criterion: nn.Module,
                   config: Config) -> Optional[float]:
    """Validate on long trajectory(ies).

    val_data is a dict with keys: qoi_hist, ctrl_hist, future_tgt, n_rec, params
    """
    if val_data is None:
        return None
    model.eval()

    qoi_dev = val_data['qoi_hist'].to(config.DEVICE)
    tgt_dev = val_data['future_tgt'].to(config.DEVICE)
    n_rec   = val_data['n_rec']

    ctrl_dev = None
    if val_data['ctrl_hist'] is not None:
        ctrl_dev = val_data['ctrl_hist'].to(config.DEVICE)

    params_dev = None
    if val_data['params'] is not None:
        params_dev = val_data['params'].to(config.DEVICE)

    with torch.no_grad():
        loss = run_recurrent_steps(
            model, qoi_dev, tgt_dev,
            n_rec, criterion, config, is_training=False,
            initial_ctrl_hist=ctrl_dev, params_input=params_dev
        )
    return loss.item()


# ===== Checkpointing =====

def _atomic_torch_save(payload, path: str) -> None:
    """Publish a checkpoint atomically so snapshot jobs never copy half a file."""
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)

def save_checkpoint_and_best(epoch, model, optimizer, scheduler,
                             train_loss, val_loss,
                             best_losses, config, log_filepath,
                             train_loss_history, val_loss_history):
    # Route through resolve_model_name so fine-tuning writes its own
    # _finetune_BCL<bcl> files and never clobbers the base checkpoint.
    model_name = resolve_model_name(config)
    if train_loss < best_losses['train']:
        best_losses['train'] = train_loss
        path = os.path.join(config.MODEL_DIR,
                            f'{model_name}_best_train.pth')
        _atomic_torch_save(model.state_dict(), path)
        msg = f"  -> Best train model (loss={train_loss:.10f})"
        if val_loss is not None:
            msg += f" val={val_loss:.10f}"
        write_log(log_filepath, msg)

    if val_loss is not None and val_loss < best_losses['val']:
        best_losses['val'] = val_loss
        path = os.path.join(config.MODEL_DIR,
                            f'{model_name}_best_val.pth')
        _atomic_torch_save(model.state_dict(), path)
        write_log(log_filepath,
                  f"  -> Best val model (val={val_loss:.10f}, train={train_loss:.10f})")

    if (epoch + 1) % config.CHECKPOINT_EPOCH_INTERVAL == 0:
        ckpt_path = os.path.join(config.MODEL_DIR,
                                 f'{model_name}_epoch{epoch+1}.pth')
        _atomic_torch_save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_train_loss': best_losses['train'],
            'best_val_loss': best_losses['val'],
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }, ckpt_path)
        write_log(log_filepath, f"  -> Checkpoint saved at epoch {epoch+1}")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_completion_manifest(config: Config, num_epochs: int,
                              best_losses, log_filepath: str) -> str:
    """Atomically mark only a genuinely completed training run as usable."""
    model_name = resolve_model_name(config)
    checkpoints = {}
    for kind in ("best_train", "best_val"):
        path = os.path.join(config.MODEL_DIR, f"{model_name}_{kind}.pth")
        if os.path.isfile(path):
            checkpoints[kind] = {"path": path, "sha256": _sha256(path)}
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "run_type": "fine_tune" if getattr(config, "FINETUNE", False) else "base",
        "model_name": model_name,
        "mesh_idx": int(config.MESH_IDX),
        "gj_coupling": str(config.GJ_COUPLING),
        "num_memory": int(config.NUM_MEMORY),
        "num_recurrent": int(config.NUM_RECURRENT),
        "dt_fml_ms": float(config.dt),
        "epochs_completed": int(num_epochs),
        "best_losses": {key: float(value) for key, value in best_losses.items()},
        "checkpoints": checkpoints,
        "log_file": log_filepath,
    }
    if getattr(config, "FINETUNE", False):
        base = config.FINETUNE_BASE_CHECKPOINT
        payload["base_checkpoint"] = {
            "path": base,
            "sha256": _sha256(base),
        }
        payload["replay_fraction"] = float(config.FINETUNE_REPLAY_FRACTION)
    path = os.path.join(config.MODEL_DIR, f"{model_name}_complete.json")
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    write_log(log_filepath, f"Completion manifest: {path}")
    return path
