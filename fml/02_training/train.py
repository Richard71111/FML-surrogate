"""
Main FML training script.

Supports all four FML input combinations:
  (i)   QoI only            (autonomous ODE, no known params)
  (ii)  QoI + Parameters    (autonomous ODE, known params fed to NN)
  (iii) QoI + Ctrl          (non-autonomous PDE with ctrl/source)
  (iv)  QoI + Ctrl + Params (non-autonomous PDE with known params)

Usage:
    python train.py            (local)
    sbatch run_training.sh     (SLURM cluster)

Workflow:
  1. Load a generated NPZ and validate its explicit Ctrl contract
  2. Optionally load validation .npz data
  3. Train FNN with recurrent loss over n_R steps
  4. Save best-train, best-val, and periodic checkpoints
"""
import time
import torch

from config import Config
import utils


if __name__ == "__main__":
    config = Config()
    log_filepath = utils.setup_logging(config)
    utils.log_configuration(log_filepath, config)

    try:
        # Weight init and DataLoader shuffling are the only stochastic parts of
        # a run. Seeding them makes a shape reproducible; leaving TORCH_SEED
        # unset keeps the historical random behaviour.
        if getattr(config, 'TORCH_SEED', None) is not None:
            torch.manual_seed(config.TORCH_SEED)
            utils.write_log(log_filepath,
                            f"TORCH_SEED: {config.TORCH_SEED}")
        else:
            utils.write_log(log_filepath,
                            "TORCH_SEED: unset (weight init and shuffling "
                            "differ between runs of the same shape)")

        finetune = getattr(config, 'FINETUNE', False)
        train_loss_history = []
        val_loss_history   = []

        # --- Load data (returns dict) ---
        fine_val_data = None
        replay_val_data = None
        if finetune:
            utils.write_log(log_filepath,
                            "=== FINE-TUNING MODE ===")
            utils.write_log(log_filepath,
                            f"base checkpoint : {config.FINETUNE_BASE_CHECKPOINT}")
            utils.write_log(log_filepath,
                            f"save as         : {config.FINETUNE_MODEL_NAME}")
            utils.write_log(log_filepath, "Loading fine-tune training data...")
            fine_train_data = utils.load_finetune_data(
                config, log_filepath, is_validation=False
            )
            utils.write_log(log_filepath,
                            "Loading synthetic solver replay data...")
            replay_train_data = utils.load_data(
                config, log_filepath, is_validation=False
            )
            train_data, replay_count = utils.mix_replay_data(
                fine_train_data, replay_train_data,
                config.FINETUNE_REPLAY_FRACTION, config.DATA_SEED,
            )
            utils.write_log(
                log_filepath,
                f"Fine-tune mix: {fine_train_data['qoi_hist'].shape[0]} cable "
                f"+ {replay_count} sine replay samples "
                f"(requested replay fraction="
                f"{config.FINETUNE_REPLAY_FRACTION:.3f})",
            )
        else:
            utils.write_log(log_filepath, "Loading training data...")
            train_data = utils.load_data(config, log_filepath, is_validation=False)

        train_loader = utils.create_dataloader(
            train_data['qoi_hist'], train_data['future_tgt'], config,
            ctrl_history=train_data['ctrl_hist'],
            params=train_data['params']
        )
        utils.write_log(log_filepath,
                        f"Training DataLoader: {len(train_loader.dataset)} samples.")
        utils.write_log(log_filepath,
                        f"Scaling status {config.SCALE_DATA}")

        utils.write_log(log_filepath, "Loading validation data...")
        if finetune:
            fine_val_data = utils.load_finetune_data(
                config, log_filepath, is_validation=True
            )
            replay_val_data = utils.subset_burst_data(
                utils.load_data(config, log_filepath, is_validation=True),
                config.FINETUNE_REPLAY_VAL_SAMPLES,
                config.DATA_SEED + 1,
            )
            val_data = fine_val_data
            fine_val_energy = utils.qoi_target_energy(fine_val_data, config)
            replay_val_energy = utils.qoi_target_energy(replay_val_data, config)
            utils.write_log(
                log_filepath,
                "Fine-tune validation uses a normalized composite score: "
                f"cable_energy={fine_val_energy:.3e}, "
                f"sine_energy={replay_val_energy:.3e}",
            )
        else:
            val_data = utils.load_data(config, log_filepath, is_validation=True)
        if val_data:
            utils.write_log(log_filepath, "Validation data loaded.")
        else:
            utils.write_log(log_filepath, "Validation data not available.")

        # --- Build model ---
        model = utils.build_model(config, log_filepath)
        if finetune:
            # Start from the pretrained weights, not random init.
            utils.load_base_checkpoint(model, config, log_filepath)
            optimizer, scheduler = utils.build_optimizer_scheduler(
                model, config,
                lr=config.FINETUNE_LEARNING_RATE,
                gamma=config.FINETUNE_LR_GAMMA,
            )
            num_epochs = config.FINETUNE_NUM_EPOCHS
        else:
            optimizer, scheduler = utils.build_optimizer_scheduler(model, config)
            num_epochs = config.NUM_EPOCHS
        criterion = utils.build_criterion(config)

        start_epoch, best_losses, train_loss_history, val_loss_history = (
            utils.load_latest_training_checkpoint(
                model, optimizer, scheduler, config, log_filepath
            )
        )
        if start_epoch > num_epochs:
            raise ValueError(
                f"resume checkpoint epoch {start_epoch} exceeds requested "
                f"NUM_EPOCHS={num_epochs}"
            )

        # --- Training loop ---
        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()

            epoch_train_loss = utils.train_epoch(
                model, train_loader, optimizer, criterion, config
            )
            if finetune:
                fine_val_raw = utils.validate_epoch(
                    model, fine_val_data, criterion, config
                )
                replay_val_raw = utils.validate_epoch(
                    model, replay_val_data, criterion, config
                )
                replay_weight = config.FINETUNE_REPLAY_FRACTION
                epoch_val_loss = (
                    (1.0 - replay_weight) * fine_val_raw / fine_val_energy
                    + replay_weight * replay_val_raw / replay_val_energy
                )
            else:
                epoch_val_loss = utils.validate_epoch(
                    model, val_data, criterion, config
                )

            train_loss_history.append(epoch_train_loss)
            val_loss_history.append(
                epoch_val_loss if epoch_val_loss is not None else float('nan')
            )

            scheduler.step()
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]

            msg = (f"Epoch [{epoch+1}/{num_epochs}] "
                f"| Train: {epoch_train_loss:.3e}")
            if epoch_val_loss is not None:
                msg += f" | Val: {epoch_val_loss:.3e}"
                if finetune:
                    msg += (f" [cable raw/norm={fine_val_raw:.3e}/"
                            f"{fine_val_raw/fine_val_energy:.3e}; "
                            f"replay raw/norm={replay_val_raw:.3e}/"
                            f"{replay_val_raw/replay_val_energy:.3e}]")
            else:
                msg += " | Val: N/A"
            msg += f" | LR: {lr:.3e} | {elapsed:.2f}s"
            utils.write_log(log_filepath, msg)

            utils.save_checkpoint_and_best(
                epoch, model, optimizer, scheduler,
                epoch_train_loss, epoch_val_loss,
                best_losses, config, log_filepath,
                train_loss_history, val_loss_history,
            )

        utils.write_completion_manifest(
            config, num_epochs, best_losses, log_filepath
        )
        utils.write_log(log_filepath, "--- Training Completed ---")

    except Exception as e:
        import traceback
        utils.write_log(log_filepath,
                        f"Training failed: {e}\n{traceback.format_exc()}")
        utils.write_log(log_filepath, "--- Training Failed ---")
        raise
