"""Validate generated NPZ files (CSV preprocessing is no longer required)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config


def validate(path, config, expected_nrec=None):
    data = np.load(path)
    required = {"qoi", "ctrl", "dt_ms"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    qoi, ctrl = data["qoi"], data["ctrl"]
    if (qoi.ndim != 3 or ctrl.ndim != 3
            or qoi.shape[0] != ctrl.shape[0]
            or ctrl.shape[2] != qoi.shape[2] + 1):
        raise ValueError(f"{path}: incompatible qoi {qoi.shape} / ctrl {ctrl.shape}")
    if ctrl.shape[1] != config.NUM_CTRL:
        raise ValueError(f"{path}: Ctrl channels {ctrl.shape[1]} != {config.NUM_CTRL}")
    if not np.isclose(float(data["dt_ms"]), config.dt, atol=1e-12, rtol=0.0):
        raise ValueError(f"{path}: dt mismatch")
    n_rec = qoi.shape[2] - config.NUM_MEMORY - 1
    if expected_nrec is not None and n_rec != expected_nrec:
        raise ValueError(
            f"{path}: recurrent steps {n_rec} != expected {expected_nrec}"
        )
    if "num_recurrent" in data.files:
        metadata_nrec = int(np.asarray(data["num_recurrent"]).item())
        if metadata_nrec != n_rec:
            raise ValueError(
                f"{path}: metadata num_recurrent={metadata_nrec} "
                f"!= shape-derived {n_rec}"
            )
    print(f"OK: {path}  qoi={qoi.shape} ctrl={ctrl.shape}")


if __name__ == "__main__":
    cfg = Config()
    validate(cfg.TRAIN_FILE, cfg, cfg.NUM_RECURRENT)
    validate(cfg.VAL_FILE, cfg, cfg.NUM_VAL_RECURRENT)
    validate(cfg.TEST_FILE, cfg)
