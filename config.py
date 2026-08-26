"""Single source of truth for data generation, training, and evaluation.

The pipeline has one control contract: QoI and Ctrl start at the same time,
and every saved Ctrl burst contains one more sample than its QoI burst.  The
extra Ctrl sample is the known target-time input used by the flow map.
"""
import argparse
import os
import sys

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


_ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--memlen", type=int, required=True)
parser.add_argument("--dataset", default="sine_one_cell")
parser.add_argument("--bcl", type=int, default=None)
parser.add_argument("--nrec", type=int, default=3)
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("--mesh-idx", "--mesh_idx", dest="mesh_idx", type=int,
                    default=1)
parser.add_argument("--gj-coupling", "--gj_coupling", dest="gj_coupling",
                    choices=("strong", "weak"), default="strong")
parser.add_argument("--hidden_layers", type=str, default=None)
parser.add_argument("--model_type", choices=("FNN", "ResNet", "LSTM"),
                    default="FNN")
parser.add_argument("--finetune", type=lambda x: x.lower() == "true",
                    default=False)
parser.add_argument("--finetune_base", choices=("best_train", "best_val"),
                    default="best_val")
args, _ = parser.parse_known_args()

_REMOVED_OPTIONS = {
    "--ctrl_mode", "--ctrl-mode",
    "--include_ina_gates", "--include-ina-gates",
    "--include_all_gates", "--include-all-gates",
}
for token in sys.argv[1:]:
    if token.split("=", 1)[0] in _REMOVED_OPTIONS:
        raise ValueError(
            f"{token.split('=', 1)[0]} has been removed; the pipeline always "
            "uses aligned Ctrl-now data with voltage-only Ctrl."
        )

if args.memlen < 0:
    raise ValueError("--memlen must be nonnegative")
if args.nrec <= 0:
    raise ValueError("--nrec must be positive")
if args.dt <= 0:
    raise ValueError("--dt must be positive")
if args.mesh_idx <= 0:
    raise ValueError("--mesh-idx must be positive")


def _parse_hidden_layers(value):
    if value is None:
        return [80, 80, 80, 80]
    try:
        widths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            "--hidden_layers must be comma-separated positive integers"
        ) from exc
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("--hidden_layers must contain positive integers")
    return widths


class Config:
    # Time and dimensions
    DT_STEP = 0.1
    DT_FML = args.dt
    DT_CONTROL = 0.5
    dt = DT_FML
    NUM_QOI = 2
    NUM_CTRL = 2
    NUM_PARAMS = 0
    NUM_MEMORY = args.memlen
    NUM_RECURRENT = args.nrec
    BURST_LEN = NUM_MEMORY + 1 + NUM_RECURRENT
    # Validation always rolls ten steps farther than training recurrence.
    NUM_VAL_RECURRENT = NUM_RECURRENT + 10
    VAL_BURST_LEN = NUM_MEMORY + 1 + NUM_VAL_RECURRENT

    # Model
    MODEL_TYPE = args.model_type
    HIDDEN_LAYER_SIZES = _parse_hidden_layers(args.hidden_layers)
    USE_BATCH_NORM = False
    if _HAS_TORCH:
        DATA_TYPE = torch.float64
        torch.set_default_dtype(DATA_TYPE)
    else:
        DATA_TYPE = None

    # Training
    NUM_SAMPLES = 100000
    # The hybrid protocol is much longer (110 ms) than the legacy clamp
    # trajectories, so use more independently sampled windows from every
    # non-test trajectory. Environment overrides remain available for small
    # smoke tests and future learning-curve studies.
    _DEFAULT_NUM_BURSTS = "50" if args.dataset == "hybrid_one_cell" else "5"
    NUM_BURSTS = int(os.environ.get("NUM_BURSTS", _DEFAULT_NUM_BURSTS))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4096"))
    LEARNING_RATE = 5e-4
    LR_GAMMA = 0.9999
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50000"))
    NUM_WORKERS = int(os.environ.get("WORKERS", "4"))
    CLIP_GRAD_NORM = None
    LOSS_WEIGHTS = None
    # Weight initialization and DataLoader shuffling were unseeded, so two runs
    # of the same shape produced different networks.  With the Ctrl history
    # columns nearly collinear the design matrix has an effective rank of only
    # ~4 out of 126 inputs, so those runs differ over a large flat manifold:
    # same training loss, different behaviour off the training distribution --
    # which is where the cable solver operates.  Set TORCH_SEED to make a run
    # reproducible; leave it unset to keep the previous (random) behaviour.
    _TORCH_SEED_ENV = os.environ.get("TORCH_SEED", "").strip()
    TORCH_SEED = int(_TORCH_SEED_ENV) if _TORCH_SEED_ENV else None
    CHECKPOINT_EPOCH_INTERVAL = int(
        os.environ.get("CHECKPOINT_EPOCH_INTERVAL", "500")
    )
    RESUME_TRAINING = os.environ.get(
        "RESUME_TRAINING", "false"
    ).strip().lower() == "true"

    # Fine tuning
    FINETUNE = args.finetune
    FINETUNE_BASE = args.finetune_base
    FINETUNE_LEARNING_RATE = 5e-5
    FINETUNE_LR_GAMMA = 0.9999
    FINETUNE_NUM_EPOCHS = int(
        os.environ.get("FINETUNE_NUM_EPOCHS", "3000")
    )
    # Preserve synthetic solver coverage while adapting to cable/AP data.
    FINETUNE_REPLAY_FRACTION = float(
        os.environ.get("FINETUNE_REPLAY_FRACTION", "0.5")
    )
    FINETUNE_REPLAY_VAL_SAMPLES = int(
        os.environ.get("FINETUNE_REPLAY_VAL_SAMPLES", "4096")
    )

    # Data generation
    T_SIM = 10000
    IGNORE_START = 0
    _DEFAULT_NUM_VAL = "30" if args.dataset == "hybrid_one_cell" else "1"
    NUM_VAL = int(os.environ.get("NUM_VAL_BURSTS", _DEFAULT_NUM_VAL))
    NUM_TEST = 1
    NUM_TEST_TRAJECTORIES = int(
        os.environ.get("NUM_TEST_TRAJECTORIES", "20")
    )
    TEST_FULL_TRAJECTORY = (
        args.dataset == "hybrid_one_cell"
        and os.environ.get(
            "TEST_FULL_TRAJECTORY", "true"
        ).strip().lower() == "true"
    )
    T_TEST = 10.0
    SCALE_DATA = False
    QOI_SCALE_FACTOR = 1000.0
    DATA_SEED = 42
    FILE_GLOB = "*.mat"
    T_VAR = "t"
    QOI_VAR = "Icleft"
    CTRL_VAR = "phi_axial"

    # Dataset
    DATASET = args.dataset
    GJ_COUPLING = args.gj_coupling
    BCL = 1000 if args.bcl is None else args.bcl
    MESH_IDX = args.mesh_idx
    _DS_FOLDER = {
        "sine_rest": "voltage_clamp_sine_rest_ds",
        "sine": "voltage_clamp_sine_ds",
        "step_vc": "voltage_clamp_ds",
        "sine_one_cell": "voltage_clamp_sine_one_cell_free_ds",
        "hybrid_one_cell": "voltage_clamp_hybrid_one_cell_ds",
    }.get(DATASET, f"voltage_clamp_{DATASET}_ds")
    STRONG_PATH = (f"/fs/ess/PAS1622/RichardSui/FML_ID_dataset/"
                   f"{_DS_FOLDER}/FEM{MESH_IDX}/strong")
    WEAK_PATH = (f"/fs/ess/PAS1622/RichardSui/FML_ID_dataset/"
                 f"{_DS_FOLDER}/FEM{MESH_IDX}/weak")

    # Paths and unambiguous names.  nm/nr/dt are present in data, model, and
    # log stems; obsolete Ctrl-mode and gating tags are intentionally absent.
    DATA_DIR = os.path.join(_ROOT, "fml", "data")
    MODEL_DIR = os.path.join(_ROOT, "fml", "model")
    LOG_ROOT = os.path.join(_ROOT, "fml", "logs")
    _BASE_LOG_DIR = os.path.join(
        LOG_ROOT, f"FEM{MESH_IDX}", f"GJ{GJ_COUPLING}"
    )
    # Keep fine-tuning histories separate from from-scratch training histories.
    # This also prevents batch discovery of base models from accidentally
    # treating a fine-tune run as another base-training run.
    LOG_DIR = (
        os.path.join(_BASE_LOG_DIR, "fine_tune")
        if FINETUNE else _BASE_LOG_DIR
    )
    EVAL_DIR = os.path.join(_ROOT, "fml", "eval")
    _DT_TAG = format(dt, "g").replace(".", "p")
    _RUN_TAG = (
        f"FEM{MESH_IDX}_GJ{GJ_COUPLING}_{DATASET}_"
        f"nm{NUM_MEMORY}_nr{NUM_RECURRENT}_dt{_DT_TAG}"
    )
    TRAIN_FILE = os.path.join(DATA_DIR, f"train_data_{_RUN_TAG}.npz")
    VAL_FILE = os.path.join(DATA_DIR, f"val_data_{_RUN_TAG}.npz")
    TEST_FILE = os.path.join(DATA_DIR, f"test_data_{_RUN_TAG}.npz")
    VAL_NPZ = VAL_FILE
    TEST_NPZ = TEST_FILE
    VAL_DATA_AVAILABLE = True

    # MODEL_TAG appends a suffix so replicate runs of one shape (for example a
    # seed study) keep separate checkpoints instead of overwriting each other.
    # Empty by default, so existing names are unchanged.
    MODEL_TAG = os.environ.get("MODEL_TAG", "").strip()
    MODEL_NAME_BASE = (
        f"FEM{MESH_IDX}_GJ{GJ_COUPLING}_VC_{DATASET}_"
        f"nm{NUM_MEMORY}_nr{NUM_RECURRENT}_dt{_DT_TAG}_{MODEL_TYPE}_"
        f"hid{'-'.join(map(str, HIDDEN_LAYER_SIZES))}"
        + (f"_{MODEL_TAG}" if MODEL_TAG else "")
    )
    _FINETUNE_PREFIX = (
        f"finetune_cable_FEM{MESH_IDX}_GJ{GJ_COUPLING.lower()}_BCL{BCL}_"
        f"nm{NUM_MEMORY}_nr{NUM_RECURRENT}_dt{_DT_TAG}"
    )
    FINETUNE_TRAIN_NPZ = os.path.join(
        DATA_DIR, f"{_FINETUNE_PREFIX}_train.npz"
    )
    FINETUNE_VAL_NPZ = os.path.join(DATA_DIR, f"{_FINETUNE_PREFIX}_val.npz")
    FINETUNE_TEST_NPZ = os.path.join(DATA_DIR, f"{_FINETUNE_PREFIX}_test.npz")
    FINETUNE_BASE_CHECKPOINT = os.environ.get(
        "FINETUNE_BASE_CHECKPOINT_OVERRIDE",
        os.path.join(MODEL_DIR, f"{MODEL_NAME_BASE}_{FINETUNE_BASE}.pth"),
    )
    FINETUNE_MODEL_NAME = f"{MODEL_NAME_BASE}_finetune_BCL{BCL}"

    def __init__(self):
        self.NUM_TRAIN_DATA = self.NUM_SAMPLES * self.NUM_BURSTS
        self.SOURCE_DIR = (
            self.STRONG_PATH
            if self.GJ_COUPLING.lower() == "strong"
            else self.WEAK_PATH
        )
        self.DEVICE = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if _HAS_TORCH else None
        )
        for directory in (self.DATA_DIR, self.MODEL_DIR,
                          self.LOG_DIR, self.EVAL_DIR):
            os.makedirs(directory, exist_ok=True)
