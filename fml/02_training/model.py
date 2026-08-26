"""Feed-forward flow-map network for the aligned Ctrl-now contract.

QoI history has ``M+1`` samples and Ctrl has ``M+2`` samples. The last Ctrl
sample is the known target-time input for the next QoI prediction.

``MODEL_TYPE=FNN`` predicts the next QoI directly. ``MODEL_TYPE=ResNet`` uses
the same MLP but adds its output to the latest QoI. Additional families can be
introduced through the model factory without changing this data contract.
"""
import torch
import torch.nn as nn
from config import Config


class FNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_qoi = config.NUM_QOI
        self.num_memory = config.NUM_MEMORY
        self.num_params = getattr(config, "NUM_PARAMS", 0)
        self.num_ctrl = getattr(config, "NUM_CTRL", 0)
        self.model_type = config.MODEL_TYPE
        if self.model_type not in {"FNN", "ResNet"}:
            raise ValueError(f"FNN implementation cannot build {self.model_type!r}")

        qoi_history_len = self.num_memory + 1
        ctrl_history_len = self.num_memory + 2
        input_dim = (
            self.num_qoi * qoi_history_len
            + self.num_ctrl * ctrl_history_len
            + self.num_params
        )

        layers = []
        in_features = input_dim
        for width in config.HIDDEN_LAYER_SIZES:
            layers.append(nn.Linear(in_features, width))
            if config.USE_BATCH_NORM:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_features = width

        # Keep stable layer names across training, evaluation, and deployment.
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_delta_out = nn.Linear(in_features, self.num_qoi)

    def forward(self, qoi_history: torch.Tensor,
                ctrl_history: torch.Tensor = None,
                params_input: torch.Tensor = None) -> torch.Tensor:
        batch_size = qoi_history.shape[0]
        qoi_history_len = self.num_memory + 1
        ctrl_history_len = self.num_memory + 2
        if qoi_history.shape[1:] != (self.num_qoi, qoi_history_len):
            raise ValueError(
                f"qoi_history shape {qoi_history.shape} does not match "
                f"[*, {self.num_qoi}, {qoi_history_len}]"
            )

        parts = [qoi_history.reshape(batch_size, -1)]
        if self.num_ctrl:
            if ctrl_history is None:
                raise ValueError("ctrl_history is required when NUM_CTRL > 0")
            if ctrl_history.shape[1:] != (self.num_ctrl, ctrl_history_len):
                raise ValueError(
                    f"ctrl_history shape {ctrl_history.shape} does not match "
                    f"[*, {self.num_ctrl}, {ctrl_history_len}]"
                )
            parts.append(ctrl_history.reshape(batch_size, -1))

        if self.num_params:
            if params_input is None:
                raise ValueError("params_input is required when NUM_PARAMS > 0")
            parts.append(params_input)

        nn_out = self.fc_delta_out(self.hidden_layers(torch.cat(parts, dim=1)))
        if self.model_type == "ResNet":
            nn_out = qoi_history[:, :, -1] + nn_out
        return nn_out.unsqueeze(-1)
