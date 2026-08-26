"""Batched FML cleft-current predictor and rolling junction histories."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch import nn


class InferenceFNN(nn.Module):
    """Checkpoint-compatible MLP using the training Ctrl contract.

    Each Ctrl channel always contains its M+1 history samples followed by the
    target-time Ctrl sample, so Ctrl has shape [B, C, M+2] before flattening.
    """

    def __init__(self, config):
        super().__init__()
        self.model_type = config.MODEL_TYPE
        if self.model_type not in {"FNN", "ResNet"}:
            raise NotImplementedError(
                f"Numerical inference supports FNN/ResNet, got {self.model_type!r}."
            )
        self.num_qoi = config.NUM_QOI
        self.num_ctrl = config.NUM_CTRL
        self.history_length = config.NUM_MEMORY + 1
        input_dim = (
            self.num_qoi * self.history_length
            + self.num_ctrl * self.history_length
            + self.num_ctrl
        )
        layers = []
        in_features = input_dim
        for width in config.HIDDEN_LAYER_SIZES:
            layers.append(nn.Linear(in_features, width))
            if config.USE_BATCH_NORM:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_features = width
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_delta_out = nn.Linear(in_features, self.num_qoi)

    def forward(
        self,
        qoi_history: torch.Tensor,
        ctrl_history: torch.Tensor,
        ctrl_current: torch.Tensor,
    ) -> torch.Tensor:
        batch = qoi_history.shape[0]
        ctrl_full = torch.cat(
            (ctrl_history, ctrl_current.unsqueeze(-1)), dim=2
        )
        parts = [
            qoi_history.reshape(batch, -1),
            ctrl_full.reshape(batch, -1),
        ]
        value = torch.cat(parts, dim=1)
        output = self.fc_delta_out(self.hidden_layers(value))
        if self.model_type == "ResNet":
            output = qoi_history[:, :, -1] + output
        return output.unsqueeze(-1)


class FMLJunctionBatch:
    """Predict all Ncell-1 junction currents in one FNN batch.

    Ctrl is exactly [phi_left, phi_right]. Gating state is not part of the
    learned model or its rolling history.
    """

    def __init__(
        self,
        config,
        n_cells: int,
        rest_voltage_mv: float,
        checkpoint: str | os.PathLike | None = None,
        subtract_rest_bias: bool = False,
    ) -> None:
        self.n_cells = int(n_cells)
        self.n_junctions = self.n_cells - 1
        self.device = config.DEVICE
        self.dtype = config.DATA_TYPE
        self.history_length = config.NUM_MEMORY + 1
        self.subtract_rest_bias = bool(subtract_rest_bias)
        expected_ctrl = 2
        if config.NUM_QOI != 2 or config.NUM_CTRL != expected_ctrl:
            raise ValueError(
                "The requested checkpoint requires NUM_QOI=2 and "
                f"NUM_CTRL={expected_ctrl}; got {config.NUM_QOI} and "
                f"{config.NUM_CTRL}.")
        if checkpoint is None:
            checkpoint = Path(config.MODEL_DIR) / f"{config.MODEL_NAME_BASE}_best_val.pth"
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FML checkpoint not found: {checkpoint}")

        self.model = InferenceFNN(config).to(self.device)
        try:
            state_dict = torch.load(
                checkpoint, map_location=self.device, weights_only=True
            )
        except TypeError:  # Compatibility with older PyTorch releases.
            state_dict = torch.load(checkpoint, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            expected = self.model.hidden_layers[0].in_features
            first_weight = state_dict.get("hidden_layers.0.weight")
            found = first_weight.shape[1] if first_weight is not None else "unknown"
            raise RuntimeError(
                "Checkpoint/input mismatch: the fixed per-channel Ctrl M+2 "
                f"contract expects first-layer width {expected}, checkpoint has {found}."
            ) from exc
        self.model.eval()

        # Uniform rest has zero axial-to-ID current. Seed the M+1 accepted
        # samples immediately preceding the first candidate time.
        self.qoi_history = torch.zeros(
            self.n_junctions,
            2,
            self.history_length,
            device=self.device,
            dtype=self.dtype,
        )
        rest_ctrl_now = self._build_ctrl_now(
            torch.full(
                (self.n_cells,), float(rest_voltage_mv),
                device=self.device, dtype=self.dtype,
            )
        )  # (n_junctions, NUM_CTRL), identical across junctions at rest
        self.ctrl_history = rest_ctrl_now.unsqueeze(-1).expand(
            self.n_junctions, expected_ctrl, self.history_length
        ).clone()

        # Measure the uniform-rest regression bias for diagnostics. It is not
        # removed by default: strict raw-prediction mode must pass both network
        # outputs to their owning cells without changing their values. The
        # correction remains available only as an explicit diagnostic option.
        with torch.no_grad():
            self.rest_current_bias = self.model(
                self.qoi_history[:1],
                self.ctrl_history[:1],
                rest_ctrl_now[:1],
            ).squeeze(0).squeeze(-1)

        self.rest_drift_mv_per_ms = self._rest_drift_mv_per_ms()
        self._check_rest_drift(checkpoint)

    def _rest_drift_mv_per_ms(self) -> float:
        """Spurious dV/dt this checkpoint injects into a cell at uniform rest.

        Physics gives zero: with no voltage difference there is no axial
        current.  An interior cell owns one side of each neighbouring
        junction, so at uniform rest it receives both predicted currents.

        The screening geometry is the production cable's (NumericalConfig
        length_um/radius_um/cm_density_uf_per_um2).  It is not read from that
        config because this is an order-of-magnitude gate, not part of the
        solve, and the junction model must stay usable without it.
        """
        capacitance_uf = 2.0 * math.pi * 11.0 * 100.0 * 1.0e-8
        return -float(self.rest_current_bias.sum().item()) / capacitance_uf

    def _check_rest_drift(self, checkpoint) -> None:
        """Reject a checkpoint that cannot hold the cable at rest.

        Nothing in an unweighted MSE on Icleft constrains the prediction at
        uniform rest, so this is set by initialization noise rather than by
        the data, and it varies wildly across hyperparameters.  It is also
        what makes the macro-endpoint solve diverge: 10 mV/ms drags the whole
        cable through the INa threshold in ~3 ms, well before any stimulus
        has propagated.  A real AP upstroke is 200-300 mV/ms and plateau
        drift is ~0.01 mV/ms, so anything above a few tenths is unusable.

        Override with FML_MAX_REST_DRIFT_MV_PER_MS (0 disables the check).
        """
        limit = float(os.environ.get("FML_MAX_REST_DRIFT_MV_PER_MS", "0.5"))
        if limit <= 0.0:
            return
        drift = abs(self.rest_drift_mv_per_ms)
        if drift <= limit:
            return
        raise ValueError(
            f"FML checkpoint injects {self.rest_drift_mv_per_ms:+.3f} mV/ms of "
            f"spurious depolarization at uniform rest (limit {limit:g} mV/ms): "
            f"{checkpoint}\n"
            f"  predicted Icleft at rest = {self.rest_current_bias.tolist()} uA "
            f"(physics: 0)\n"
            "  A checkpoint this far off the resting manifold will not produce "
            "a usable cable solve. Check that its training data was generated "
            "with a stable cleft sub-step (see "
            "error_analysis/ROOT_CAUSE_cleft_substep.md), then screen with "
            "error_analysis/scan_rest_bias.py.\n"
            "  Set FML_MAX_REST_DRIFT_MV_PER_MS=0 to bypass."
        )

    def adjacent_voltage(self, voltage_mv: torch.Tensor) -> torch.Tensor:
        """Return [phi_left, phi_right] for every cable junction."""
        if voltage_mv.shape != (self.n_cells,):
            raise ValueError(
                f"voltage shape {tuple(voltage_mv.shape)}; "
                f"expected ({self.n_cells},).")
        return torch.stack((voltage_mv[:-1], voltage_mv[1:]), dim=1)

    def _build_ctrl_now(self, voltage_mv: torch.Tensor) -> torch.Tensor:
        return self.adjacent_voltage(voltage_mv)

    def _predict_current_impl(
        self,
        voltage_mv: torch.Tensor,
        ctrl_now: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ctrl_now is None:
            ctrl_now = self._build_ctrl_now(voltage_mv)
        raw_prediction = self.model(
            self.qoi_history,
            self.ctrl_history,
            ctrl_now,
        ).squeeze(-1)
        # Preserve the two independently predicted junction-side currents.
        # They are not constrained to have equal magnitude or opposite sign.
        prediction = raw_prediction
        if self.subtract_rest_bias:
            prediction = prediction - self.rest_current_bias.unsqueeze(0)
        return prediction, ctrl_now

    @torch.no_grad()
    def predict_current(
        self,
        voltage_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the candidate Icleft from history and candidate voltage."""
        prediction, ctrl_now = self._predict_current_impl(voltage_mv)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("FML predicted NaN/Inf cleft current.")
        return prediction, ctrl_now

    def _predict_current_and_jacobian_autograd(
        self,
        voltage_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autograd reference for current and voltage Jacobian."""
        with torch.enable_grad():
            ctrl_now = self._build_ctrl_now(voltage_mv.detach()).detach()
            ctrl_now.requires_grad_(True)
            prediction, _ = self._predict_current_impl(
                voltage_mv, ctrl_now=ctrl_now
            )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("FML predicted NaN/Inf cleft current.")
            jac_rows = []
            n_sides = prediction.shape[1]
            for side in range(n_sides):
                (grad,) = torch.autograd.grad(
                    prediction[:, side].sum(), ctrl_now,
                    retain_graph=(side < n_sides - 1),
                    allow_unused=True,
                )
                jac_rows.append(
                    torch.zeros_like(ctrl_now[:, :2])
                    if grad is None else grad[:, :2]
                )
            jac = torch.stack(jac_rows, dim=1)
        return prediction.detach(), ctrl_now.detach(), jac.detach()

    @torch.no_grad()
    def _predict_current_and_jacobian_analytic(
        self,
        voltage_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate this FNN and its 2-column voltage Jacobian in one pass.

        The deployed model is a stack of Linear/(optional BatchNorm)/ReLU
        layers.  Propagating only two tangent columns through that stack is
        exactly forward-mode differentiation, but avoids constructing an
        autograd graph and launching two backward passes at every Newton
        evaluation.
        """
        ctrl_now = self._build_ctrl_now(voltage_mv.detach()).detach()
        batch_size = self.qoi_history.shape[0]
        qoi_flat = self.qoi_history.reshape(batch_size, -1)
        ctrl_full = torch.cat(
            (self.ctrl_history, ctrl_now.unsqueeze(-1)), dim=2
        )
        combined_input = torch.cat(
            [qoi_flat, ctrl_full.reshape(batch_size, -1)], dim=1
        )

        tangent = torch.zeros(
            batch_size, combined_input.shape[1], 2,
            device=combined_input.device, dtype=combined_input.dtype,
        )
        ctrl_full_length = self.history_length + 1
        for channel in range(self.model.num_ctrl):
            current_column = (
                qoi_flat.shape[1]
                + channel * ctrl_full_length
                + ctrl_full_length - 1
            )
            tangent[:, current_column, channel] = 1.0

        value = combined_input
        for layer in self.model.hidden_layers:
            if isinstance(layer, nn.Linear):
                value = layer(value)
                tangent = torch.einsum("oi,bik->bok", layer.weight, tangent)
            elif isinstance(layer, nn.BatchNorm1d):
                if layer.training:
                    raise RuntimeError(
                        "Analytic FML Jacobian requires BatchNorm evaluation mode."
                    )
                scale = torch.rsqrt(layer.running_var + layer.eps)
                if layer.affine:
                    scale = scale * layer.weight
                value = layer(value)
                tangent = tangent * scale.view(1, -1, 1)
            elif isinstance(layer, nn.ReLU):
                active = value > 0
                value = layer(value)
                tangent = tangent * active.unsqueeze(-1)
            else:
                raise RuntimeError(
                    "Unsupported FML layer for analytic Jacobian: "
                    f"{type(layer).__name__}."
                )

        prediction = self.model.fc_delta_out(value)
        jac = torch.einsum(
            "oi,bik->bok", self.model.fc_delta_out.weight, tangent
        )
        if self.model.model_type == "ResNet":
            prediction = prediction + self.qoi_history[:, :, -1]
        if self.subtract_rest_bias:
            prediction = prediction - self.rest_current_bias.unsqueeze(0)
        if not torch.isfinite(prediction).all() or not torch.isfinite(jac).all():
            raise FloatingPointError("FML predicted NaN/Inf current or Jacobian.")
        return prediction, ctrl_now, jac

    def predict_current_and_jacobian(
        self,
        voltage_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict Icleft_n and d(Icleft)/d([phi_left, phi_right]) per junction.

        For Newton's method on the implicit phi<->Icleft coupling
        (macro endpoint solver). The FNN processes every junction as an
        independent batch row (no cross-junction mixing inside the
        network), so the Jacobian of each junction's 2 outputs w.r.t. its
        own 2 voltage inputs is block-diagonal across junctions. The two
        voltage tangent columns are propagated directly through the FNN in
        the same batched forward pass, with no autograd graph/backward pass.

        Returns (prediction, ctrl_now, jac) where
        ``jac[:, out_side, in_side]`` is d(prediction[:, out_side]) /
        d([phi_left, phi_right][in_side]) for the same junction -- only the
        voltage columns of ctrl_now (indices 0, 1).
        The production path uses exact forward-mode chain propagation through
        the small FNN.  ``_predict_current_and_jacobian_autograd`` is retained
        as a reference implementation for regression tests.
        """
        return self._predict_current_and_jacobian_analytic(voltage_mv)

    @torch.no_grad()
    def commit(self, current: torch.Tensor, ctrl_now: torch.Tensor) -> None:
        """Append the accepted current/voltage sample to both histories."""
        self.qoi_history = torch.cat(
            (self.qoi_history[:, :, 1:], current.unsqueeze(-1)), dim=2
        )
        self.ctrl_history = torch.cat(
            (self.ctrl_history[:, :, 1:], ctrl_now.unsqueeze(-1)), dim=2
        )
