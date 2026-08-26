"""Vectorized ORd11 adapter for one axial compartment per cable cell."""

from __future__ import annotations

import math
import logging
import shutil
import warnings
from dataclasses import dataclass

import torch

from . import Initial_ORd11, Ord11_model


_EAGER_ORD11_MODEL = Ord11_model
_ORD11_COMPILED = False


def configure_ord11_compilation(
    enabled: bool, device: torch.device | None = None
) -> tuple[bool, str]:
    """Select eager or torch.compile ORd11 before constructing the models.

    Compilation has a substantial one-off CPU cost but fuses the hundreds of
    small tensor operations in ORd11. It is therefore beneficial for long
    simulations and deliberately optional for short diagnostics.
    """
    global Ord11_model, _ORD11_COMPILED
    enabled = bool(enabled)
    mode = "reduce-overhead" if device is not None and device.type == "cuda" else "default"
    if enabled and not _ORD11_COMPILED:
        # reduce-overhead enables CUDA-graph machinery and is useful on CUDA,
        # but on CPU it only emits graph-partition warnings for ORd11's slice
        # assignments. The default CPU Inductor mode still fuses the kernel.
        if device is None or device.type == "cpu":
            # OSC's PyTorch config otherwise selects the deprecated Intel
            # Classic compiler and prints remark #10441. The system GNU C++
            # compiler is supported by CPU Inductor and avoids that warning.
            gxx = shutil.which("g++")
            if gxx is not None:
                from torch._inductor import config as inductor_config
                inductor_config.cpp.cxx = gxx
            warnings.filterwarnings("ignore", message="Can't initialize NVML")
            logging.getLogger("torch.utils.cpp_extension").setLevel(logging.ERROR)
        Ord11_model = torch.compile(_EAGER_ORD11_MODEL, mode=mode)
    elif not enabled:
        Ord11_model = _EAGER_ORD11_MODEL
    _ORD11_COMPILED = enabled
    return _ORD11_COMPILED, mode if enabled else "eager"


def _ord_time(time_ms: float, device: torch.device, dtype: torch.dtype):
    if _ORD11_COMPILED:
        # A tensor value is a dynamic graph input. A Python float would cause
        # Dynamo to specialize/recompile for every electrical time point.
        return torch.as_tensor(time_ms, device=device, dtype=dtype)
    return float(time_ms)

# The outer reduced solver supplies the membrane-patch current fractions.
# For MATLAB equivalence, axial patches use ``1-loc`` while terminal-disc
# patches use ``loc / cell_port_count``.  The stimulus remains a separate
# axial-only source and is never multiplied by ``f_I``.


@dataclass(frozen=True)
class CellGeometry:
    """Cell geometry and capacitance parameters in the ORd11 units."""

    length_um: float = 100.0
    radius_um: float = 11.0
    cm_density_uf_per_um2: float = 1.0e-8

    @property
    def axial_area_um2(self) -> float:
        return 2.0 * math.pi * self.radius_um * self.length_um

    @property
    def disc_area_um2(self) -> float:
        return math.pi * self.radius_um**2

    @property
    def total_area_um2(self) -> float:
        return self.axial_area_um2 + 2.0 * self.disc_area_um2

    @property
    def total_capacitance_uf(self) -> float:
        return self.total_area_um2 * self.cm_density_uf_per_um2

    @property
    def axial_capacitance_uf(self) -> float:
        return self.axial_area_um2 * self.cm_density_uf_per_um2


class ORd11CableIonicModel:
    """Own and advance the 40 non-voltage ORd11 states for all cells.

    Voltage is advanced by the outer cable solver. ``Ord11_model`` advances
    concentrations with Forward Euler and gates with its existing Rush-Larsen
    formulas, exactly as supplied in this repository.
    """

    NUM_STATES = 40

    def __init__(
        self,
        n_cells: int,
        dt_ms: float,
        device: torch.device,
        dtype: torch.dtype,
        geometry: CellGeometry | None = None,
        stimulus_cell: int = 0,
        stimulus_amplitude: float = 50.0,
        stimulus_duration_ms: float = 2.0,
        stimulus_bcl_ms: float = 1000.0,
        f_i: torch.Tensor | None = None,
    ) -> None:
        if n_cells < 2:
            raise ValueError("A cable requires at least two cells.")
        if not 0 <= stimulus_cell < n_cells:
            raise ValueError(
                f"stimulus_cell={stimulus_cell} is outside [0, {n_cells - 1}].")

        self.n_cells = n_cells
        self.dt_ms = float(dt_ms)
        self.device = device
        self.dtype = dtype
        self.geometry = geometry or CellGeometry()
        self.stimulus_cell = int(stimulus_cell)
        self.stimulus_amplitude = float(stimulus_amplitude)
        self.stimulus_duration_ms = float(stimulus_duration_ms)
        self.stimulus_bcl_ms = float(stimulus_bcl_ms)
        initial = Initial_ORd11(device=device, dtype=dtype)
        if initial.numel() != self.NUM_STATES + 1:
            raise ValueError(
                f"Initial_ORd11 returned {initial.numel()} values; expected 41.")
        self.rest_voltage_mv = float(initial[0].item())
        # State-major packing: [state0(all cells), state1(all cells), ...].
        self.gating_state = (
            initial[1:, None].repeat(1, n_cells).reshape(-1).clone()
        )

        stimulus_mask = torch.zeros(n_cells, device=device, dtype=dtype)
        stimulus_mask[stimulus_cell] = 1.0
        if f_i is None:
            f_i_tensor = torch.ones(14, device=device, dtype=dtype)
        else:
            f_i_tensor = torch.as_tensor(f_i, device=device, dtype=dtype).clone()
            if f_i_tensor.shape != (14,):
                raise ValueError(
                    f"f_i shape {tuple(f_i_tensor.shape)}; expected (14,)."
                )
            if not torch.isfinite(f_i_tensor).all():
                raise ValueError("f_i contains NaN/Inf.")

        self.parameters = {
            "dt": (
                torch.as_tensor(self.dt_ms, device=device, dtype=dtype)
                if _ORD11_COMPILED else self.dt_ms
            ),
            "N": n_cells,
            "L": self.geometry.length_um,
            "r": self.geometry.radius_um,
            "Ctot": self.geometry.total_capacitance_uf,
            "f_I": f_i_tensor,
            "iina": 0,
            "iinal": 1,
            "iito": 2,
            "iical": 3,
            "iikr": 4,
            "iiks": 5,
            "iik1": 6,
            "iinaca_i": 7,
            "iinaca_ss": 8,
            "iinak": 9,
            "iikb": 10,
            "iinab": 11,
            "iicab": 12,
            "iipca": 13,
            "fSERCA": 1.0,
            "fRyR": 1.0,
            "ftauhL": 1.0,
            "fCaMKa": 1.0,
            "fIleak": 1.0,
            "fJrel": 1.0,
            "celltype": 0,
            "bcl": float(stimulus_bcl_ms),
            "stim_dur": float(stimulus_duration_ms),
            "stim_amp": float(stimulus_amplitude),
            "indstim": stimulus_mask,
        }
        self.bulk_concentrations = torch.cat(
            [
                torch.full((n_cells,), 140.0, device=device, dtype=dtype),
                torch.full((n_cells,), 5.4, device=device, dtype=dtype),
                torch.full((n_cells,), 1.8, device=device, dtype=dtype),
            ]
        )

    @property
    def capacitance_uf(self) -> float:
        return self.geometry.axial_capacitance_uf

    def set_dt_ms(self, dt_ms: float) -> None:
        """Set the accepted electrical/ionic step used by the next ORd update.

        MATLAB uses 0.01 ms during the first 50 ms of each pacing cycle and
        0.1 ms afterwards.  The Python adapter therefore cannot keep ``dt`` as
        an immutable construction-time value.
        """
        dt_ms = float(dt_ms)
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive.")
        self.dt_ms = dt_ms
        if _ORD11_COMPILED:
            self.parameters["dt"].fill_(dt_ms)
        else:
            self.parameters["dt"] = dt_ms

    def stimulus_current_ua(self, time_ms: float) -> torch.Tensor:
        """Return applied stimulus current in the ORd11 Iion convention."""
        active = (float(time_ms) % self.stimulus_bcl_ms) < self.stimulus_duration_ms
        current = torch.zeros(
            self.n_cells, device=self.device, dtype=self.dtype
        )
        if active:
            # Ord11_model uses Rcg=2 and subtracts Istim inside Iion.
            current[self.stimulus_cell] = (
                2.0
                * self.geometry.total_capacitance_uf
                * self.stimulus_amplitude
            )
        return current

    @torch.no_grad()
    def current_only(
        self, time_ms: float, voltage_mv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Iion(t, phi) without committing the newly computed gates."""
        if voltage_mv.shape != (self.n_cells,):
            raise ValueError(
                f"voltage shape {tuple(voltage_mv.shape)}; "
                f"expected ({self.n_cells},).")
        full_state = torch.cat((voltage_mv, self.gating_state))
        ionic_current, components, gates_new, _ = Ord11_model(
            _ord_time(time_ms, self.device, self.dtype),
            full_state,
            self.parameters,
            self.bulk_concentrations,
            self.device,
            self.dtype,
        )
        if not torch.isfinite(ionic_current).all() or not torch.isfinite(gates_new).all():
            raise FloatingPointError(
                f"ORd11 produced NaN/Inf at t={time_ms:.6f} ms.")
        return ionic_current, components

    @torch.no_grad()
    def current_and_advance_gates(
        self, time_ms: float, voltage_mv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Iion(t_n) and advance ionic states from n to n+1."""
        if voltage_mv.shape != (self.n_cells,):
            raise ValueError(
                f"voltage shape {tuple(voltage_mv.shape)}; "
                f"expected ({self.n_cells},).")
        full_state = torch.cat((voltage_mv, self.gating_state))
        ionic_current, components, gates_new, _ = Ord11_model(
            _ord_time(time_ms, self.device, self.dtype),
            full_state,
            self.parameters,
            self.bulk_concentrations,
            self.device,
            self.dtype,
        )
        if not torch.isfinite(ionic_current).all() or not torch.isfinite(gates_new).all():
            raise FloatingPointError(
                f"ORd11 produced NaN/Inf at t={time_ms:.6f} ms.")
        self.gating_state = gates_new
        return ionic_current, components


@torch.no_grad()
def current_and_advance_models(
    time_ms: float,
    models: tuple[ORd11CableIonicModel, ...],
    voltages_mv: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Advance several compatible ORd11 patch groups in one vectorized call.

    Axial cells and the two active terminal patches share the same ORd11
    equations/geometry but carry different ``f_I`` vectors and stimulus masks.
    Calling the eager ORd11 kernel once for their concatenated compartments
    avoids paying its large Python/operator-dispatch overhead twice per
    electrical step. State ownership remains separate after the call.
    """
    if not models or len(models) != len(voltages_mv):
        raise ValueError("models and voltages_mv must be non-empty and aligned.")

    first = models[0]
    total_n = sum(model.n_cells for model in models)
    gate_blocks = []
    concentration_blocks = []
    f_i_blocks = []
    stimulus_blocks = []
    sizes = []
    for model, voltage in zip(models, voltages_mv):
        if model.device != first.device or model.dtype != first.dtype:
            raise ValueError("All combined ORd11 models must share device/dtype.")
        if model.geometry != first.geometry:
            raise ValueError("All combined ORd11 models must share geometry.")
        if not math.isclose(model.dt_ms, first.dt_ms, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("All combined ORd11 models must use the same dt.")
        if voltage.shape != (model.n_cells,):
            raise ValueError(
                f"voltage shape {tuple(voltage.shape)}; expected ({model.n_cells},)."
            )
        n = model.n_cells
        sizes.append(n)
        gate_blocks.append(model.gating_state.reshape(model.NUM_STATES, n))
        concentration_blocks.append(model.bulk_concentrations.reshape(3, n))
        f_i = model.parameters["f_I"]
        if f_i.shape == (14,):
            f_i = f_i[:, None].expand(14, n)
        elif f_i.shape != (14, n):
            raise ValueError(
                f"f_I shape {tuple(f_i.shape)}; expected (14,) or (14, {n})."
            )
        f_i_blocks.append(f_i)
        stimulus_blocks.append(
            float(model.parameters["stim_amp"]) * model.parameters["indstim"]
        )

    combined_voltage = torch.cat(voltages_mv)
    combined_gates = torch.cat(gate_blocks, dim=1).reshape(-1)
    combined_state = torch.cat((combined_voltage, combined_gates))
    combined_bulk = torch.cat(concentration_blocks, dim=1).reshape(-1)
    parameters = dict(first.parameters)
    parameters["N"] = total_n
    parameters["f_I"] = torch.cat(f_i_blocks, dim=1)
    # Ord11 multiplies stim_amp*indstim. Store each group's effective amplitude
    # in the combined mask so boundary patches retain their zero stimulus even
    # though the axial and boundary groups use one shared parameter dictionary.
    parameters["stim_amp"] = 1.0
    parameters["indstim"] = torch.cat(stimulus_blocks)

    ionic_current, components, gates_new, _ = Ord11_model(
        _ord_time(time_ms, first.device, first.dtype),
        combined_state,
        parameters,
        combined_bulk,
        first.device,
        first.dtype,
    )
    if not torch.isfinite(ionic_current).all() or not torch.isfinite(gates_new).all():
        raise FloatingPointError(
            f"Combined ORd11 produced NaN/Inf at t={time_ms:.6f} ms."
        )

    gate_matrix = gates_new.reshape(first.NUM_STATES, total_n)
    current_parts = []
    component_parts = []
    start = 0
    for model, n in zip(models, sizes):
        stop = start + n
        model.gating_state = gate_matrix[:, start:stop].reshape(-1).clone()
        current_parts.append(ionic_current[start:stop])
        component_parts.append(components[:, start:stop])
        start = stop
    return tuple(current_parts), tuple(component_parts)

