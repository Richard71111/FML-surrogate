"""Shared data structures and equations for the cable solver."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class CableResult:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    boundary_voltage_mv: np.ndarray
    current_time_ms: np.ndarray
    time_step_ms: np.ndarray
    ionic_current_ua: np.ndarray
    cleft_current_ua: np.ndarray
    cell_cleft_current_ua: np.ndarray
    boundary_current_ua: np.ndarray
    boundary_ionic_current_ua: np.ndarray
    stimulus_current_ua: np.ndarray
    fml_commit_time_ms: np.ndarray
    sigma: np.ndarray
    dt_ms: float
    dt_fine_ms: float
    dt_coarse_ms: float
    fine_window_ms: float
    capacitance_uf: float
    boundary_capacitance_uf: float
    boundary_area_um2: float
    gmyo_ms: float
    bcl_ms: float
    nbeats: int
    iteration_wall_time_s: np.ndarray
    simulation_wall_time_s: float
    coupling_iterations: np.ndarray
    coupling_converged: np.ndarray
    coupling_residual_mv: np.ndarray
    record_fml_only: bool
    record_last_beat_only: bool


def build_cleft_incidence(
    n_cells: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map each independently predicted junction-side current to its owner."""
    sigma = torch.zeros(
        (n_cells, 2 * (n_cells - 1)), device=device, dtype=dtype
    )
    junction = torch.arange(n_cells - 1, device=device)
    sigma[junction, 2 * junction] = 1.0
    sigma[junction + 1, 2 * junction + 1] = 1.0
    return sigma


def terminal_boundary_load(
    voltage_mv: torch.Tensor,
    boundary_voltage_old_mv: torch.Tensor,
    boundary_ionic_current_ua: torch.Tensor,
    dt_ms: float,
    capacitance_uf: float,
    gmyo_ms: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implicitly eliminate MATLAB's two active terminal-disc nodes."""
    load = torch.zeros_like(voltage_mv)
    if boundary_ionic_current_ua.shape != (2,):
        raise ValueError("boundary_ionic_current_ua must have shape (2,).")
    if not enabled:
        return load, boundary_voltage_old_mv, torch.zeros(
            2, device=voltage_mv.device, dtype=voltage_mv.dtype
        )

    c_over_dt = float(capacitance_uf) / float(dt_ms)
    denominator = c_over_dt + float(gmyo_ms)
    endpoints = torch.stack((voltage_mv[0], voltage_mv[-1]))
    boundary_new = (
        c_over_dt * boundary_voltage_old_mv
        + float(gmyo_ms) * endpoints
        - boundary_ionic_current_ua
    ) / denominator
    endpoint_current = float(gmyo_ms) * (endpoints - boundary_new)
    load[0] += endpoint_current[0]
    load[-1] += endpoint_current[1]
    derivative = torch.full(
        (2,),
        float(gmyo_ms) * c_over_dt / denominator,
        device=voltage_mv.device,
        dtype=voltage_mv.dtype,
    )
    return load, boundary_new, derivative


def choose_electrical_dt(
    time_ms: float,
    bcl_ms: float,
    dt_fine_ms: float,
    dt_coarse_ms: float,
    fine_window_ms: float,
    adaptive_dt: bool,
) -> tuple[float, float]:
    """Return the fixed or phase-dependent micro-step and next switch."""
    cycle = math.floor((time_ms + 1.0e-12) / bcl_ms)
    cycle_start = cycle * bcl_ms
    phase = time_ms - cycle_start
    window_end = min(cycle_start + fine_window_ms, cycle_start + bcl_ms)
    cycle_end = cycle_start + bcl_ms
    if not adaptive_dt:
        return dt_fine_ms, cycle_end
    if phase >= fine_window_ms - 1.0e-12:
        return dt_coarse_ms, cycle_end
    return dt_fine_ms, window_end
