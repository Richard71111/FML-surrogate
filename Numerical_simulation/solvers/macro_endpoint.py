"""Macro-step FML / micro-step ORd11 cable integrator.

The neural flow map is evaluated and committed only on its trained clock H.
Inside a macro interval, the two endpoint port currents are linearly
reconstructed on the electrical micro-grid.  Because the right endpoint
current depends on the unknown right endpoint voltage, a macro fixed-point
iteration solves the coupled endpoint problem.  Ionic and boundary states are
rolled back between trials and are committed only after the macro interval has
converged.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from Numerical_simulation.solvers.core import (
    CableResult,
    build_cleft_incidence,
    choose_electrical_dt,
    terminal_boundary_load,
)
from Numerical_simulation.fml_model.junction import FMLJunctionBatch
from Numerical_simulation.ionic_model.adapter import (
    ORd11CableIonicModel,
    current_and_advance_models,
)


def _geometry_constants(
    ionic_model: ORd11CableIonicModel,
    rho_myo_kohm_um: float,
    boundary_area_um2: float | None,
) -> tuple[float, float, float, float]:
    capacitance = float(ionic_model.capacitance_uf)
    if boundary_area_um2 is None:
        boundary_area_um2 = ionic_model.geometry.disc_area_um2
    boundary_area_um2 = float(boundary_area_um2)
    if boundary_area_um2 <= 0.0:
        raise ValueError("boundary_area_um2 must be positive.")
    boundary_capacitance = (
        boundary_area_um2 * ionic_model.geometry.cm_density_uf_per_um2
    )
    r_um = ionic_model.geometry.radius_um
    l_um = ionic_model.geometry.length_um
    rmyo_kohm = float(rho_myo_kohm_um) * (l_um / 2.0) / (math.pi * r_um**2)
    return capacitance, boundary_capacitance, boundary_area_um2, 1.0 / rmyo_kohm


@torch.no_grad()
def simulate_macro_endpoint_linear(
    ionic_model: ORd11CableIonicModel,
    boundary_ionic_model: ORd11CableIonicModel,
    junction_model: FMLJunctionBatch,
    bcl_ms: float,
    nbeats: int,
    progress_every: int = 0,
    timing_interval_ms: float = 500.0,
    coupling_iters: int = 100,
    coupling_tol: float = 1.0e-6,
    coupling_relax: float = 0.5,
    coupling_min_relax: float = 1.0e-3,
    fail_on_nonconvergence: bool = True,
    *,
    dt_fine_ms: float = 0.01,
    dt_coarse_ms: float = 0.1,
    fine_window_ms: float = 50.0,
    adaptive_dt: bool = True,
    fml_dt_ms: float = 0.1,
    rho_myo_kohm_um: float = 1500.0,
    terminal_boundary: bool = True,
    boundary_area_um2: float | None = None,
    record_fml_only: bool = False,
    record_last_beat_only: bool = False,
) -> CableResult:
    """Integrate with one converged FML endpoint per trained macro step.

    For every interval ``[t_k,t_{k+1}]`` of length ``fml_dt_ms``:

    1. predict ``Icleft(t_{k+1})`` from the fixed FML history and a candidate
       endpoint voltage;
    2. reconstruct Icleft linearly from its accepted left endpoint;
    3. integrate ORd11/voltage on the fine or coarse electrical grid;
    4. iterate the endpoint voltage, restoring all ORd11 states between
       trials; and
    5. commit exactly one current/ctrl sample to FML history.

    No FML call at a micro-step is allowed to redefine the macro endpoint
    from that micro-step's voltage.
    """
    for name, value in (
        ("bcl_ms", bcl_ms),
        ("dt_fine_ms", dt_fine_ms),
        ("dt_coarse_ms", dt_coarse_ms),
        ("fml_dt_ms", fml_dt_ms),
        ("rho_myo_kohm_um", rho_myo_kohm_um),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if nbeats <= 0 or coupling_iters <= 0:
        raise ValueError("nbeats and coupling_iters must be positive.")
    if not (0.0 < coupling_relax <= 1.0):
        raise ValueError("coupling_relax must lie in (0,1].")
    if not (0.0 < coupling_min_relax <= coupling_relax):
        raise ValueError("coupling_min_relax must lie in (0,coupling_relax].")
    if not (0.0 <= fine_window_ms <= bcl_ms):
        raise ValueError("fine_window_ms must lie in [0,bcl_ms].")
    if not np.isclose(float(bcl_ms), ionic_model.stimulus_bcl_ms):
        raise ValueError("Simulation and ORd11 stimulus BCL must agree.")
    if ionic_model.n_cells != junction_model.n_cells:
        raise ValueError("Ionic and junction models use different cell counts.")
    if boundary_ionic_model.n_cells != 2:
        raise ValueError("Boundary ionic model must contain two patches.")
    macro_count = float(bcl_ms) * int(nbeats) / float(fml_dt_ms)
    if not np.isclose(macro_count, round(macro_count), rtol=0.0, atol=1.0e-9):
        raise ValueError("Total time must be an integer number of FML intervals.")

    n_cells = ionic_model.n_cells
    device, dtype = ionic_model.device, ionic_model.dtype
    sigma = build_cleft_incidence(n_cells, device, dtype)
    (
        capacitance,
        boundary_capacitance,
        boundary_area_um2,
        gmyo_ms,
    ) = _geometry_constants(ionic_model, rho_myo_kohm_um, boundary_area_um2)

    voltage = torch.full(
        (n_cells,), ionic_model.rest_voltage_mv, device=device, dtype=dtype
    )
    boundary_voltage = torch.full(
        (2,), ionic_model.rest_voltage_mv, device=device, dtype=dtype
    )

    # The Ctrl-now map can determine Icleft(0) from the
    # pre-t=0 rest histories and phi(0).  This accepted sample becomes the
    # left endpoint of the first macro interval.
    cleft_left, ctrl_zero = junction_model.predict_current(voltage)
    junction_model.commit(cleft_left, ctrl_zero)

    save_from_ms = (
        float(max(0, int(nbeats) - 1)) * float(bcl_ms)
        if record_last_beat_only
        else 0.0
    )
    retain_initial = save_from_ms <= 1.0e-12
    time_hist = [0.0] if retain_initial else []
    voltage_hist = [voltage.cpu().numpy().copy()] if retain_initial else []
    boundary_voltage_hist = (
        [boundary_voltage.cpu().numpy().copy()] if retain_initial else []
    )
    current_time_hist: list[float] = []
    dt_hist: list[float] = []
    ionic_hist: list[np.ndarray] = []
    cleft_hist: list[np.ndarray] = []
    cell_cleft_hist: list[np.ndarray] = []
    boundary_current_hist: list[np.ndarray] = []
    boundary_ionic_hist: list[np.ndarray] = []
    stimulus_hist: list[np.ndarray] = []
    fml_commit_time_hist: list[float] = []
    coupling_iterations_hist: list[int] = []
    coupling_converged_hist: list[bool] = []
    coupling_residual_hist: list[float] = []
    iteration_times: list[float] = []

    def restore_states(axial_gates: torch.Tensor, boundary_gates: torch.Tensor) -> None:
        ionic_model.gating_state = axial_gates.clone()
        boundary_ionic_model.gating_state = boundary_gates.clone()

    def integrate_interval(
        t_left: float,
        v_left: torch.Tensor,
        vb_left: torch.Tensor,
        q_left: torch.Tensor,
        q_right: torch.Tensor,
        axial_gates: torch.Tensor,
        boundary_gates: torch.Tensor,
        *,
        retain_trajectory: bool,
    ):
        """Run one rollback-safe macro trial with prescribed endpoint currents."""
        restore_states(axial_gates, boundary_gates)
        v = v_left.clone()
        vb = vb_left.clone()
        t = float(t_left)
        t_right = min(t + float(fml_dt_ms), t_end_ms)
        records = []
        while t < t_right - 1.0e-12:
            requested_dt, next_switch = choose_electrical_dt(
                t,
                float(bcl_ms),
                float(dt_fine_ms),
                float(dt_coarse_ms),
                float(fine_window_ms),
                bool(adaptive_dt),
            )
            target = min(t + requested_dt, next_switch, t_right)
            for event in (next_switch, t_right):
                if abs(target - event) <= 1.0e-9:
                    target = float(event)
                    break
            dt = target - t
            if dt <= 1.0e-12:
                raise RuntimeError(
                    f"Non-positive macro micro-step at t={t:.12f}; target={target:.12f}."
                )
            alpha = float((target - t_left) / (t_right - t_left))
            q = q_left + alpha * (q_right - q_left)
            cell_q = sigma @ q.reshape(-1)

            ionic_model.set_dt_ms(dt)
            if terminal_boundary:
                boundary_ionic_model.set_dt_ms(dt)
                (iion, iion_boundary), _ = current_and_advance_models(
                    t, (ionic_model, boundary_ionic_model), (v, vb)
                )
            else:
                iion, _ = ionic_model.current_and_advance_gates(t, v)
                iion_boundary = torch.zeros(2, device=device, dtype=dtype)
            stimulus = ionic_model.stimulus_current_ua(t)

            # Interior voltage is explicit in Iion and prescribed Icleft.
            # The two linear terminal boundary nodes are eliminated
            # implicitly, exactly as in the production IMEX solver.
            v_new = v - (dt / capacitance) * (iion + cell_q)
            boundary_load, vb_new, boundary_derivative = terminal_boundary_load(
                v_new,
                vb,
                iion_boundary,
                dt,
                boundary_capacitance,
                gmyo_ms,
                terminal_boundary,
            )
            residual = v_new - v + (dt / capacitance) * (
                iion + cell_q + boundary_load
            )
            diagonal = torch.ones_like(v_new)
            if terminal_boundary:
                diagonal[0] += (dt / capacitance) * boundary_derivative[0]
                diagonal[-1] += (dt / capacitance) * boundary_derivative[1]
            v_new = v_new - residual / diagonal
            boundary_load, vb_new, _ = terminal_boundary_load(
                v_new,
                vb,
                iion_boundary,
                dt,
                boundary_capacitance,
                gmyo_ms,
                terminal_boundary,
            )

            if not (
                torch.isfinite(v_new).all()
                and torch.isfinite(vb_new).all()
                and torch.isfinite(q).all()
            ):
                raise FloatingPointError(
                    f"NaN/Inf in macro trial at t={target:.6f} ms."
                )
            if retain_trajectory:
                records.append(
                    (
                        float(target),
                        float(dt),
                        v_new.cpu().numpy().copy(),
                        vb_new.cpu().numpy().copy(),
                        iion.cpu().numpy().copy(),
                        q.cpu().numpy().copy(),
                        cell_q.cpu().numpy().copy(),
                        np.array(
                            [boundary_load[0].item(), boundary_load[-1].item()],
                            dtype=np.float64,
                        ),
                        iion_boundary.cpu().numpy().copy(),
                        stimulus.cpu().numpy().copy(),
                    )
                )
            v, vb, t = v_new, vb_new, float(target)
        return v, vb, records

    t_end_ms = float(bcl_ms) * int(nbeats)
    time_ms = 0.0
    macro_step = 0
    accepted_micro_steps = 0
    next_timing_ms = timing_interval_ms if timing_interval_ms > 0.0 else math.inf
    simulation_start = time.perf_counter()

    while time_ms < t_end_ms - 1.0e-12:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        macro_start = time.perf_counter()
        v_start = voltage.clone()
        vb_start = boundary_voltage.clone()
        axial_gates_start = ionic_model.gating_state.clone()
        boundary_gates_start = boundary_ionic_model.gating_state.clone()

        # A causal, current-hold micro solve supplies a good initial endpoint
        # voltage.  It is disposable: all ionic states are restored before
        # the first coupled trial.
        endpoint_guess, _, _ = integrate_interval(
            time_ms,
            v_start,
            vb_start,
            cleft_left,
            cleft_left,
            axial_gates_start,
            boundary_gates_start,
            retain_trajectory=False,
        )
        restore_states(axial_gates_start, boundary_gates_start)

        relax = float(coupling_relax)
        previous_residual = math.inf
        converged = False
        accepted_records = []
        accepted_boundary = vb_start
        cleft_right = cleft_left
        max_residual = math.inf
        identity = torch.eye(n_cells, device=device, dtype=dtype)
        junction_index = torch.arange(n_cells - 1, device=device)
        for outer in range(int(coupling_iters)):
            cleft_trial, _, fml_jac = (
                junction_model.predict_current_and_jacobian(
                    endpoint_guess
                )
            )
            endpoint_trial, boundary_trial, trial_records = integrate_interval(
                time_ms,
                v_start,
                vb_start,
                cleft_left,
                cleft_trial,
                axial_gates_start,
                boundary_gates_start,
                retain_trajectory=True,
            )
            # Root form R(v_R) = v_R - Phi_H(v_R).  Phi_H includes the
            # rollback micro integration with q_R=FML(v_R).
            residual_vec = endpoint_guess - endpoint_trial
            max_residual = float(residual_vec.abs().max().item())
            residual_norm = float(torch.linalg.vector_norm(residual_vec).item())
            used_iters = outer + 1
            accepted_records = trial_records
            accepted_boundary = boundary_trial
            cleft_right = cleft_trial
            if max_residual < float(coupling_tol):
                converged = True
                voltage = endpoint_trial
                break

            # Approximate macro Newton Jacobian.  With target-time current on
            # each micro-step, the coefficient multiplying q_R is exactly
            # sum(dt_r*alpha_r).  Iion trajectory sensitivities are omitted;
            # a line search checks the complete rollback integration, so this
            # remains a safe quasi-Newton direction rather than an assumed
            # exact derivative.
            macro_width = min(float(fml_dt_ms), t_end_ms - time_ms)
            current_weight_ms = sum(
                record[1] * ((record[0] - time_ms) / macro_width)
                for record in trial_records
            )
            dcleft_dv = torch.zeros(
                (2 * (n_cells - 1), n_cells), device=device, dtype=dtype
            )
            for side in range(2):
                row = 2 * junction_index + side
                dcleft_dv[row, junction_index] = fml_jac[:, side, 0]
                dcleft_dv[row, junction_index + 1] = fml_jac[:, side, 1]
            macro_jacobian = identity + (
                float(current_weight_ms) / capacitance
            ) * (sigma @ dcleft_dv)
            try:
                newton_delta = torch.linalg.solve(macro_jacobian, -residual_vec)
            except RuntimeError:
                newton_delta = torch.full_like(residual_vec, float("nan"))

            accepted_line_search = False
            if torch.isfinite(newton_delta).all():
                scale = 1.0
                for _ in range(10):
                    guess_line = endpoint_guess + scale * newton_delta
                    q_line, _ = junction_model.predict_current(
                        guess_line
                    )
                    endpoint_line, boundary_line, records_line = integrate_interval(
                        time_ms,
                        v_start,
                        vb_start,
                        cleft_left,
                        q_line,
                        axial_gates_start,
                        boundary_gates_start,
                        retain_trajectory=True,
                    )
                    residual_line = guess_line - endpoint_line
                    norm_line = float(
                        torch.linalg.vector_norm(residual_line).item()
                    )
                    if norm_line < residual_norm:
                        endpoint_guess = guess_line
                        endpoint_trial = endpoint_line
                        boundary_trial = boundary_line
                        trial_records = records_line
                        residual_vec = residual_line
                        max_residual = float(residual_line.abs().max().item())
                        residual_norm = norm_line
                        accepted_records = records_line
                        accepted_boundary = boundary_line
                        cleft_right = q_line
                        accepted_line_search = True
                        if max_residual < float(coupling_tol):
                            converged = True
                            voltage = endpoint_line
                        break
                    scale *= 0.5
            if converged:
                break
            if not accepted_line_search:
                # Contractive fallback where the approximate Newton tangent
                # is poor (most often during the sodium upstroke).
                if max_residual > 0.95 * previous_residual:
                    relax = max(0.5 * relax, float(coupling_min_relax))
                endpoint_guess = endpoint_guess - relax * residual_vec
            previous_residual = max_residual

        if not converged:
            message = (
                "WARNING: macro endpoint solve did not converge at "
                f"t={time_ms:.6f} ms after {used_iters} iterations "
                f"(residual={max_residual:.3e} mV, relax={relax:.3e})."
            )
            if fail_on_nonconvergence:
                raise RuntimeError(message)
            print(message, flush=True)
            voltage = endpoint_trial

        # The final retained trial owns the accepted ionic states.  Evaluate
        # once at its actual endpoint so the committed Ctrl is exactly the
        # accepted voltage rather than the last fixed-point guess.
        cleft_commit, ctrl_commit = junction_model.predict_current(voltage)
        endpoint_current_mismatch = float(
            (cleft_commit - cleft_right).abs().max().item()
        )
        if not np.isfinite(endpoint_current_mismatch):
            raise FloatingPointError("Non-finite endpoint current mismatch.")
        junction_model.commit(cleft_commit, ctrl_commit)
        cleft_left = cleft_commit
        boundary_voltage = accepted_boundary
        time_ms = min((macro_step + 1) * float(fml_dt_ms), t_end_ms)
        fml_commit_time_hist.append(float(time_ms))
        macro_step += 1

        for record_index, record in enumerate(accepted_records):
            (
                target,
                dt,
                v_np,
                vb_np,
                iion_np,
                q_np,
                cell_q_np,
                boundary_np,
                iion_boundary_np,
                stimulus_np,
            ) = record
            is_macro_endpoint = record_index == len(accepted_records) - 1
            if is_macro_endpoint:
                # Saved endpoint and committed endpoint must describe the same
                # accepted macro state/history pair.
                q_np = cleft_commit.cpu().numpy().copy()
                cell_q_np = (sigma @ cleft_commit.reshape(-1)).cpu().numpy().copy()
            in_retained_window = target >= save_from_ms - 1.0e-9
            if in_retained_window and ((not record_fml_only) or is_macro_endpoint):
                time_hist.append(target)
                voltage_hist.append(v_np)
                boundary_voltage_hist.append(vb_np)
                current_time_hist.append(target)
                dt_hist.append(dt)
                ionic_hist.append(iion_np)
                cleft_hist.append(q_np)
                cell_cleft_hist.append(cell_q_np)
                boundary_current_hist.append(boundary_np)
                boundary_ionic_hist.append(iion_boundary_np)
                stimulus_hist.append(stimulus_np)
            coupling_iterations_hist.append(int(used_iters))
            coupling_converged_hist.append(bool(converged))
            coupling_residual_hist.append(float(max_residual))
            accepted_micro_steps += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - macro_start
        per_micro = elapsed / max(1, len(accepted_records))
        iteration_times.extend([per_micro] * len(accepted_records))

        if time_ms + 1.0e-10 >= next_timing_ms:
            print(
                f"macro {macro_step:7d}  t={time_ms:10.3f} ms  "
                f"outer={used_iters:3d} residual={max_residual:.3e} mV  "
                f"wall={elapsed:.6f} s",
                flush=True,
            )
            while next_timing_ms <= time_ms + 1.0e-10:
                next_timing_ms += timing_interval_ms
        if progress_every and macro_step % progress_every == 0:
            print(
                f"macro {macro_step:7d} t={time_ms:10.3f}/{t_end_ms:.3f} ms "
                f"V=[{voltage.min().item():.3f},{voltage.max().item():.3f}] "
                f"outer={used_iters}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    simulation_elapsed = time.perf_counter() - simulation_start
    iteration_times_arr = np.asarray(iteration_times, dtype=np.float64)
    convergence_arr = np.asarray(coupling_converged_hist, dtype=bool)
    iterations_arr = np.asarray(coupling_iterations_hist, dtype=np.int64)
    residual_arr = np.asarray(coupling_residual_hist, dtype=np.float64)
    fml_times = np.asarray(fml_commit_time_hist, dtype=np.float64)
    expected = np.arange(1, fml_times.size + 1, dtype=np.float64) * float(fml_dt_ms)
    if not np.allclose(fml_times, expected, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("Macro solver committed FML history off its trained clock.")
    print(
        "Macro endpoint-linear timing: "
        f"total={simulation_elapsed:.6f} s, macros={macro_step}, "
        f"accepted_micro_steps={accepted_micro_steps}, "
        f"mean_outer={iterations_arr.mean():.3f}, "
        f"max_outer={iterations_arr.max()}, "
        f"max_residual={residual_arr.max():.3e} mV",
        flush=True,
    )

    return CableResult(
        time_ms=np.asarray(time_hist, dtype=np.float64),
        voltage_mv=np.asarray(voltage_hist, dtype=np.float64),
        boundary_voltage_mv=np.asarray(boundary_voltage_hist, dtype=np.float64),
        current_time_ms=np.asarray(current_time_hist, dtype=np.float64),
        time_step_ms=np.asarray(dt_hist, dtype=np.float64),
        ionic_current_ua=np.asarray(ionic_hist, dtype=np.float64),
        cleft_current_ua=np.asarray(cleft_hist, dtype=np.float64),
        cell_cleft_current_ua=np.asarray(cell_cleft_hist, dtype=np.float64),
        boundary_current_ua=np.asarray(boundary_current_hist, dtype=np.float64),
        boundary_ionic_current_ua=np.asarray(boundary_ionic_hist, dtype=np.float64),
        stimulus_current_ua=np.asarray(stimulus_hist, dtype=np.float64),
        fml_commit_time_ms=fml_times,
        sigma=sigma.cpu().numpy(),
        dt_ms=float(fml_dt_ms),
        dt_fine_ms=float(dt_fine_ms),
        dt_coarse_ms=float(dt_coarse_ms),
        fine_window_ms=float(fine_window_ms),
        capacitance_uf=float(capacitance),
        boundary_capacitance_uf=float(boundary_capacitance),
        boundary_area_um2=float(boundary_area_um2),
        gmyo_ms=float(gmyo_ms),
        bcl_ms=float(bcl_ms),
        nbeats=int(nbeats),
        iteration_wall_time_s=iteration_times_arr,
        simulation_wall_time_s=float(simulation_elapsed),
        coupling_iterations=iterations_arr,
        coupling_converged=convergence_arr,
        coupling_residual_mv=residual_arr,
        record_fml_only=bool(record_fml_only),
        record_last_beat_only=bool(record_last_beat_only),
    )
