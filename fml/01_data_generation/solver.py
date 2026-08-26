"""
Python Physical Twin for Case 3: Van der Pol Oscillator.

System: dx1/dt = x2,  dx2/dt = mu*(1 - x1^2)*x2 - x1 + u
mu = 1.0 (fixed Van der Pol parameter)

NONLINEAR system — uses RK4 integration (no exact discretization).

Three time scales:
    DT_STEP    = 0.01  (fine integration, used by RK4)
    DT_FML     = 0.1   (FML recording interval, 10 substeps)
    DT_CONTROL = 0.5   (control hold period, 5 FML steps)

Usage:
    python solver.py          # Run accuracy test
"""
import numpy as np


# ===========================================================================
# System parameter
# ===========================================================================

MU = 1.0  # Van der Pol nonlinearity parameter

DT_STEP = 0.01  # Fine integration step for discrete_step


# ===========================================================================
# ODE right-hand side
# ===========================================================================

def rhs(t, x, ctrl):
    """ODE right-hand side: Van der Pol with control.

    dx1/dt = x2
    dx2/dt = mu*(1 - x1^2)*x2 - x1 + ctrl
    """
    dxdt = np.zeros_like(x)
    dxdt[0] = x[1]
    dxdt[1] = MU * (1.0 - x[0]**2) * x[1] - x[0] + ctrl
    return dxdt


def rk4_step(t, x, dt, ctrl):
    """Single RK4 step with constant control."""
    k1 = rhs(t, x, ctrl)
    k2 = rhs(t + 0.5*dt, x + 0.5*dt*k1, ctrl)
    k3 = rhs(t + 0.5*dt, x + 0.5*dt*k2, ctrl)
    k4 = rhs(t + dt, x + dt*k3, ctrl)
    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ===========================================================================
# Discrete step: one FML step using RK4 substeps
# ===========================================================================

def discrete_step(x, u, dt):
    """Single FML-interval step using RK4 with fine substeps.

    Integrates from t to t+dt using dt_step=DT_STEP sub-steps.
    Control u is held constant over the interval.

    Args:
        x: State vector [N_STATES]
        u: Control scalar
        dt: FML interval (DT_FML = 0.1)

    Returns:
        x_next: State at t+dt
    """
    n_sub = round(dt / DT_STEP)
    assert abs(n_sub * DT_STEP - dt) < 1e-12, \
        f"dt ({dt}) must be integer multiple of DT_STEP ({DT_STEP})"

    ctrl = float(u)
    x_curr = np.array(x, dtype=np.float64)
    t = 0.0
    for _ in range(n_sub):
        x_curr = rk4_step(t, x_curr, DT_STEP, ctrl)
        t += DT_STEP
    return x_curr


# ===========================================================================
# Main solver: integrate with subsampling
# ===========================================================================

def solve(x0, ctrl_signal, T, dt_step, dt_fml):
    """
    Solve the ODE with RK4 and subsample at FML recording interval.

    Args:
        x0: Initial state [N_STATES]
        ctrl_signal: Array of control values at FML intervals [n_fml_steps+1]
                     or callable(t) -> scalar
        T: Total simulation time
        dt_step: Fine integration time step
        dt_fml: Coarser FML recording interval

    Returns:
        t_fml: Time points at dt_fml intervals [n_fml_steps + 1]
        x_fml: State at dt_fml intervals [N_STATES, n_fml_steps + 1]
    """
    ratio = round(dt_fml / dt_step)
    assert abs(ratio * dt_step - dt_fml) < 1e-12, \
        f"dt_fml ({dt_fml}) must be integer multiple of dt_step ({dt_step})"

    n_sim_steps = round(T / dt_step)
    n_fml_steps = round(T / dt_fml)
    n_states = len(x0)

    x_fml = np.zeros((n_states, n_fml_steps + 1), dtype=np.float64)
    t_fml = np.zeros(n_fml_steps + 1, dtype=np.float64)

    x = np.array(x0, dtype=np.float64)
    x_fml[:, 0] = x
    t_fml[0] = 0.0

    fml_idx = 1
    for step in range(1, n_sim_steps + 1):
        t = (step - 1) * dt_step

        # Get control at current FML interval
        fml_step_idx = (step - 1) // ratio
        if callable(ctrl_signal):
            ctrl = ctrl_signal(t)
        else:
            ctrl = ctrl_signal[min(fml_step_idx, len(ctrl_signal) - 1)]

        x = rk4_step(t, x, dt_step, ctrl)

        if step % ratio == 0:
            x_fml[:, fml_idx] = x
            t_fml[fml_idx] = step * dt_step
            fml_idx += 1

    return t_fml, x_fml


# ===========================================================================
# Accuracy test
# ===========================================================================

def accuracy_test():
    """Verify solver accuracy against known limit cycle behavior."""
    print("=" * 60)
    print("Van der Pol Solver Accuracy Test")
    print("=" * 60)

    dt_fml = 0.1

    # Test 1: RK4 convergence rate (u=0, autonomous)
    # dt_step must divide dt_fml evenly
    x0 = [2.0, 0.0]
    T = 10.0

    print(f"\nTest 1: RK4 convergence (u=0, x0={x0}, T={T})")
    print(f"  VDP limit cycle with mu={MU}")
    print(f"{'dt_step':>10s} {'substeps':>10s} {'x1(T)':>14s} {'x2(T)':>14s} {'Diff_x1':>12s}")

    prev_x1 = None
    for dt_step in [0.1, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001]:
        ctrl = np.zeros(round(T / dt_fml) + 1)
        t, x = solve(x0, ctrl, T, dt_step, dt_fml)

        x1_final = x[0, -1]
        x2_final = x[1, -1]
        n_sub = round(dt_fml / dt_step)

        if prev_x1 is not None:
            diff = abs(x1_final - prev_x1)
            print(f"{dt_step:10.5f} {n_sub:10d} {x1_final:14.10f} {x2_final:14.10f} {diff:12.6e}")
        else:
            print(f"{dt_step:10.5f} {n_sub:10d} {x1_final:14.10f} {x2_final:14.10f}          ---")
        prev_x1 = x1_final

    # Test 2: Verify limit cycle existence (u=0)
    print(f"\nTest 2: Limit cycle check (long run, u=0)")
    x0_lc = [0.1, 0.0]
    T_long = 50.0
    ctrl = np.zeros(round(T_long / dt_fml) + 1)
    t, x = solve(x0_lc, ctrl, T_long, DT_STEP, dt_fml)

    # Check if trajectory settles onto limit cycle (amplitude ~ 2 for mu=1)
    x1_late = x[0, -200:]  # last 200 FML steps
    amplitude = (np.max(x1_late) - np.min(x1_late)) / 2.0
    print(f"  Late-time x1 amplitude: {amplitude:.4f} (expected ~2.0 for mu={MU})")
    print(f"  x1 range: [{np.min(x1_late):.4f}, {np.max(x1_late):.4f}]")

    # Test 3: discrete_step consistency
    print(f"\nTest 3: discrete_step vs solve consistency")
    x0_ds = np.array([1.5, -1.0])
    u_test = 0.5
    n_steps = 100

    # Via discrete_step
    x_ds = x0_ds.copy()
    for _ in range(n_steps):
        x_ds = discrete_step(x_ds, u_test, dt_fml)

    # Via solve
    ctrl_const = np.full(n_steps + 1, u_test)
    T_test = n_steps * dt_fml
    t_sol, x_sol = solve(x0_ds, ctrl_const, T_test, DT_STEP, dt_fml)

    err_x1 = abs(x_ds[0] - x_sol[0, -1])
    err_x2 = abs(x_ds[1] - x_sol[1, -1])
    print(f"  discrete_step: x1={x_ds[0]:.10f}, x2={x_ds[1]:.10f}")
    print(f"  solve:         x1={x_sol[0,-1]:.10f}, x2={x_sol[1,-1]:.10f}")
    print(f"  Error: x1={err_x1:.3e}, x2={err_x2:.3e}")
    print(f"  (Should be near machine precision)")


if __name__ == '__main__':
    accuracy_test()
