# ORd11--FML cable solver

This directory has one production numerical method:
`macro_endpoint_linear`.  It is the method recorded by
`BCL200_nbeats5_nm30_nr6_dt0p02_finetune_true_job12466743`:

- the FML model advances only on its trained clock, $H=DT_{FML}$;
- the cleft current is reconstructed linearly between two FML endpoints;
- the unknown right endpoint is closed by rollback-safe quasi-Newton
  iteration, with line search and a relaxed fixed-point fallback;
- the electrical/ORd equations may use smaller micro-steps inside the fast
  pacing window without adding samples to the FML history.

## Equations

For axial cell voltages $V\in\mathbb{R}^{N}$, ORd11 states $G$, and the
two independently predicted currents at every junction
$q=(I_{\mathrm{cleft}}^L,I_{\mathrm{cleft}}^R)$, the reduced cable equation is

$$
C_{ax}\frac{dV}{dt}
=-\left[I_{ion}(V,G,t)+\Sigma q+I_b(V,V_b)\right].
$$

Here $\Sigma$ maps each raw left/right FML output only to its owner cell; the
code does **not** force the two currents to be opposite.  The axial membrane
capacitance is

$$
C_{ax}=2\pi rL C_m.
$$

The stimulus is included by the axial ORd11 model, is applied only to the
selected cell, and is not multiplied by the ionic-current localization factor
$f_I$.  Axial patches use the MATLAB-compatible $1-\mathrm{loc}$ factor.

## Macro endpoint and micro-step discretization

Let $t_{k+1}=t_k+H$.  Given accepted $q_k$, a candidate endpoint voltage
$\widehat V_{k+1}$ produces

$$
q_{k+1}=\mathcal F_{FML}(\text{history},\widehat V_{k+1}).
$$

On every electrical target time $t\in(t_k,t_{k+1}]$, the current is

$$
q(t)=(1-\alpha)q_k+\alpha q_{k+1},\qquad
\alpha=\frac{t-t_k}{H}.
$$

With micro-step $h$, the axial update uses current-time ORd11 current and
target-time reconstructed cleft current:

$$
\frac{C_{ax}}{h}(V^{n+1}-V^n)
+I_{ion}^{n}+\Sigma q^{n+1}+I_b^{n+1}=0.
$$

For fixed $q^{n+1}$, $I_{ion}^{n}$ and the non-terminal cells are Forward
Euler.  The repository ORd11 kernel advances concentrations by Forward Euler
and its gating variables by Rush--Larsen.  The linear terminal load is
eliminated implicitly.  Thus, `linear` describes the within-macro current
reconstruction; it does not mean that the whole solver is a second-order
method.

The endpoint residual is

$$
R(\widehat V_{k+1})=\widehat V_{k+1}
-\Phi_H\!\left(q_k,\mathcal F_{FML}(\widehat V_{k+1})\right).
$$

A current-hold integration supplies the initial voltage guess.  Every trial
restores the same ORd11 and boundary states, integrates the whole macro
interval, and evaluates $R$.  A quasi-Newton step uses the FML Jacobian;
line search checks the full rollback integration.  If that direction is poor,
a relaxed fixed-point step is used.  After convergence, the accepted FML
current and voltage are committed to history exactly once.

## Terminal boundary condition

Each cable end is connected through half-cell myoplasmic conductance
$g_{myo}$ to an active ORd11 terminal-disc voltage $V_b$:

$$
C_b\frac{V_b^{n+1}-V_b^n}{h}
=-I_{ion,b}^{n}-g_{myo}(V_b^{n+1}-V_e^{n+1}),
$$

or equivalently

$$
V_b^{n+1}=
\frac{(C_b/h)V_b^n+g_{myo}V_e^{n+1}-I_{ion,b}^{n}}
{C_b/h+g_{myo}}.
$$

The endpoint load is $I_b=g_{myo}(V_e^{n+1}-V_b^{n+1})$.  This is the
MATLAB active terminal-disc boundary, not a zero-flux boundary.  Geometry,
capacitance, $g_{myo}$, time-step policy, and nonlinear tolerances are fixed
in `config_numerical.py`.

## Code flow

1. `run_numerical_simulation.sh` supplies pacing and checkpoint metadata.
2. `run_simulation.py` constructs the axial and terminal ORd11 models, loads
   the FML checkpoint, and calls the single solver.
3. `solvers/macro_endpoint.py` initializes $q_0$, performs the macro
   endpoint solve, integrates all micro-steps, and commits one FML sample per
   $H$.
4. `solvers/core.py` supplies the owner-current map, electrical step selector,
   and implicit terminal boundary elimination.
5. Results and numerical metadata are written to `output/.../simulation.npz`
   and `summary.json`; optional MATLAB comparison is performed afterward.

The standard configuration uses $h=0.01$ ms during the first 50 ms of every
beat and $h=H$ outside that window.  These values are configured in
`config_numerical.py`; `DT_FML` must match the selected model's training step.

## FEM/GJ model and output layout

The launch scripts accept integer `MESH_IDX` (aliases: `FEM_MESH`, `FEM`) and
case-insensitive `GJ_COUPLING=strong|weak` (alias: `GJ`). They derive `MODEL_NAME_BASE`
from these values and the network settings, so it should not be supplied
manually.  Results and Slurm logs use the same hierarchy as training logs:

```
Numerical_simulation/output/FEM<MESH_IDX>/GJ<coupling>/<run>/
Numerical_simulation/logs/FEM<MESH_IDX>/GJ<coupling>/<job>_<jobid>.out
Numerical_simulation/logs/FEM<MESH_IDX>/GJ<coupling>/<job>_<jobid>.err
```
