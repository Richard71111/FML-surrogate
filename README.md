# FML ID Local Surrogate

A neural flow map (FML) surrogate for the intercalated-disc cleft current
`Icleft`, and the cable solver that runs it in place of the resolved FEM
junction.

The repository holds **code only**. Training data, checkpoints and simulation
output are generated artefacts and are excluded by `.gitignore`; they are
reproducible from this code plus the raw FEM voltage-clamp dataset.

## Layout

```
config.py                  single source of truth for shapes, paths and names
                           (imported by both halves through thin shims)

fml/                       learning the flow map
  01_data_generation/      raw FEM .mat -> burst NPZ on the DT_FML clock
  02_training/             recurrent training and cable fine-tuning
  03_evaluation/           checkpoint scoring and loss curves

Numerical_simulation/      using the flow map
  run_simulation.py        entry point for the ORd11 + FML cable
  solvers/                 macro-endpoint integrator with rollback line search
  fml_model/               checkpoint loader and rolling junction histories
  ionic_model/             ORd11 membrane model
  analysis/                comparison against the MATLAB reference cable
  visualization/           plots
  docs/                    numerical method write-up
```

`config.py` lives at the repository root because both halves reach it as
`../../config.py`; keep it there.

## Pipeline

1. **Generate the FEM dataset** — MATLAB, in the companion repository
   [`FEM-ID-Hetg-tissue-simulation`](https://github.com/Richard71111/FEM-ID-Hetg-tissue-simulation).
   One voltage-clamped cell, one free cell, per mesh and gap-junction
   coupling.
2. **Build training NPZs** — `fml/01_data_generation/submit.sh`, which
   resamples onto the flow map's own step `DT_FML` and cuts bursts.
3. **Train** — `fml/02_training/submit_train.sh`; optionally fine-tune on real
   cable action potentials with `run_finetune.sh`.
4. **Simulate** — `Numerical_simulation/run_numerical_simulation.sh`.

## Time steps

Three different steps appear, and they are independent:

| symbol | what it is | where it is set |
|---|---|---|
| `dt` | FEM voltage/ODE step | MATLAB generator (`DT_FEM`) |
| `dtS` | explicit sub-step of the cleft **concentration** update | derived from the mesh by `cleft_substep_count.m` |
| `DT_FML` | the flow map's macro step and the NPZ sampling interval | `--dt` here |

`dtS` is a stability constraint, not an accuracy choice: the cleft
concentration update is explicit forward Euler, and its limit scales with the
mesh conductance/volume ratio. Pinning `dtS = dt` puts the
higher-conductance meshes past that limit and produces a qualitatively wrong
trajectory rather than a merely coarse one.

The current datasets are generated at `dt = 0.005 ms`, `dtS = 0.001 ms`, and
trained at `DT_FML = 0.05 ms`.

## Conventions

- **Ctrl contract**: every saved burst has one more Ctrl sample than QoI
  samples. The extra sample is the known target-time voltage the flow map
  conditions on.
- **QoI** is `Icleft`, two independently predicted per-junction side currents;
  they are deliberately not constrained to be equal and opposite, because the
  cleft is its own extracellular compartment.
- Paths to the shared dataset and mesh library are absolute and point at the
  OSC `/fs/ess/PAS1622` filesystem; change them in `config.py` to run
  elsewhere.
