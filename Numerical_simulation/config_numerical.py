"""Fixed physical and numerical settings for the production cable solver.

Edit this file when intentionally changing the discretization, geometry, or
solver tolerances.  Runtime/pacing choices and FML checkpoint metadata remain
command-line options exposed by ``run_numerical_simulation.sh``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


DEFAULT_FEM_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "FEM-ID-Hetg-tissue-simulation"
    / "data"
    / "384"
)


@dataclass(frozen=True)
class NumericalConfig:
    """Single source of truth for the validated macro endpoint solver."""

    macro_reconstruction: str = "linear"

    # Electrical/ionic micro-grid. Inside each pacing fine window the solver
    # uses dt_fine_ms. At runtime, dt_coarse_ms is replaced by the selected
    # model's DT_FML so one coarse electrical step equals one FML macro step.
    adaptive_dt: bool = True
    dt_fine_ms: float = 0.01
    dt_coarse_ms: float = 0.1
    fine_window_ms: float = 50.0

    # MATLAB/FEM physical configuration.
    rho_myo_kohm_um: float = 1500.0
    terminal_boundary: bool = True
    fem_data_dir: str = str(DEFAULT_FEM_DATA_DIR)
    length_um: float = 100.0
    radius_um: float = 11.0
    cm_density_uf_per_um2: float = 1.0e-8
    subtract_rest_bias: bool = False

    # Macro endpoint quasi-Newton/rollback solve.
    coupling_iters: int = 300
    coupling_tol_mv: float = 1.0e-6
    coupling_relax: float = 0.5
    coupling_min_relax: float = 1.0e-3
    fail_on_nonconvergence: bool = True

    # Execution/output policy. FAST_RUN changes only output overhead.
    device: str = "cpu"
    torch_threads: int = 1
    compile_ord11: str = "auto"
    compile_threshold_steps: int = 20_000
    progress_every: int = 0
    timing_interval_ms: float = 500.0
    record_fml_only: bool = False
    record_last_beat_only: bool = False
    make_plots: bool = True

    def for_fml_dt(self, dt_fml_ms: float) -> "NumericalConfig":
        """Use the model clock as the electrical step outside fine windows."""
        dt_fml_ms = float(dt_fml_ms)
        if dt_fml_ms <= 0.0:
            raise ValueError("DT_FML must be positive.")
        if dt_fml_ms + 1.0e-12 < self.dt_fine_ms:
            raise ValueError(
                f"DT_FML={dt_fml_ms:g} ms is smaller than the configured "
                f"fine step {self.dt_fine_ms:g} ms."
            )
        return replace(self, dt_coarse_ms=dt_fml_ms)

    def for_fast_run(self, enabled: bool) -> "NumericalConfig":
        """Disable dense output/plots without changing numerical physics."""
        if not enabled:
            return self
        return replace(self, record_fml_only=True, make_plots=False)

    def for_last_beat_only(self, enabled: bool) -> "NumericalConfig":
        """Retain trajectory samples only from the final pacing cycle."""
        return replace(self, record_last_beat_only=bool(enabled))

    def load_boundary_area_um2(self, mesh_idx: int) -> tuple[float, Path]:
        """Load ``sum(FEM_data.partition_surface)`` for the selected mesh."""
        import numpy as np
        from scipy.io import loadmat

        mesh_idx = int(mesh_idx)
        if mesh_idx <= 0:
            raise ValueError("MESH_IDX must be a positive integer.")

        mesh_path = (
            Path(self.fem_data_dir).expanduser()
            / f"FEMDATA_{mesh_idx}.mat"
        ).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                "Cannot load terminal boundary area: expected mesh file "
                f"{mesh_path}"
            )

        contents = loadmat(
            mesh_path,
            squeeze_me=True,
            struct_as_record=False,
            variable_names=("FEM_data",),
        )
        fem_data = contents.get("FEM_data")
        partition_surface = getattr(
            fem_data, "partition_surface", None
        )
        if partition_surface is None:
            raise KeyError(
                f"{mesh_path} does not contain "
                "FEM_data.partition_surface."
            )

        surface = np.asarray(
            partition_surface, dtype=np.float64
        ).reshape(-1)
        if surface.size == 0 or not np.all(np.isfinite(surface)):
            raise ValueError(
                "FEM_data.partition_surface must be nonempty and finite in "
                f"{mesh_path}"
            )
        area_um2 = float(np.sum(surface, dtype=np.float64))
        if area_um2 <= 0.0:
            raise ValueError(
                "sum(FEM_data.partition_surface) must be positive in "
                f"{mesh_path}; got {area_um2}."
            )
        return area_um2, mesh_path


NUMERICAL_CONFIG = NumericalConfig()
