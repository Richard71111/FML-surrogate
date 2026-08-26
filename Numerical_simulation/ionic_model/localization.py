"""Intercalated-disc (ID) channel-localization fractions for ORd11.

Standalone reference implementation of the disc/axial current split used by
``run_generic_RCL_multi_mesh.m`` (SHWeinberg/ID_Cleft_Model_2025 on GitHub),
lines 96-99 and 314-322:

    locINa   = 0.7
    locIK1   = 0.2
    locICa   = 0.2
    locINaK  = 0.2
    locUniform = 2*Adisc/Atot   % Ito, IKr, IKs: distributed by membrane area

That MATLAB model splits each cell's channels between the disc-facing
membrane (feeds the cleft/ephaptic coupling) and the axial/lateral membrane
(feeds that cell's own "regular" transmembrane current), using ``loc`` as the
disc fraction. This module computes the COMPLEMENTARY axial fraction
(``1 - loc``) per current, in the same 14-current order
``ionic_model.adapter.ORd11CableIonicModel`` uses for its ``f_I`` parameter, so a
caller can build an ``f_I`` vector that matches the axial-only share of
INa/IK1/ICaL/INaK/Ito/IKr/IKs the MATLAB reference would send to a cell's own
membrane. Every other current is untouched (loc=0 in the MATLAB source, so
its full axial fraction is 1.0).

The reduced MATLAB-equivalent axial KCL uses this complementary axial
fraction together with the raw axial-to-ID port current.  The unused terminal
face is a separate active membrane patch and uses ``loc / cell_port_count``;
for a 1-D cable ``cell_port_count=2``.  The stimulus is independent of these
fractions and remains axial-only.
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# ID-localization fractions (disc-facing membrane share of each current).
# Source: run_generic_RCL_multi_mesh.m, SHWeinberg/ID_Cleft_Model_2025.
# ---------------------------------------------------------------------------
LOC_INA = 0.7
LOC_IK1 = 0.2
LOC_ICAL = 0.2
LOC_INAK = 0.2
# locUniform = 2*Adisc/Atot is geometry-dependent; see disc_area_fraction()
# and locUniform_fraction() below rather than a fixed constant.

# Current name -> index, matching adapter.ORd11CableIonicModel's
# "iina"/"iinal"/... parameter map (0-based; MATLAB's p.iina etc. are the
# same order, 1-based).
CURRENT_INDEX = {
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
}
NUM_CURRENTS = len(CURRENT_INDEX)

# Currents explicitly localized in the MATLAB source (everything else has
# loc=0, i.e. stays 100% axial).
_EXPLICIT_DISC_FRACTIONS = {
    "iina": LOC_INA,
    "iik1": LOC_IK1,
    "iical": LOC_ICAL,
    "iinak": LOC_INAK,
}
# Ito, IKr, IKs: distributed proportional to membrane area (locUniform),
# not a fixed constant -- computed per-geometry, see disc_uniform_fraction().
_AREA_PROPORTIONAL_CURRENTS = ("iito", "iikr", "iiks")


def disc_uniform_fraction(disc_area_um2: float, total_area_um2: float) -> float:
    """``locUniform = 2*Adisc/Atot`` -- disc fraction assumed for Ito/IKr/IKs."""
    return 2.0 * disc_area_um2 / total_area_um2


def build_disc_fractions(
    disc_area_um2: float, total_area_um2: float
) -> list[float]:
    """Disc-localized fraction of each current, in CURRENT_INDEX order."""
    loc_uniform = disc_uniform_fraction(disc_area_um2, total_area_um2)
    disc_frac = [0.0] * NUM_CURRENTS
    for name, value in _EXPLICIT_DISC_FRACTIONS.items():
        disc_frac[CURRENT_INDEX[name]] = value
    for name in _AREA_PROPORTIONAL_CURRENTS:
        disc_frac[CURRENT_INDEX[name]] = loc_uniform
    return disc_frac


def build_axial_f_i(disc_area_um2: float, total_area_um2: float) -> list[float]:
    """Axial (non-disc) fraction of each current -- ``1 - disc_fraction``.

    This is the vector that would replace ``f_I = ones(14)`` in
    ``adapter.py`` if the reduced cell equation's Iion is meant to
    carry only the axial-membrane share of each current. See the module
    docstring: tested directly, breaks AP firing in the current
    single-compartment-per-cell model -- do not wire this in without also
    resolving how the disc share is reintroduced (see the docstring).
    """
    return [1.0 - d for d in build_disc_fractions(disc_area_um2, total_area_um2)]


def build_axial_f_i_from_geometry(geometry) -> list[float]:
    """Convenience wrapper taking an ``adapter.CellGeometry`` instance
    (or any object exposing ``.disc_area_um2`` / ``.total_area_um2``)."""
    return build_axial_f_i(geometry.disc_area_um2, geometry.total_area_um2)


def build_axial_f_i_tensor(geometry, device=None, dtype=None):
    """Same as ``build_axial_f_i_from_geometry``, returned as a torch tensor
    with the given device/dtype, ready to assign to
    ``ORd11CableIonicModel.parameters["f_I"]`` for experimentation."""
    import torch

    values = build_axial_f_i_from_geometry(geometry)
    return torch.tensor(values, device=device, dtype=dtype)


def build_boundary_f_i_tensor(
    geometry, cell_port_count: int = 2, device=None, dtype=None
):
    """MATLAB terminal-disc fractions ``loc / cell_port_count``."""
    import torch

    if cell_port_count <= 0:
        raise ValueError("cell_port_count must be positive.")
    disc = build_disc_fractions(geometry.disc_area_um2, geometry.total_area_um2)
    values = [value / float(cell_port_count) for value in disc]
    return torch.tensor(values, device=device, dtype=dtype)


def summarize(geometry, currents: Sequence[str] | None = None) -> str:
    """Human-readable table of disc/axial fractions for the given geometry,
    for the currents in ``currents`` (default: all 14)."""
    disc = build_disc_fractions(geometry.disc_area_um2, geometry.total_area_um2)
    names = currents or list(CURRENT_INDEX)
    lines = [f"{'current':>10}  {'disc':>8}  {'axial':>8}"]
    for name in names:
        idx = CURRENT_INDEX[name]
        lines.append(f"{name:>10}  {disc[idx]:8.4f}  {1.0 - disc[idx]:8.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual check against the default 100um x 11um cell geometry.
    from .adapter import CellGeometry

    geo = CellGeometry()
    print(f"disc_area_um2={geo.disc_area_um2:.4f}  "
          f"total_area_um2={geo.total_area_um2:.4f}  "
          f"locUniform={disc_uniform_fraction(geo.disc_area_um2, geo.total_area_um2):.4f}")
    print(summarize(geo))
