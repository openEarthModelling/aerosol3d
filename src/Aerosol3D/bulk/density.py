# src/Aerosol3D/bulk/density.py
"""Effective (volume-weighted) density for bulk aerosol optics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from Aerosol3D.bulk.integration import integrate_distribution

if TYPE_CHECKING:
    from Aerosol3D.bulk.datastructs import SizeDistribution


def compute_effective_density(
    radii_nm: np.ndarray,
    densities_kg_m3: np.ndarray,
    size_distribution: SizeDistribution,
    n_quad: int = 256,
    method: str = "quad",
) -> float:
    """Volume-weighted effective density.

    .. math:: \\rho_{eff} = \\frac{\\int \\rho(r) r^3 n(r) dr}{\\int r^3 n(r) dr}

    where ``n(r)`` is the size-distribution PDF. For a single material
    (constant density), reduces to that material's density.

    Args:
        radii_nm: Sample radii (nm) at which density is known.
        densities_kg_m3: Material density (kg/m³) at each sample radius.
        size_distribution: Size distribution providing the PDF and bounds.
        n_quad: Quadrature points for ``method="fixed_quad"``.
        method: ``"quad"`` (adaptive) or ``"fixed_quad"``.

    Notes:
        Density is linearly interpolated over ``radii_nm`` and extrapolated
        linearly outside the sampled range; callers should ensure
        ``radii_nm`` spans the size distribution's significant support to
        avoid relying on extrapolated values in the tails.

    Returns:
        Effective density in kg/m³.
    """
    radii_nm = np.asarray(radii_nm, dtype=float)
    densities_kg_m3 = np.asarray(densities_kg_m3, dtype=float)
    order = np.argsort(radii_nm)
    radii_nm = radii_nm[order]
    densities_kg_m3 = densities_kg_m3[order]

    if np.allclose(densities_kg_m3, densities_kg_m3[0]):
        return float(densities_kg_m3[0])

    rho_of_r = interp1d(radii_nm, densities_kg_m3, kind="linear", fill_value="extrapolate")

    num = integrate_distribution(lambda r: rho_of_r(r) * r**3, size_distribution, n_quad, method)
    den = size_distribution.moment(3)
    if den <= 0:
        raise ValueError("Size distribution volume moment is non-positive")
    return float(num / den)
