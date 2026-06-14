"""Stage 1: Build bulk aerosol optics from a size distribution, save to NetCDF.

For each radius in RADII_NM, build an AerosolParticle, run Mie solves across
the wavelength grid, and aggregate per-radius AerosolOpticsData into a
BulkAerosolOpticsData via BulkOpticsBuilder. The bulk result auto-fills
``effective_density_kg_m3`` (Task 1-2) and ``legendre_moments_beta``
(Task 3-4, the g_l = k_l/(2l+1) form expected by libRadtran).

A second, monodisperse bulk (single radius) is also produced for the
Stage 3 comparison.

Usage:
    python compute_bulk_optics.py
"""

import logging

import numpy as np

from Aerosol3D import (
    AerosolParticle,
    MixingState,
    SimulationConfig,
    create_sphere,
    preset_material,
    solve_optics,
)
from Aerosol3D.bulk import BulkOpticsBuilder, SizeDistribution
from Aerosol3D.optics.optics_export import from_optical_results

from config import (
    BULK_OPTICS_NC,
    MATERIAL,
    MONODISPERSE_OPTICS_NC,
    N_LEGENDRE,
    RADII_NM,
    R_EFF_MONODISPERSE_NM,
    RG_NM,
    SIGMA_LN,
    WAVELENGTHS_NM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_particle(radius_nm: float, material_name: str) -> AerosolParticle:
    """Build a bare sphere AerosolParticle for the given radius and material.

    Uses the public top-level API:
      - ``AerosolParticle(name=..., mixing_state=..., unit="nm")``
      - ``particle.add_mesh("core", create_sphere((0, 0, 0), r), material)``
    """
    material = preset_material(material_name)
    particle = AerosolParticle(
        name=f"{material_name}_r{int(radius_nm)}nm",
        mixing_state=MixingState.INTERNAL,
        unit="nm",
    )
    particle.add_mesh("core", create_sphere((0, 0, 0), radius_nm), material)
    return particle


def _solve_wavelengths(particle: AerosolParticle, wavelengths_nm) -> list:
    """Run Mie optical solves for each wavelength, return list of OpticalResult."""
    results = []
    for wl in wavelengths_nm:
        config = SimulationConfig(
            wavelength=wl,
            source="solar",
            precision="medium",
        )
        result = solve_optics(
            particle,
            config,
            solver="MIE",
            voxel_size=None,
            compute_phase_func=True,
        )
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
    return results


def _build_bulk(radii_nm, material_name: str, sd: SizeDistribution, n_legendre: int):
    """Build a BulkAerosolOpticsData for the given radii and size distribution.

    Each per-radius entry is built via ``from_optical_results`` with the
    material density in kg/m^3 (Task 2 field), which lets the builder
    auto-compute ``effective_density_kg_m3`` (Task 1/4).
    """
    material = preset_material(material_name)
    density_kg_m3 = float(material.density) * 1000.0  # g/cm^3 -> kg/m^3

    builder = BulkOpticsBuilder(
        size_distribution=sd,
        radii_nm=np.asarray(radii_nm, dtype=float),
        n_legendre=n_legendre,
    )

    for r_nm in radii_nm:
        particle = _build_particle(float(r_nm), material_name)
        results = _solve_wavelengths(particle, WAVELENGTHS_NM)
        optics = from_optical_results(
            results,
            n_legendre=n_legendre,
            material_name=material_name,
            density_kg_m3=density_kg_m3,
        )
        builder.add(float(r_nm), optics)
        # Light per-radius logging
        idx_550 = int(np.argmin(np.abs(optics.wavelength_nm - 550.0)))
        logger.info(
            "  r=%6.1f nm: C_ext@550=%.4e nm^2, SSA@550=%.4f, g@550=%.4f",
            float(r_nm),
            float(optics.C_ext[idx_550]),
            float(optics.SSA[idx_550]),
            float(optics.g[idx_550]),
        )

    bulk = builder.compute()
    return bulk


def _log_bulk_summary(bulk, label: str) -> None:
    """Print key bulk fields, including Task 1-4 outputs."""
    idx_550 = int(np.argmin(np.abs(bulk.wavelength_nm - 550.0)))
    logger.info("[%s] bulk optics summary:", label)
    logger.info(
        "  wavelengths   : %d (%.0f-%.0f nm)",
        bulk.wavelength_nm.size,
        float(bulk.wavelength_nm[0]),
        float(bulk.wavelength_nm[-1]),
    )
    logger.info("  r_eff_nm      : %.3f", float(bulk.r_eff_nm))
    logger.info("  C_ext@550     : %.4e nm^2/particle", float(bulk.C_ext[idx_550]))
    logger.info("  SSA@550       : %.4f", float(bulk.SSA[idx_550]))
    logger.info("  g@550         : %.4f", float(bulk.g[idx_550]))
    logger.info(
        "  n_legendre    : %d (beta shape=%s)",
        bulk.n_legendre,
        tuple(bulk.beta.shape),
    )
    if bulk.legendre_moments_beta is not None:
        logger.info(
            "  legendre_moments_beta: shape=%s, beta_0=%.4f, beta_1=%.4f",
            tuple(bulk.legendre_moments_beta.shape),
            float(bulk.legendre_moments_beta[idx_550, 0]),
            float(bulk.legendre_moments_beta[idx_550, 1]),
        )
    else:
        logger.warning("  legendre_moments_beta is None (Task 3/4 wiring failed?)")
    if bulk.effective_density_kg_m3 is not None:
        logger.info("  effective_density_kg_m3: %.2f", float(bulk.effective_density_kg_m3))
    else:
        logger.warning("  effective_density_kg_m3 is None (Task 1/4 wiring failed?)")


def main():
    logger.info("Stage 1: building bulk aerosol optics (aerosol3D)")
    logger.info(
        "  material=%s, size dist=lognormal(rg=%.1f nm, sigma_ln=%.2f)",
        MATERIAL,
        RG_NM,
        SIGMA_LN,
    )
    logger.info("  radii (%d): %s", len(RADII_NM), RADII_NM)

    # ---- Bulk (lognormal size distribution) --------------------------------
    sd_bulk = SizeDistribution.lognormal(rg_nm=RG_NM, sigma_ln=SIGMA_LN)
    bulk = _build_bulk(RADII_NM, MATERIAL, sd_bulk, N_LEGENDRE)
    _log_bulk_summary(bulk, "bulk")

    bulk.to_netcdf(str(BULK_OPTICS_NC))
    logger.info("Saved bulk optics -> %s", BULK_OPTICS_NC)

    # ---- Monodisperse (single radius, for comparison) ----------------------
    sd_mono = SizeDistribution.lognormal(rg_nm=R_EFF_MONODISPERSE_NM, sigma_ln=1e-6)  # near-delta
    mono = _build_bulk([R_EFF_MONODISPERSE_NM], MATERIAL, sd_mono, N_LEGENDRE)
    _log_bulk_summary(mono, "monodisperse")

    mono.to_netcdf(str(MONODISPERSE_OPTICS_NC))
    logger.info("Saved monodisperse optics -> %s", MONODISPERSE_OPTICS_NC)

    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
