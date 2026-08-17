"""Stage 1: Build per-block bulk aerosol optics for the multi-component demo.

For each block in ``BLOCKS`` (black_carbon, sulfate, mineral_dust), build a
``BulkAerosolOpticsData`` from a lognormal size distribution via per-radius Mie
solves, and save it to ``output/bulk_<name>.nc``.

Usage:
    python compute_bulk_optics.py
"""

import logging

import numpy as np
from config import BLOCKS, N_LEGENDRE, WAVELENGTHS_NM, bulk_nc

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_particle(radius_nm: float, material_name: str) -> AerosolParticle:
    material = preset_material(material_name)
    particle = AerosolParticle(
        name=f"{material_name}_r{int(radius_nm)}nm",
        mixing_state=MixingState.INTERNAL,
        unit="nm",
    )
    particle.add_mesh("core", create_sphere((0, 0, 0), radius_nm), material)
    return particle


def _solve_wavelengths(particle, wavelengths_nm) -> list:
    results = []
    for wl in wavelengths_nm:
        config = SimulationConfig(wavelength=wl, source="solar", precision="medium")
        result = solve_optics(
            particle, config, solver="MIE", voxel_size=None, compute_phase_func=True
        )
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
    return results


def _build_bulk(radii_nm, material_name: str, sd: SizeDistribution, n_legendre: int):
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
    return builder.compute()


def main():
    logger.info("Stage 1: building per-block bulk optics for %d blocks", len(BLOCKS))
    for b in BLOCKS:
        logger.info(
            "  [%s] material=%s lognormal(rg=%.1f nm, sigma_ln=%.2f), tau@550=%.2f",
            b["name"],
            b["material"],
            b["rg_nm"],
            b["sigma_ln"],
            b["tau_550"],
        )
        sd = SizeDistribution.lognormal(rg_nm=b["rg_nm"], sigma_ln=b["sigma_ln"])
        bulk = _build_bulk(b["radii_nm"], b["material"], sd, N_LEGENDRE)
        bulk.to_netcdf(str(bulk_nc(b["name"])))
        idx_550 = int(np.argmin(np.abs(bulk.wavelength_nm - 550.0)))
        logger.info(
            "    -> %d wl, r_eff=%.1f nm, SSA@550=%.3f, g@550=%.3f; saved %s",
            bulk.wavelength_nm.size,
            float(bulk.r_eff_nm),
            float(bulk.SSA[idx_550]),
            float(bulk.g[idx_550]),
            bulk_nc(b["name"]).name,
        )
    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
