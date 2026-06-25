"""Stage 2: Run radiative transfer with pyRadtran using bulk aerosol optics.

Loads a BulkAerosolOpticsData NetCDF, wraps it in a pyRadtran ``BulkSpecies``,
builds an exponential mass profile matching a target optical depth at 550 nm,
and runs DISORT via the ``Scene`` + ``Runner.execute`` API.

Supports a ``--tag`` argument so the same script drives both the bulk and
monodisperse runs (``rt_bulk.nc`` / ``rt_monodisperse.nc``).

Usage:
    python run_radiative_transfer.py --tag bulk
    python run_radiative_transfer.py --tag monodisperse
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from Aerosol3D.bulk import BulkAerosolOpticsData

from pyradtran import Scene, Runner
from pyradtran.models.aerosol_composite import BulkSpecies, CompositeAerosol
from pyradtran.models.blocks import MassProfile, PlacedBlock

from config import (
    ALTITUDE_GRID_KM,
    BULK_OPTICS_NC,
    LIBRADTRAN_DATA,
    MONODISPERSE_OPTICS_NC,
    N_LEGENDRE,
    RT_BULK_NC,
    RT_MONODISPERSE_NC,
    SCENE_CONFIG,
    SCALE_HEIGHT_KM,
    TOTAL_OPTICAL_DEPTH_550,
    WAVELENGTHS_NM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_mass_profile(
    bulk: BulkAerosolOpticsData,
    optical_depth_550: float,
    altitude_grid_km,
    scale_height_km: float,
) -> np.ndarray:
    """Invert a target OD@550 into a per-layer mass concentration profile.

    The bulk OD at 550 nm is

        tau_550 = sum_layers  beta_ext_per_mass_550 * rho_layer * dz

    where ``beta_ext_per_mass_550 = C_ext_550 / mass_per_particle``. Using the
    effective (volume-weighted) density from Task 1/4 and the size-distribution
    third moment, we can solve for the surface mass concentration ``rho_0``
    that produces the requested column OD under an exponential profile.

    The returned profile is expressed on the pyRadtran layer grid
    (length ``len(altitude_grid_km) - 1``), indexed from the TOP layer
    DOWNWARDS to match the descending-altitude layer order used by
    ``MassProfile`` / ``CompositeAerosol``.
    """
    idx_550 = int(np.argmin(np.abs(bulk.wavelength_nm - 550.0)))
    C_ext_550_nm2 = float(bulk.C_ext[idx_550])

    rho = float(bulk.effective_density_kg_m3)
    if rho is None:
        raise ValueError("BulkAerosolOpticsData.effective_density_kg_m3 is required")
    if bulk.size_distribution is None:
        raise ValueError("BulkAerosolOpticsData.size_distribution is required")

    # Third moment of the size distribution = E[r^3] (nm^3 per particle).
    r3_nm3 = float(bulk.size_distribution.moment(3.0))
    vol_m3 = (4.0 / 3.0) * np.pi * r3_nm3 * 1e-27  # nm^3 -> m^3
    mass_per_particle_kg = rho * vol_m3

    # Mass-normalized extinction at 550 nm (m^2 / kg)
    C_ext_550_m2 = C_ext_550_nm2 * 1e-18  # nm^2 -> m^2
    beta_ext_per_mass_550 = C_ext_550_m2 / mass_per_particle_kg

    # Exponential profile: rho(z) = rho_0 * exp(-z/H)
    # Column OD = integral_0^infty beta_ext * rho_0 * exp(-z/H) dz
    #           = beta_ext * rho_0 * H  (H in meters)
    H_m = scale_height_km * 1000.0
    rho_0_kg_m3 = optical_depth_550 / (beta_ext_per_mass_550 * H_m)

    # Layer centers (descending) — match MassProfile layer order
    alt = np.asarray(altitude_grid_km, dtype=float)
    alt_centers = 0.5 * (alt[:-1] + alt[1:])  # ascending layer centers
    # exp(-z/H) at each layer center
    rho_profile = rho_0_kg_m3 * np.exp(-alt_centers / scale_height_km)
    return rho_profile


def build_composite_aerosol(bulk: BulkAerosolOpticsData, output_dir: Path) -> CompositeAerosol:
    """Wrap a BulkAerosolOpticsData into a pyRadtran CompositeAerosol."""
    species = BulkSpecies(bulk=bulk)

    mass_profile = compute_mass_profile(
        bulk=bulk,
        optical_depth_550=TOTAL_OPTICAL_DEPTH_550,
        altitude_grid_km=ALTITUDE_GRID_KM,
        scale_height_km=SCALE_HEIGHT_KM,
    )

    placed = PlacedBlock(
        block=species,
        profile=MassProfile(kg_m3_per_layer=tuple(mass_profile.tolist())),
    )

    wavelengths_um = (np.asarray(WAVELENGTHS_NM, dtype=float) / 1000.0).tolist()

    aerosol = CompositeAerosol(
        pieces=[placed],
        wavelength_grid_um=wavelengths_um,
        altitude_grid_km=list(ALTITUDE_GRID_KM),
        n_legendre=N_LEGENDRE,
        output_dir=output_dir,
    )
    return aerosol


def build_scene(aerosol: CompositeAerosol) -> Scene:
    """Build the pyRadtran Scene (chainable builder)."""
    cfg = SCENE_CONFIG
    scene = (
        Scene()
        .set_atmosphere(
            profile=cfg["atmosphere"]["profile"],
            altitude=cfg["atmosphere"]["altitude"],
        )
        .set_source_solar(sza=cfg["source"]["sza"])
        .set_wavelength(
            cfg["wavelength"]["min_nm"],
            cfg["wavelength"]["max_nm"],
        )
        .set_solver(
            method=cfg["solver"]["method"],
            streams=cfg["solver"]["streams"],
            disort_intcor=cfg["solver"].get("disort_intcor"),
            pseudospherical=cfg["solver"].get("pseudospherical", False),
        )
        .set_surface(albedo=cfg["surface"]["albedo"])
        .set_output(
            quantities=cfg["output"]["quantities"],
            quantity=cfg["output"]["quantity"],
            format=cfg["output"]["format"],
        )
        .set_aerosol(aerosol)
    )
    return scene


def run_radiative_transfer(optics_path: str, output_path: str, tag: str) -> None:
    """Load bulk optics, build scene, execute pyRadtran, save NetCDF."""
    logger.info("Running radiative transfer [%s] for %s", tag, optics_path)

    bulk = BulkAerosolOpticsData.from_netcdf(optics_path)
    logger.info(
        "Loaded bulk: %d wavelengths, r_eff=%.2f nm, rho_eff=%.2f kg/m^3",
        bulk.wavelength_nm.size,
        float(bulk.r_eff_nm or 0.0),
        float(bulk.effective_density_kg_m3 or 0.0),
    )

    output_dir = Path(output_path).parent
    aerosol = build_composite_aerosol(bulk, output_dir=output_dir)
    scene = build_scene(aerosol)

    data_path = os.environ.get("PYRADTRAN_DATA_PATH", LIBRADTRAN_DATA)
    if not os.path.isdir(data_path):
        logger.warning("libRadtran data path not found: %s", data_path)
        logger.warning("Set PYRADTRAN_DATA_PATH to your libRadtran data/ dir.")

    logger.info("Executing pyRadtran DISORT...")
    result = Runner.execute(scene, data_path=data_path)
    result.to_netcdf(output_path)
    logger.info("Saved RT result -> %s", output_path)
    logger.info("Result dimensions: %s", dict(result.sizes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        choices=["bulk", "monodisperse"],
        default="bulk",
        help="Which optics source to use (default: bulk)",
    )
    args = parser.parse_args()

    if args.tag == "bulk":
        optics_path = str(BULK_OPTICS_NC)
        output_path = str(RT_BULK_NC)
    else:
        optics_path = str(MONODISPERSE_OPTICS_NC)
        output_path = str(RT_MONODISPERSE_NC)

    if not Path(optics_path).exists():
        logger.error("Optics file not found: %s", optics_path)
        logger.error("Run compute_bulk_optics.py first.")
        raise SystemExit(1)

    run_radiative_transfer(optics_path, output_path, args.tag)
    logger.info("Stage 2 [%s] complete.", args.tag)


if __name__ == "__main__":
    main()
