"""Stage 2: comprehensive multi-component RT + every pyradtran.viz plot + workflow.

This is the LEGO capstone: three aerosol blocks (black_carbon + sulfate +
mineral_dust) are externally mixed into one ``CompositeAerosol`` and run through
DISORT. The script then exercises the FULL ``pyradtran.viz`` plot surface and the
component-attribution workflow:

  Composite diagnostics (analytic, no RT):
    - evaluate_composite_on_grid -> plot_composite_optics (tau / ssa / g)
    - evaluate_blocks_on_grid    -> plot_block_profiles (per-block tau(z), rho(z))

  RT result plots (full composite, real DISORT):
    - plot_spectral, plot_flux_profile, plot_budget (via add_budget_vars),
      plot_heating_rate (if libRadtran emits it), plot_rt_overview

  Workflow (component attribution, leave-one-out):
    - compute_component_attribution -> plot_component_attribution

Requires libRadtran (set PYRADTRAN_DATA_PATH or rely on LIBRADTRAN_DATA below).
Run ``compute_bulk_optics.py`` first.

Usage:
    python run_comprehensive.py
"""

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Aerosol3D.bulk import BulkAerosolOpticsData

from pyradtran import Runner, Scene
from pyradtran.core.output_parser import HEATING_RATE_COLUMN
from pyradtran.core.postprocess import (
    add_budget_vars,
    evaluate_blocks_on_grid,
    evaluate_composite_on_grid,
)
from pyradtran.models.aerosol_composite import BulkSpecies, CompositeAerosol
from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
from pyradtran.viz import (
    plot_block_profiles,
    plot_budget,
    plot_component_attribution,
    plot_composite_optics,
    plot_flux_profile,
    plot_heating_rate,
    plot_rt_overview,
    plot_spectral,
    save,
    set_theme,
)
from pyradtran.workflow import compute_component_attribution

from config import ALTITUDE_GRID_KM, BLOCKS, LIBRADTRAN_DATA, N_LEGENDRE, OUTPUT_DIR, REF_NM, SCENE_CONFIG, WAVELENGTHS_NM, bulk_nc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_FLUX_VARS = {"edir", "edn", "eup", "udir", "udn", "uup"}


def _save(fig, name):
    path = OUTPUT_DIR / name
    save(fig, str(path), formats=("png",))
    plt.close(fig)
    logger.info("  saved %s", path.name)


def build_composite() -> CompositeAerosol:
    """Load per-block bulk optics, invert each block's target OD@550 into a mass
    profile via the API (od_to_mass_profile), and assemble the composite."""
    pieces = []
    for b in BLOCKS:
        bulk = BulkAerosolOpticsData.from_netcdf(bulk_nc(b["name"]))
        species = BulkSpecies(bulk=bulk, name=b["name"])
        profile = od_to_mass_profile(
            species,
            tau_ref=b["tau_550"],
            ref_nm=REF_NM,
            altitude_km=ALTITUDE_GRID_KM,
            scale_height_km=b["scale_height_km"],
        )
        pieces.append(PlacedBlock(block=species, profile=profile))
        logger.info(
            "  block %-14s tau@550=%.2f H=%.1f km", b["name"], b["tau_550"], b["scale_height_km"]
        )
    wl_um = (np.asarray(WAVELENGTHS_NM, dtype=float) / 1000.0).tolist()
    return CompositeAerosol(
        pieces=pieces,
        wavelength_grid_um=wl_um,
        altitude_grid_km=list(ALTITUDE_GRID_KM),
        n_legendre=N_LEGENDRE,
        output_dir=OUTPUT_DIR,
    )


def build_scene(aerosol: CompositeAerosol) -> Scene:
    cfg = SCENE_CONFIG
    return (
        Scene()
        .set_atmosphere(profile=cfg["atmosphere"]["profile"], altitude=cfg["atmosphere"]["altitude"])
        .set_source_solar(sza=cfg["source"]["sza"])
        .set_wavelength(cfg["wavelength"]["min_nm"], cfg["wavelength"]["max_nm"])
        .set_solver(
            method=cfg["solver"]["method"],
            streams=cfg["solver"]["streams"],
            disort_intcor=cfg["solver"].get("disort_intcor"),
            pseudospherical=cfg["solver"].get("pseudospherical", False),
        )
        .set_surface(albedo=cfg["surface"]["albedo"])
        .set_output(**cfg["output"])
        .set_aerosol(aerosol)
    )


def _resolve_heating_var(ds):
    """Return the heating-rate variable name if present, else None.

    libRadtran's heating-rate column naming depends on solver/output settings
    (open item O1): it may already be ``heating_rate`` or appear as an extra
    unnamed column. Normalize to HEATING_RATE_COLUMN when found.
    """
    if HEATING_RATE_COLUMN in ds.data_vars:
        return HEATING_RATE_COLUMN
    extras = [v for v in ds.data_vars if v not in _FLUX_VARS and v != "wavelength"]
    if len(extras) == 1:
        return extras[0]
    return None


def main():
    set_theme("publication")
    data_path = os.environ.get("PYRADTRAN_DATA_PATH", LIBRADTRAN_DATA)
    if not Path(data_path).is_dir():
        logger.warning("libRadtran data dir not found: %s", data_path)

    logger.info("=== Building 3-block composite ===")
    composite = build_composite()

    # ------------------------------------------------------------------
    # Composite diagnostics (analytic mixing, no RT)
    # ------------------------------------------------------------------
    logger.info("=== Composite diagnostics (analytic) ===")
    wl_um = (np.asarray(WAVELENGTHS_NM, dtype=float) / 1000.0).tolist()
    grid_ds = evaluate_composite_on_grid(composite, wl_um, ALTITUDE_GRID_KM, n_legendre=N_LEGENDRE)
    for q in ("tau", "ssa", "g"):
        fig, _ = plot_composite_optics(grid_ds, quantity=q)
        _save(fig, f"composite_{q}.png")

    block_dict = evaluate_blocks_on_grid(composite, wl_um, ALTITUDE_GRID_KM, n_legendre=N_LEGENDRE)
    fig, _ = plot_block_profiles(block_dict, quantity="tau")
    _save(fig, "block_tau_profiles.png")
    fig, _ = plot_block_profiles(block_dict, quantity="rho")
    _save(fig, "block_rho_profiles.png")

    # ------------------------------------------------------------------
    # RT result plots (full composite, real DISORT)
    # ------------------------------------------------------------------
    logger.info("=== RT run (full composite, DISORT) ===")
    rt = Runner.execute(build_scene(composite), data_path=data_path)
    logger.info("  RT data_vars=%s dims=%s", list(rt.data_vars), dict(rt.sizes))

    fig, _ = plot_spectral(rt)
    _save(fig, "rt_spectral.png")
    fig, _ = plot_flux_profile(rt, variable="edir", wavelength_nm=550.0)
    _save(fig, "rt_flux_profile_edir.png")

    rt_budget = add_budget_vars(rt)
    rt_budget.to_netcdf(str(OUTPUT_DIR / "rt_full_budget.nc"))
    fig, _ = plot_budget(rt_budget)
    _save(fig, "rt_budget.png")

    hvar = _resolve_heating_var(rt)
    if hvar is not None:
        if hvar != HEATING_RATE_COLUMN:
            rt = rt.rename({hvar: HEATING_RATE_COLUMN})
            logger.info("  renamed heating column '%s' -> '%s'", hvar, HEATING_RATE_COLUMN)
        fig, _ = plot_heating_rate(rt, wavelength_nm=550.0)
        _save(fig, "rt_heating_rate.png")
    else:
        logger.warning("  no heating-rate column in RT output; skipping plot_heating_rate")

    rt.to_netcdf(str(OUTPUT_DIR / "rt_full.nc"))
    fig, _ = plot_rt_overview(rt, wavelength_nm=550.0)
    _save(fig, "rt_overview.png")

    # ------------------------------------------------------------------
    # Component attribution (leave-one-out, parallel RT)
    # ------------------------------------------------------------------
    logger.info("=== Component attribution (N+1 DISORT runs) ===")

    # Run the leave-one-out scenes sequentially via Runner.execute. The workflow's
    # execute_many is an injected callable, so we choose the strategy here:
    # Runner.execute_many uses a ProcessPoolExecutor that (a) swallows per-scene
    # exceptions and (b) fails on these CompositeAerosol scenes (pickling across
    # processes). Sequential execution surfaces real errors and is fine for 4 runs.
    def execute_many(scenes):
        return [Runner.execute(s, data_path=data_path) for s in scenes]

    result = compute_component_attribution(build_scene, composite, execute_many)
    logger.info("  contributions: %s", list(result.contributions))
    fig, _ = plot_component_attribution(result, variable="edir", level="surface")
    _save(fig, "attribution_edir.png")

    logger.info("=== Comprehensive demo complete. Figures in %s ===", OUTPUT_DIR)


if __name__ == "__main__":
    main()
