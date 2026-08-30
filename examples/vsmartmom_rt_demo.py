#!/usr/bin/env python3
"""vSmartMOM Column RT Demo — Run radiative transfer with Aerosol3D bulk optics.

This example demonstrates the complete workflow from synthetic bulk aerosol
optical properties to top-of-atmosphere (TOA) reflectance via vSmartMOM:

    1. Create synthetic BulkAerosolOpticsData (multi-wavelength)
    2. Define a vertical number-concentration profile
    3. Run radiative transfer with VSmartMOMRunner
    4. Visualize TOA reflectance and BOA transmittance

Requirements:
    - Aerosol3D installed (pip install -e .)
    - Julia with vSmartMOM.jl installed (for actual RT computation)

    If Julia/vSmartMOM is not available, the script falls back to showing
    the input setup and expected workflow.

Usage:
    python examples/vsmartmom_rt_demo.py [--julia-project PATH] [--output-dir DIR]

Output:
    - Console: layer optical depths, total column τ
    - Figures: toa_reflectance.png, boa_transmittance.png, tau_profile.png
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# Ensure src is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData
from Aerosol3D.vsmartmom import VSmartMOMResult, VSmartMOMRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Wavelengths (nm) — 3 channels: blue, green, red
WAVELENGTHS_NM = np.array([470.0, 550.0, 670.0])
N_WL = len(WAVELENGTHS_NM)
N_LEGENDRE = 16

# Vertical profile: 4 layers (boundary layer, lower free trop, upper free trop, tropopause)
# Heights are layer *boundaries* in meters
HEIGHTS_M = np.array([0.0, 1000.0, 3000.0, 6000.0, 12000.0])
# Number concentrations per layer (cm^-3)
NUMBER_CONC_CM3 = np.array([500.0, 150.0, 30.0, 5.0])

# Viewing geometry
SZA_DEG = 30.0
VZA_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
VAZ_DEG = np.zeros_like(VZA_DEG)


def _make_synthetic_bulk(wavelengths: np.ndarray, n_legendre: int) -> BulkAerosolOpticsData:
    """Create synthetic bulk aerosol optical properties.

    Uses a simplified Mie-like model where:
    - C_ext ~ wavelength^-1.3 (Angström-like)
    - SSA increases with wavelength (less absorption at longer λ)
    - g decreases with wavelength (less forward scattering at longer λ)
    """
    n_wl = len(wavelengths)

    # Extinction cross-section per particle (nm²)
    # Reference: 1000 nm² at 550 nm, scaling as λ^-1.3
    C_ext = 1000.0 * (wavelengths / 550.0) ** (-1.3)

    # SSA: 0.75 at 470nm → 0.88 at 550nm → 0.92 at 670nm
    SSA = 0.6 + 0.35 * (wavelengths / 400.0) ** 0.5
    SSA = np.clip(SSA, 0.5, 0.98)

    C_sca = C_ext * SSA
    C_abs = C_ext - C_sca

    # Asymmetry parameter g: 0.72 → 0.65 → 0.58
    g = 0.72 - 0.14 * (wavelengths - 470.0) / 200.0
    g = np.clip(g, 0.3, 0.85)

    # Legendre coefficients in vSmartMOM convention: beta_l = (2l+1) * g^l
    # beta[..., 0] == 1 by construction
    beta = np.zeros((n_wl, n_legendre))
    for l in range(n_legendre):
        beta[:, l] = (2 * l + 1) * g**l

    return BulkAerosolOpticsData(
        wavelength_nm=wavelengths.copy(),
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        SSA=SSA,
        g=g,
        beta=beta,
        n_legendre=n_legendre,
        r_eff_nm=150.0,
        size_distribution=None,
        theta_rad=None,
        phi_rad=None,
        P11=None,
        radii_nm=None,
        radii_weights=None,
        per_radius_C_ext=None,
        per_radius_C_sca=None,
        per_radius_beta=None,
        interpolation_method="",
        integration_method="",
        integration_n_points=0,
        fallback_used=False,
        fallback_wavelengths=None,
        tau_ref=None,
        concentration_method=None,
        concentration_kwargs=None,
    )


def _print_setup(bulk: BulkAerosolOpticsData) -> None:
    """Print the input setup summary."""
    print("\n" + "=" * 70)
    print("  vSmartMOM Column RT Demo — Input Setup")
    print("=" * 70)

    print("\n  Bulk Aerosol Optics:")
    print(f"    {'λ (nm)':>10} | {'C_ext':>12} | {'C_sca':>12} | {'SSA':>8} | {'g':>8}")
    print(f"    {'-' * 60}")
    for i, wl in enumerate(bulk.wavelength_nm):
        print(
            f"    {wl:>10.0f} | {bulk.C_ext[i]:>12.1f} | "
            f"{bulk.C_sca[i]:>12.1f} | {bulk.SSA[i]:>8.4f} | {bulk.g[i]:>8.4f}"
        )

    print("\n  Vertical Profile:")
    print(
        f"    {'Layer':>8} | {'z_bottom (m)':>14} | {'z_top (m)':>12} | {'dz (m)':>10} | {'N (cm⁻³)':>10}"
    )
    print(f"    {'-' * 65}")
    dz = np.diff(HEIGHTS_M)
    for i in range(len(NUMBER_CONC_CM3)):
        print(
            f"    {i + 1:>8} | {HEIGHTS_M[i]:>14.0f} | "
            f"{HEIGHTS_M[i + 1]:>12.0f} | {dz[i]:>10.0f} | {NUMBER_CONC_CM3[i]:>10.1f}"
        )

    # Compute layer optical depths
    tau_per_layer = (
        NUMBER_CONC_CM3[np.newaxis, :] * bulk.C_ext[:, np.newaxis] * dz[np.newaxis, :] * 1e-6
    )
    print("\n  Layer Optical Depths:")
    print(
        f"    {'λ (nm)':>10} | {'τ_layer 1':>12} | {'τ_layer 2':>12} | {'τ_layer 3':>12} | {'τ_total':>12}"
    )
    print(f"    {'-' * 65}")
    for i, wl in enumerate(bulk.wavelength_nm):
        print(
            f"    {wl:>10.0f} | {tau_per_layer[i, 0]:>12.4f} | "
            f"{tau_per_layer[i, 1]:>12.4f} | {tau_per_layer[i, 2]:>12.4f} | "
            f"{tau_per_layer[i].sum():>12.4f}"
        )

    print("\n  Geometry:")
    print(f"    SZA = {SZA_DEG}°")
    print(f"    VZA = {VZA_DEG}°")
    print(f"    VAZ = {VAZ_DEG}°")
    print("=" * 70)


def _plot_results(result: VSmartMOMResult, output_dir: Path) -> None:
    """Plot TOA reflectance, BOA transmittance, and τ profile."""
    import matplotlib.pyplot as plt

    wl = result.wavelengths
    wl_colors = {470.0: "#3498db", 550.0: "#2ecc71", 670.0: "#e74c3c"}

    # ------------------------------------------------------------------
    # Figure 1: TOA Reflectance vs Viewing Angle
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for j, w in enumerate(wl):
        color = wl_colors.get(float(w), "#333333")
        # R shape: [n_stokes, n_vza, n_wl]; take Stokes I (index 0)
        ax1.plot(
            result.vza,
            result.R[0, :, j],
            "o-",
            color=color,
            label=f"{w:.0f} nm",
            markersize=8,
            lw=2,
        )

    ax1.set_xlabel("Viewing Zenith Angle (°)", fontsize=12)
    ax1.set_ylabel("TOA Reflectance R", fontsize=12)
    ax1.set_title("Top-of-Atmosphere Reflectance", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "toa_reflectance.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'toa_reflectance.png'}")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: BOA Transmittance vs Viewing Angle
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for j, w in enumerate(wl):
        color = wl_colors.get(float(w), "#333333")
        ax2.plot(
            result.vza,
            result.T[0, :, j],
            "s-",
            color=color,
            label=f"{w:.0f} nm",
            markersize=8,
            lw=2,
        )

    ax2.set_xlabel("Viewing Zenith Angle (°)", fontsize=12)
    ax2.set_ylabel("BOA Transmittance T", fontsize=12)
    ax2.set_title("Bottom-of-Atmosphere Transmittance", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "boa_transmittance.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'boa_transmittance.png'}")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3: Optical Depth Profile
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    layer_centers = (HEIGHTS_M[:-1] + HEIGHTS_M[1:]) / 2.0
    for j, w in enumerate(wl):
        color = wl_colors.get(float(w), "#333333")
        ax3.barh(
            layer_centers,
            result.tau_per_layer[j, :],
            height=np.diff(HEIGHTS_M) * 0.7,
            color=color,
            alpha=0.6,
            label=f"{w:.0f} nm",
        )

    ax3.set_xlabel("Optical Depth τ", fontsize=12)
    ax3.set_ylabel("Altitude (m)", fontsize=12)
    ax3.set_title("Layer Optical Depth Profile", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.invert_yaxis()
    fig3.tight_layout()
    fig3.savefig(output_dir / "tau_profile.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'tau_profile.png'}")
    plt.close(fig3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--julia-project",
        type=str,
        default=None,
        help="Path to Julia project with vSmartMOM installed. If omitted, uses the Julia default environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "output"),
        help="Directory for output figures (default: examples/output/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Create synthetic bulk aerosol optics
    # ------------------------------------------------------------------
    logger.info("Creating synthetic bulk aerosol optics...")
    bulk = _make_synthetic_bulk(WAVELENGTHS_NM, N_LEGENDRE)

    # ------------------------------------------------------------------
    # Step 2: Print setup
    # ------------------------------------------------------------------
    _print_setup(bulk)

    # ------------------------------------------------------------------
    # Step 3: Check Julia availability
    # ------------------------------------------------------------------
    if not shutil.which("julia"):
        print("\n" + "!" * 70)
        print("  Julia not found on PATH.")
        print("  To run this example, install Julia and vSmartMOM.jl:")
        print()
        print("    1. Download Julia from https://julialang.org/downloads/")
        print("    2. Install vSmartMOM in a Julia project:")
        print("       julia> using Pkg")
        print('       julia> Pkg.activate("path/to/project")')
        print('       julia> Pkg.add("vSmartMOM")')
        print()
        print("    3. Run this example with --julia-project:")
        print("       python examples/vsmartmom_rt_demo.py --julia-project /path/to/project")
        print("!" * 70)
        print("\n  The input data above is ready for RT computation.")
        return 0

    julia_project: Path | None = None
    if args.julia_project:
        julia_project = Path(args.julia_project)
        if not julia_project.exists():
            logger.error(f"Julia project path does not exist: {julia_project}")
            return 1

    # ------------------------------------------------------------------
    # Step 4: Run radiative transfer
    # ------------------------------------------------------------------
    if julia_project is not None:
        logger.info(f"Running vSmartMOM RT (Julia project: {julia_project})...")
    else:
        logger.info("Running vSmartMOM RT (Julia default environment)...")
    runner = VSmartMOMRunner(
        julia_project=julia_project,
        cleanup_temp=True,
    )

    try:
        result = runner.run_rt(
            bulk=bulk,
            heights=HEIGHTS_M,
            number_conc=NUMBER_CONC_CM3,
            sza=SZA_DEG,
            vza=VZA_DEG,
            vaz=VAZ_DEG,
        )
    except RuntimeError as e:
        logger.error(f"Radiative transfer failed: {e}")
        return 1

    # ------------------------------------------------------------------
    # Step 5: Print results summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Radiative Transfer Results")
    print("=" * 70)
    print(f"\n  R shape: {result.R.shape}  (stokes, vza, wavelength)")
    print(f"  T shape: {result.T.shape}  (stokes, vza, wavelength)")
    print("\n  TOA Reflectance (Stokes I):")
    print(f"    {'λ (nm)':>10} | {'VZA=0°':>10} | {'VZA=30°':>10} | {'VZA=60°':>10}")
    print(f"    {'-' * 45}")
    for j, wl in enumerate(result.wavelengths):
        # Find indices closest to 0°, 30°, 60°
        i0 = np.argmin(np.abs(result.vza - 0.0))
        i30 = np.argmin(np.abs(result.vza - 30.0))
        i60 = np.argmin(np.abs(result.vza - 60.0))
        print(
            f"    {wl:>10.0f} | {result.R[0, i0, j]:>10.6f} | "
            f"{result.R[0, i30, j]:>10.6f} | {result.R[0, i60, j]:>10.6f}"
        )

    print("\n  BOA Transmittance (Stokes I):")
    print(f"    {'λ (nm)':>10} | {'VZA=0°':>10} | {'VZA=30°':>10} | {'VZA=60°':>10}")
    print(f"    {'-' * 45}")
    for j, wl in enumerate(result.wavelengths):
        i0 = np.argmin(np.abs(result.vza - 0.0))
        i30 = np.argmin(np.abs(result.vza - 30.0))
        i60 = np.argmin(np.abs(result.vza - 60.0))
        print(
            f"    {wl:>10.0f} | {result.T[0, i0, j]:>10.6f} | "
            f"{result.T[0, i30, j]:>10.6f} | {result.T[0, i60, j]:>10.6f}"
        )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 6: Visualize
    # ------------------------------------------------------------------
    logger.info("Generating figures...")
    try:
        _plot_results(result, output_dir)
    except ImportError:
        logger.warning("matplotlib not available, skipping figure generation")

    # ------------------------------------------------------------------
    # Step 7: Save result to NetCDF
    # ------------------------------------------------------------------
    result_path = output_dir / "vsmartmom_result.nc"
    result.to_netcdf(str(result_path))
    logger.info(f"Result saved to: {result_path}")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
