"""Cross-validation of the asymmetry parameter g via dual-path consistency check.

Path A (phase-function quadrature):
    For each radius, compute g(r) from P11(theta) via angular quadrature:
        g(r) = integral(P11 * cos(theta) * sin(theta) dtheta)
               / integral(P11 * sin(theta) dtheta)
    Then energy-weighted integration over the size distribution.

Path B (Legendre moment):
    Extract beta_1 from bulk data (already computed by BulkOpticsBuilder).
        g = beta_1 / 3.0   (vSmartMOM convention)

Validation: |g_A - g_B| < tol (default tol = 1e-5)

Usage:
    python examples/bulk_cross_validation.py \
        --bulk-input output/bulk_aerosol.nc \
        --singles-input output/mie_r*.nc \
        --tol 1e-5
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from Aerosol3D.bulk.datastructs import SizeDistribution
from Aerosol3D.bulk.integration import integrate_distribution_vectorized
from Aerosol3D.bulk.interpolation import LinearPCHIPInterpolator
from Aerosol3D.bulk.io import bulk_from_netcdf
from Aerosol3D.optics.optics_export import AerosolOpticsData


def compute_g_from_phase_function(theta_rad: np.ndarray, P11: np.ndarray) -> float:  # noqa: N803
    """Compute asymmetry parameter g from azimuthally-averaged P11.

    Args:
        theta_rad: Scattering angles in radians, shape (n_theta,).
        P11: Phase function values, shape (n_theta,).

    Returns:
        Asymmetry parameter g.
    """
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    numerator = _trapz(P11 * cos_theta * sin_theta, theta_rad)
    denominator = _trapz(P11 * sin_theta, theta_rad)

    if abs(denominator) < 1e-30:
        return float("nan")

    return float(numerator / denominator)


def load_single_size_data(paths: list[str]) -> list[AerosolOpticsData]:
    """Load single-size AerosolOpticsData from NetCDF files.

    Args:
        paths: List of NetCDF file paths.

    Returns:
        List of AerosolOpticsData instances, sorted by r_eff_nm.
    """
    results = []
    for p in paths:
        data = AerosolOpticsData.from_netcdf(p)
        results.append(data)

    # Sort by effective radius
    results.sort(key=lambda d: d.r_eff_nm)
    return results


def extract_radius_from_path(path: str) -> float | None:
    """Attempt to extract radius (nm) from a filename like 'mie_r050.nc'.

    Returns None if no numeric pattern is found.
    """
    import re

    m = re.search(r"r(\d+)", Path(path).stem)
    if m:
        return float(m.group(1))
    return None


def compute_g_path_a(
    single_data: list[AerosolOpticsData],
    size_distribution: SizeDistribution,
    wavelength_nm: float,
) -> float:
    """Compute g via Path A: phase-function quadrature + size distribution.

    For each single-size dataset, compute g(r) from P11(theta) at the
    requested wavelength, then energy-weight integrate over the size
    distribution using PCHIP interpolation.

    Args:
        single_data: List of single-size AerosolOpticsData, sorted by radius.
        size_distribution: Size distribution for energy-weighted integration.
        wavelength_nm: Target wavelength in nm.

    Returns:
        Bulk asymmetry parameter g_A.
    """
    radii = np.array([d.r_eff_nm for d in single_data])
    g_per_radius = []
    C_sca_per_radius = []

    for data in single_data:
        # Find closest wavelength index
        wl_idx = int(np.argmin(np.abs(data.wavelength_nm - wavelength_nm)))

        if data.P11 is None or data.theta_rad is None:
            raise ValueError(
                f"Single-size data for r={data.r_eff_nm:.1f} nm "
                "missing phase-function data (P11/theta)."
            )

        # Azimuthal average: mean over phi axis
        # P11 shape: (n_wavelength, n_theta, n_phi)
        P11_wl = data.P11[wl_idx]  # (n_theta, n_phi)
        P11_avg = np.mean(P11_wl, axis=1)  # (n_theta,)

        g_r = compute_g_from_phase_function(data.theta_rad, P11_avg)
        g_per_radius.append(g_r)
        C_sca_per_radius.append(data.C_sca[wl_idx])

    g_per_radius = np.array(g_per_radius)
    C_sca_per_radius = np.array(C_sca_per_radius)

    # Build PCHIP interpolators for g(r) and C_sca(r)
    g_interp = LinearPCHIPInterpolator(radii, g_per_radius)
    C_sca_interp = LinearPCHIPInterpolator(radii, C_sca_per_radius)

    # Energy-weighted integrand: g(r) * C_sca(r) * n(r)
    def integrand(r: np.ndarray) -> np.ndarray:
        return g_interp(r) * C_sca_interp(r)

    numerator = integrate_distribution_vectorized(integrand, size_distribution, n_quad=256)

    def denominator_integrand(r: np.ndarray) -> np.ndarray:
        return C_sca_interp(r)

    denominator = integrate_distribution_vectorized(
        denominator_integrand, size_distribution, n_quad=256
    )

    if abs(denominator) < 1e-30:
        return float("nan")

    return float(numerator / denominator)


def compute_g_path_b(bulk: object, wavelength_nm: float) -> float:
    """Compute g via Path B: Legendre moment beta_1 from bulk data.

    Args:
        bulk: BulkAerosolOpticsData instance.
        wavelength_nm: Target wavelength in nm.

    Returns:
        Asymmetry parameter g_B = beta_1 / 3.0.
    """
    wl_idx = int(np.argmin(np.abs(bulk.wavelength_nm - wavelength_nm)))
    beta_1 = bulk.beta[wl_idx, 1]
    return float(beta_1 / 3.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-validate asymmetry parameter g via dual-path consistency."
    )
    parser.add_argument(
        "--bulk-input",
        required=True,
        help="Path to bulk aerosol NetCDF file (from BulkOpticsBuilder).",
    )
    parser.add_argument(
        "--singles-input",
        required=True,
        nargs="+",
        help="One or more single-size NetCDF files (glob patterns supported).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance for |g_A - g_B| consistency check (default: 1e-5).",
    )
    return parser.parse_args()


def resolve_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns into a flat list of file paths."""
    paths: list[str] = []
    for pat in patterns:
        matched = glob.glob(pat)
        if matched:
            paths.extend(sorted(matched))
        elif Path(pat).exists():
            paths.append(pat)
        else:
            print(f"Warning: pattern '{pat}' matched no files.", file=sys.stderr)
    return paths


def main() -> int:
    """Run the cross-validation and return exit code."""
    args = parse_args()

    # Load bulk data
    print(f"Loading bulk data: {args.bulk_input}")
    bulk = bulk_from_netcdf(args.bulk_input)
    print(f"  Wavelengths: {bulk.wavelength_nm}")
    print(f"  n_legendre:  {bulk.n_legendre}")
    if bulk.size_distribution is not None:
        print(f"  Size dist:   {bulk.size_distribution.dist_type}")
    print()

    # Resolve single-size file paths
    single_paths = resolve_paths(args.singles_input)
    if not single_paths:
        print("Error: No single-size files found.", file=sys.stderr)
        return 1

    single_data = load_single_size_data(single_paths)
    if len(single_data) < 2:
        print(
            "Error: At least 2 single-size datasets are required for PCHIP interpolation.",
            file=sys.stderr,
        )
        return 1
    radii = [d.r_eff_nm for d in single_data]
    print(f"Loading {len(single_data)} single-size datasets...")
    print(f"  Radii range: {min(radii):.1f} - {max(radii):.1f} nm")
    print()

    # Determine size distribution for integration
    size_distribution = bulk.size_distribution
    if size_distribution is None:
        print(
            "Warning: bulk data has no size_distribution metadata; "
            "using default lognormal (rg=100, sigma_ln=0.5).",
            file=sys.stderr,
        )
        size_distribution = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)

    # Cross-validate per wavelength
    print("=" * 72)
    print(
        f"{'Wavelength (nm)':>16} {'g_A (quad)':>14} {'g_B (beta_1/3)':>16} {'|diff|':>14} {'Status':>8}"
    )
    print("-" * 72)

    all_pass = True
    for wl in bulk.wavelength_nm:
        g_a = compute_g_path_a(single_data, size_distribution, wl)
        g_b = compute_g_path_b(bulk, wl)

        diff = abs(g_a - g_b)
        status = "PASS" if diff < args.tol else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{wl:16.2f} {g_a:14.8f} {g_b:16.8f} {diff:14.6e} {status:>8}")

        # Hint about coefficient normalization confusion
        if abs(g_b) > 1e-10 and abs(g_a / g_b - 3.0) < 0.1:
            print(
                "  HINT: g_A/g_B ~ 3.0 suggests coefficient-normalization confusion. "
                "compute_legendre_moments() returns k_l = (2l+1)*integral, "
                "but beta_l = k_l/(2l+1). Divide by (2l+1) before bulk integration."
            )

    print("=" * 72)

    if all_pass:
        print(f"\nAll wavelengths PASS (tol = {args.tol:.0e}).")
        return 0
    else:
        print(f"\nOne or more wavelengths FAIL (tol = {args.tol:.0e}).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
