import numpy as np
import pytest

from Aerosol3D.bulk.builder import BulkOpticsBuilder
from Aerosol3D.bulk.datastructs import SizeDistribution
from Aerosol3D.optics.optics_export import AerosolOpticsData


def _per_radius(radius_nm, density=1800.0):
    n_l = 4
    moments_beta = np.zeros((1, n_l))
    moments_beta[0, 0] = 1.0
    moments_beta[0, 1] = 0.5  # g
    return AerosolOpticsData(
        wavelength_nm=np.array([500.0]),
        C_ext=np.array([5.0]),
        C_sca=np.array([4.5]),
        C_abs=np.array([0.5]),
        SSA=np.array([0.9]),
        g=np.array([0.5]),
        r_eff_nm=radius_nm,
        n_legendre=n_l,
        legendre_moments_beta=moments_beta,
        density_kg_m3=density,
    )


def test_builder_fills_effective_density_and_gform():
    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.4)
    radii = np.array([80.0, 100.0, 125.0])
    builder = BulkOpticsBuilder(size_distribution=sd, radii_nm=radii, n_legendre=4)
    for r in radii:
        builder.add(float(r), _per_radius(float(r)))
    bulk = builder.compute()
    # single material -> effective density equals material density
    assert bulk.effective_density_kg_m3 == pytest.approx(1800.0)
    # g_l form present and consistent with beta
    assert bulk.legendre_moments_beta is not None
    l_vals = np.arange(bulk.n_legendre)
    np.testing.assert_allclose(bulk.legendre_moments_beta, bulk.beta / (2 * l_vals + 1))
