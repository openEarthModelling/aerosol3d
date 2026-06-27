import numpy as np
import pytest
import xarray as xr

from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData


def _make_bulk(beta=None, legendre_moments_beta=None, density=None):
    # NOTE: default ``beta`` is zeros, which violates the documented invariant
    # ``beta[..., 0] == 1``. This is acceptable ONLY because these tests do not
    # assert ``beta`` invariants — every test needing a valid beta passes one.
    n_wl, n_l = 2, 4
    return BulkAerosolOpticsData(
        wavelength_nm=np.array([400.0, 600.0]),
        C_ext=np.array([10.0, 8.0]),
        C_sca=np.array([9.0, 7.0]),
        C_abs=np.array([1.0, 1.0]),
        SSA=np.array([0.9, 0.875]),
        g=np.array([0.5, 0.5]),
        beta=(np.zeros((n_wl, n_l)) if beta is None else beta),
        n_legendre=n_l,
        legendre_moments_beta=legendre_moments_beta,
        effective_density_kg_m3=density,
    )


def test_dual_legendre_fields_present():
    beta = np.array([[1.0, 1.5, 1.0, 0.5], [1.0, 1.5, 1.0, 0.5]])
    gform = beta / (2 * np.arange(4) + 1)
    b = _make_bulk(beta=beta, legendre_moments_beta=gform, density=1800.0)
    assert b.effective_density_kg_m3 == pytest.approx(1800.0)
    assert b.legendre_moments_beta is not None
    np.testing.assert_allclose(b.legendre_moments_beta, gform)


def test_dual_form_netcdf_roundtrip(tmp_path):
    beta = np.array([[1.0, 1.5, 1.0, 0.5], [1.0, 1.5, 1.0, 0.5]])
    gform = beta / (2 * np.arange(4) + 1)
    b = _make_bulk(beta=beta, legendre_moments_beta=gform, density=1800.0)
    path = tmp_path / "bulk.nc"
    b.to_netcdf(path)
    loaded = BulkAerosolOpticsData.from_netcdf(path)
    assert loaded.effective_density_kg_m3 == pytest.approx(1800.0)
    assert loaded.legendre_moments_beta is not None
    np.testing.assert_allclose(loaded.legendre_moments_beta, gform)


def test_dual_form_absent_reads_as_none(tmp_path):
    # A bulk object written WITHOUT legendre_moments_beta / effective_density_kg_m3
    # must read back as None and must NOT write them to the NetCDF file.
    b = _make_bulk()  # no legendre_moments_beta, no density
    path = tmp_path / "bulk_no_dual.nc"
    b.to_netcdf(path)
    loaded = BulkAerosolOpticsData.from_netcdf(path)
    assert loaded.legendre_moments_beta is None
    assert loaded.effective_density_kg_m3 is None
    ds = xr.open_dataset(path)
    assert "legendre_moments_beta" not in ds
    assert "effective_density_kg_m3" not in ds.attrs
