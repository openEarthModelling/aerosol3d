import numpy as np
import pytest
import xarray as xr

from Aerosol3D.optics.datastructs import (
    CrossSections,
    OpticalResult,
    SimulationConfig,
)
from Aerosol3D.optics.optics_export import (
    AerosolOpticsData,
    from_optical_results,
)


def test_density_field_default_none():
    data = AerosolOpticsData(
        wavelength_nm=np.array([500.0]),
        C_ext=np.array([1.0]),
        C_sca=np.array([0.9]),
        C_abs=np.array([0.1]),
        SSA=np.array([0.9]),
        g=np.array([0.5]),
        r_eff_nm=100.0,
    )
    assert data.density_kg_m3 is None


def test_density_netcdf_roundtrip(tmp_path):
    data = AerosolOpticsData(
        wavelength_nm=np.array([500.0]),
        C_ext=np.array([1.0]),
        C_sca=np.array([0.9]),
        C_abs=np.array([0.1]),
        SSA=np.array([0.9]),
        g=np.array([0.5]),
        r_eff_nm=100.0,
        density_kg_m3=1800.0,
    )
    path = tmp_path / "optics.nc"
    data.to_netcdf(path)
    loaded = AerosolOpticsData.from_netcdf(path)
    assert loaded.density_kg_m3 == pytest.approx(1800.0)


def test_density_absent_reads_as_none(tmp_path):
    data = AerosolOpticsData(
        wavelength_nm=np.array([500.0]),
        C_ext=np.array([1.0]),
        C_sca=np.array([0.9]),
        C_abs=np.array([0.1]),
        SSA=np.array([0.9]),
        g=np.array([0.5]),
        r_eff_nm=100.0,
        # density_kg_m3 intentionally omitted
    )
    path = tmp_path / "optics_no_density.nc"
    data.to_netcdf(path)
    loaded = AerosolOpticsData.from_netcdf(path)
    assert loaded.density_kg_m3 is None
    assert "density_kg_m3" not in xr.open_dataset(path).attrs


def _make_result(wl=550.0):
    cs = CrossSections(
        wavelength=wl,
        C_ext=100.0,
        C_sca=80.0,
        C_abs=20.0,
        Q_ext=2.0,
        Q_sca=1.6,
        Q_abs=0.4,
        SSA=0.8,
        g=0.7,
        r_eff=200.0,
    )
    cfg = SimulationConfig(wavelength=wl)
    return OpticalResult(
        config=cfg,
        cross_sections=cs,
        phase_function=None,
        solver="MIE",
    )


def test_from_optical_results_threads_density():
    results = [_make_result()]
    optics = from_optical_results(results, n_legendre=4, density_kg_m3=1800.0)
    assert optics.density_kg_m3 == pytest.approx(1800.0)
