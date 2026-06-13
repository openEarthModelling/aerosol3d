"""Generic multi-wavelength aerosol optical property export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .datastructs import OpticalResult

from .legendre import compute_legendre_moments


@dataclass
class AerosolOpticsData:
    """Multi-wavelength aerosol optical properties."""

    wavelength_nm: np.ndarray
    C_ext: np.ndarray
    C_sca: np.ndarray
    C_abs: np.ndarray
    SSA: np.ndarray
    g: np.ndarray
    r_eff_nm: float
    density_kg_m3: float | None = None

    theta_rad: np.ndarray | None = None
    phi_rad: np.ndarray | None = None
    P11: np.ndarray | None = None

    legendre_moments: np.ndarray | None = None
    legendre_moments_beta: np.ndarray | None = None
    n_legendre: int = 32

    solver: str = ""
    material: str = ""
    refractive_index_real: np.ndarray | None = None
    refractive_index_imag: np.ndarray | None = None

    def to_netcdf(self, path: str) -> None:
        """Save optical properties to NetCDF file."""
        import xarray as xr

        data_vars = {
            "C_ext_nm2": (["wavelength"], self.C_ext),
            "C_sca_nm2": (["wavelength"], self.C_sca),
            "C_abs_nm2": (["wavelength"], self.C_abs),
            "SSA": (["wavelength"], self.SSA),
            "g": (["wavelength"], self.g),
        }
        coords = {
            "wavelength_nm": (["wavelength"], self.wavelength_nm),
        }
        attrs = {
            "r_eff_nm": float(self.r_eff_nm),
            "n_legendre": self.n_legendre,
            "solver": self.solver,
            "material": self.material,
        }
        if self.density_kg_m3 is not None:
            attrs["density_kg_m3"] = float(self.density_kg_m3)

        if self.P11 is not None:
            assert self.theta_rad is not None and self.phi_rad is not None
            coords["theta_deg"] = (["theta"], np.degrees(self.theta_rad))
            coords["phi_deg"] = (["phi"], np.degrees(self.phi_rad))
            data_vars["P11"] = (["wavelength", "theta", "phi"], self.P11)

        if self.legendre_moments is not None:
            coords["legendre_order"] = (
                ["legendre_order"],
                np.arange(self.n_legendre),
            )
            data_vars["legendre_moments"] = (
                ["wavelength", "legendre_order"],
                self.legendre_moments,
            )

        if self.legendre_moments_beta is not None:
            data_vars["legendre_moments_beta"] = (
                ["wavelength", "legendre_order"],
                self.legendre_moments_beta,
            )

        if self.refractive_index_real is not None:
            data_vars["refractive_index_real"] = (
                ["wavelength"],
                self.refractive_index_real,
            )
        if self.refractive_index_imag is not None:
            data_vars["refractive_index_imag"] = (
                ["wavelength"],
                self.refractive_index_imag,
            )

        ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str) -> AerosolOpticsData:
        """Load optical properties from NetCDF file."""
        import xarray as xr

        ds = xr.open_dataset(path)

        P11 = ds["P11"].values if "P11" in ds else None
        theta_rad = np.radians(ds["theta_deg"].values) if "theta_deg" in ds else None
        phi_rad = np.radians(ds["phi_deg"].values) if "phi_deg" in ds else None
        legendre_moments = ds["legendre_moments"].values if "legendre_moments" in ds else None
        legendre_moments_beta = (
            ds["legendre_moments_beta"].values if "legendre_moments_beta" in ds else None
        )
        n_real = ds["refractive_index_real"].values if "refractive_index_real" in ds else None
        n_imag = ds["refractive_index_imag"].values if "refractive_index_imag" in ds else None

        obj = cls(
            wavelength_nm=ds["wavelength_nm"].values,
            C_ext=ds["C_ext_nm2"].values,
            C_sca=ds["C_sca_nm2"].values,
            C_abs=ds["C_abs_nm2"].values,
            SSA=ds["SSA"].values,
            g=ds["g"].values,
            r_eff_nm=float(ds.attrs["r_eff_nm"]),
            density_kg_m3=float(ds.attrs["density_kg_m3"]) if "density_kg_m3" in ds.attrs else None,
            theta_rad=theta_rad,
            phi_rad=phi_rad,
            P11=P11,
            legendre_moments=legendre_moments,
            legendre_moments_beta=legendre_moments_beta,
            n_legendre=int(ds.attrs.get("n_legendre", 32)),
            solver=ds.attrs.get("solver", ""),
            material=ds.attrs.get("material", ""),
            refractive_index_real=n_real,
            refractive_index_imag=n_imag,
        )
        ds.close()
        return obj


def from_optical_results(
    results: list[OpticalResult],
    n_legendre: int = 32,
    material_name: str = "",
    density_kg_m3: float | None = None,
) -> AerosolOpticsData:
    """Build AerosolOpticsData from a list of OpticalResult.

    If phase functions are available, Legendre moments are auto-computed.

    Args:
        results: List of OpticalResult, one per wavelength.
        n_legendre: Number of Legendre moments to compute.
        material_name: Optional material name for metadata.
        density_kg_m3: Material density in kg/m^3 (convert from
            Material.density in g/cm^3 x1000).

    Returns:
        AerosolOpticsData with all extracted optical properties.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("results list cannot be empty")

    n_wl = len(results)
    wavelength_nm = np.array([r.cross_sections.wavelength for r in results])
    C_ext = np.array([r.cross_sections.C_ext for r in results])
    C_sca = np.array([r.cross_sections.C_sca for r in results])
    C_abs = np.array([r.cross_sections.C_abs for r in results])
    SSA = np.array([r.cross_sections.SSA for r in results])
    g = np.array([r.cross_sections.g for r in results])
    r_eff_nm = results[0].cross_sections.r_eff

    has_pf = results[0].phase_function is not None
    theta_rad = None
    phi_rad = None
    P11 = None
    legendre_moments = None
    legendre_moments_beta = None

    if has_pf:
        pf0 = results[0].phase_function
        assert pf0 is not None  # guaranteed by has_pf
        theta_rad = pf0.theta
        phi_rad = pf0.phi
        n_theta = len(theta_rad)
        n_phi = len(phi_rad)
        P11 = np.zeros((n_wl, n_theta, n_phi))
        legendre_moments = np.zeros((n_wl, n_legendre))
        legendre_moments_beta = np.zeros((n_wl, n_legendre))

        l_vals = np.arange(n_legendre)
        for i, r in enumerate(results):
            pf = r.phase_function
            assert pf is not None
            P11[i, :, :] = pf.P11
            moments = compute_legendre_moments(pf, n_legendre=n_legendre)
            legendre_moments[i, :] = moments
            legendre_moments_beta[i, :] = moments / (2 * l_vals + 1)

    return AerosolOpticsData(
        wavelength_nm=wavelength_nm,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        SSA=SSA,
        g=g,
        r_eff_nm=r_eff_nm,
        density_kg_m3=density_kg_m3,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        P11=P11,
        legendre_moments=legendre_moments,
        legendre_moments_beta=legendre_moments_beta,
        n_legendre=n_legendre,
        solver=results[0].solver,
        material=material_name,
    )
