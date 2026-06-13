"""NetCDF I/O for bulk aerosol optical properties."""

from __future__ import annotations

import ast

import numpy as np

from .datastructs import BulkAerosolOpticsData, SizeDistribution


def bulk_to_netcdf(bulk: BulkAerosolOpticsData, path: str) -> None:
    """Save bulk aerosol optical properties to a NetCDF file.

    Args:
        bulk: The bulk aerosol optics data to persist.
        path: Output file path.
    """
    import xarray as xr

    data_vars: dict = {
        "C_ext_nm2": (["wavelength"], bulk.C_ext),
        "C_sca_nm2": (["wavelength"], bulk.C_sca),
        "C_abs_nm2": (["wavelength"], bulk.C_abs),
        "SSA": (["wavelength"], bulk.SSA),
        "g": (["wavelength"], bulk.g),
        "beta": (["wavelength", "legendre_order"], bulk.beta),
    }

    coords: dict = {
        "wavelength_nm": (["wavelength"], bulk.wavelength_nm),
        "legendre_order": (["legendre_order"], np.arange(bulk.n_legendre)),
    }

    attrs: dict = {
        "r_eff_nm": float(bulk.r_eff_nm) if bulk.r_eff_nm is not None else "",
        "n_legendre": bulk.n_legendre,
        "interpolation_method": bulk.interpolation_method,
        "integration_method": bulk.integration_method,
        "integration_n_points": bulk.integration_n_points,
        "fallback_used": str(bulk.fallback_used),
        "fallback_wavelengths": str(bulk.fallback_wavelengths or []),
    }

    if bulk.tau_ref is not None:
        attrs["tau_ref"] = float(bulk.tau_ref)
    if bulk.effective_density_kg_m3 is not None:
        attrs["effective_density_kg_m3"] = float(bulk.effective_density_kg_m3)
    if bulk.concentration_method is not None:
        attrs["concentration_method"] = bulk.concentration_method
    if bulk.concentration_kwargs is not None:
        attrs["concentration_kwargs"] = str(bulk.concentration_kwargs)

    # Size distribution metadata
    if bulk.size_distribution is not None:
        attrs["size_distribution_type"] = bulk.size_distribution.dist_type
        attrs["size_distribution_params"] = str(bulk.size_distribution.params)
        attrs["size_distribution_r_min_nm"] = float(bulk.size_distribution.r_min_nm)
        attrs["size_distribution_r_max_nm"] = float(bulk.size_distribution.r_max_nm)

    # Per-radius provenance
    if bulk.radii_nm is not None:
        coords["radii_nm"] = (["radius"], bulk.radii_nm)
    if bulk.radii_weights is not None:
        data_vars["radii_weights"] = (["radius"], bulk.radii_weights)
    if bulk.per_radius_C_ext is not None:
        data_vars["per_radius_C_ext_nm2"] = (
            ["radius", "wavelength"],
            bulk.per_radius_C_ext,
        )
    if bulk.per_radius_C_sca is not None:
        data_vars["per_radius_C_sca_nm2"] = (
            ["radius", "wavelength"],
            bulk.per_radius_C_sca,
        )
    if bulk.per_radius_beta is not None:
        data_vars["per_radius_beta"] = (
            ["radius", "wavelength", "legendre_order"],
            bulk.per_radius_beta,
        )
    if bulk.legendre_moments_beta is not None:
        data_vars["legendre_moments_beta"] = (
            ["wavelength", "legendre_order"],
            bulk.legendre_moments_beta,
        )

    # Optional phase function grids
    if bulk.P11 is not None:
        assert bulk.theta_rad is not None and bulk.phi_rad is not None
        coords["theta_deg"] = (["theta"], np.degrees(bulk.theta_rad))
        coords["phi_deg"] = (["phi"], np.degrees(bulk.phi_rad))
        data_vars["P11"] = (["wavelength", "theta", "phi"], bulk.P11)

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path)


def bulk_from_netcdf(path: str) -> BulkAerosolOpticsData:
    """Load bulk aerosol optical properties from a NetCDF file.

    Args:
        path: Path to the NetCDF file.

    Returns:
        Reconstructed ``BulkAerosolOpticsData`` instance.
    """
    import xarray as xr

    ds = xr.open_dataset(path)

    # Reconstruct SizeDistribution if metadata is present
    size_distribution: SizeDistribution | None = None
    sd_type = ds.attrs.get("size_distribution_type")
    if sd_type is not None:
        params = ast.literal_eval(ds.attrs.get("size_distribution_params", "{}"))
        r_min = float(ds.attrs.get("size_distribution_r_min_nm", 1.0))
        r_max = float(ds.attrs.get("size_distribution_r_max_nm", 1e5))
        if sd_type == "lognormal":
            size_distribution = SizeDistribution.lognormal(
                rg_nm=params["rg_nm"], sigma_ln=params["sigma_ln"]
            )
        elif sd_type == "gamma":
            size_distribution = SizeDistribution.gamma(
                reff_nm=params["reff_nm"], veff=params["veff"]
            )
        else:
            # For "custom" we cannot recover the callable; store metadata only
            size_distribution = SizeDistribution(
                dist_type=sd_type,
                pdf=lambda r: np.zeros_like(r),  # placeholder
                params=params,
                r_min_nm=r_min,
                r_max_nm=r_max,
            )

    # Parse optional fields
    radii_nm = ds["radii_nm"].values if "radii_nm" in ds else None
    radii_weights = ds["radii_weights"].values if "radii_weights" in ds else None
    per_radius_C_ext = ds["per_radius_C_ext_nm2"].values if "per_radius_C_ext_nm2" in ds else None
    per_radius_C_sca = ds["per_radius_C_sca_nm2"].values if "per_radius_C_sca_nm2" in ds else None
    per_radius_beta = ds["per_radius_beta"].values if "per_radius_beta" in ds else None
    legendre_moments_beta = (
        ds["legendre_moments_beta"].values if "legendre_moments_beta" in ds else None
    )

    theta_rad = np.radians(ds["theta_deg"].values) if "theta_deg" in ds else None
    phi_rad = np.radians(ds["phi_deg"].values) if "phi_deg" in ds else None
    P11 = ds["P11"].values if "P11" in ds else None

    fallback_used = ds.attrs.get("fallback_used", "False") == "True"
    fallback_wavelengths_raw = ds.attrs.get("fallback_wavelengths", "[]")
    fallback_wavelengths: list[float] | None = ast.literal_eval(fallback_wavelengths_raw)
    if fallback_wavelengths == []:
        fallback_wavelengths = None

    tau_ref = float(ds.attrs["tau_ref"]) if "tau_ref" in ds.attrs else None
    effective_density_kg_m3 = (
        float(ds.attrs["effective_density_kg_m3"])
        if "effective_density_kg_m3" in ds.attrs
        else None
    )
    concentration_method = ds.attrs.get("concentration_method")
    concentration_kwargs_raw = ds.attrs.get("concentration_kwargs")
    concentration_kwargs = (
        ast.literal_eval(concentration_kwargs_raw) if concentration_kwargs_raw is not None else None
    )

    r_eff_raw = ds.attrs.get("r_eff_nm", None)
    r_eff_nm = float(r_eff_raw) if r_eff_raw not in (None, "") else None

    obj = BulkAerosolOpticsData(
        wavelength_nm=ds["wavelength_nm"].values,
        C_ext=ds["C_ext_nm2"].values,
        C_sca=ds["C_sca_nm2"].values,
        C_abs=ds["C_abs_nm2"].values,
        SSA=ds["SSA"].values,
        g=ds["g"].values,
        beta=ds["beta"].values,
        n_legendre=int(ds.attrs.get("n_legendre", ds.sizes.get("legendre_order", 0))),
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        P11=P11,
        size_distribution=size_distribution,
        radii_nm=radii_nm,
        radii_weights=radii_weights,
        per_radius_C_ext=per_radius_C_ext,
        per_radius_C_sca=per_radius_C_sca,
        per_radius_beta=per_radius_beta,
        legendre_moments_beta=legendre_moments_beta,
        effective_density_kg_m3=effective_density_kg_m3,
        r_eff_nm=r_eff_nm,
        interpolation_method=ds.attrs.get("interpolation_method", ""),
        integration_method=ds.attrs.get("integration_method", ""),
        integration_n_points=int(ds.attrs.get("integration_n_points", 0)),
        fallback_used=fallback_used,
        fallback_wavelengths=fallback_wavelengths,
        tau_ref=tau_ref,
        concentration_method=concentration_method,
        concentration_kwargs=concentration_kwargs,
    )
    ds.close()
    return obj


def bulk_to_vsmartmom_netcdf(bulk: BulkAerosolOpticsData, path: str, tau_ref: float = 0.5) -> None:
    """Export bulk aerosol optics to a minimal vSmartMOM-compatible NetCDF.

    Args:
        bulk: The bulk aerosol optics data.
        path: Output file path.
        tau_ref: Reference optical depth for the column.
    """
    import xarray as xr

    data_vars = {
        "bulk_SSA": (["wavelength"], bulk.SSA),
        "bulk_C_ext_nm2": (["wavelength"], bulk.C_ext),
        "bulk_C_sca_nm2": (["wavelength"], bulk.C_sca),
        "bulk_beta": (["wavelength", "legendre_order"], bulk.beta),
    }
    coords = {
        "wavelength_nm": (["wavelength"], bulk.wavelength_nm),
        "legendre_order": (["legendre_order"], np.arange(bulk.n_legendre)),
    }
    attrs = {
        "r_eff_nm": float(bulk.r_eff_nm) if bulk.r_eff_nm is not None else "",
        "tau_ref": float(tau_ref),
        "n_legendre": bulk.n_legendre,
    }

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path)
