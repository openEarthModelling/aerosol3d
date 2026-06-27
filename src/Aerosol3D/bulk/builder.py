"""BulkOpticsBuilder — assemble per-radius optics into bulk properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData, SizeDistribution
    from Aerosol3D.optics.optics_export import AerosolOpticsData

logger = logging.getLogger(__name__)


@dataclass
class BulkOpticsBuilder:
    """Build bulk aerosol optical properties from per-radius entries.

    Usage::

        builder = BulkOpticsBuilder(
            size_distribution=sd,
            radii_nm=radii,
            n_legendre=32,
        )
        for r in radii:
            builder.add(r, optics[r])
        bulk = builder.compute()
    """

    size_distribution: SizeDistribution
    radii_nm: np.ndarray
    n_legendre: int = 32
    _entries: dict[float, AerosolOpticsData] = field(default_factory=dict)
    _wavelengths: np.ndarray | None = None

    def add(self, radius: float, optics: AerosolOpticsData) -> None:
        """Add a per-radius optical property entry.

        Args:
            radius: Particle radius in nanometers.
            optics: ``AerosolOpticsData`` for this radius.

        Raises:
            TypeError: If ``optics`` is not an ``AerosolOpticsData`` instance.
            ValueError: If ``n_legendre`` or wavelength grid is inconsistent.
        """
        from Aerosol3D.optics.optics_export import AerosolOpticsData

        if not isinstance(optics, AerosolOpticsData):
            raise TypeError(
                f"optics must be an AerosolOpticsData instance, got {type(optics).__name__}"
            )

        if optics.n_legendre != self.n_legendre:
            raise ValueError(
                f"n_legendre mismatch: builder expects {self.n_legendre}, got {optics.n_legendre}"
            )

        wl = np.asarray(optics.wavelength_nm, dtype=float)
        if self._wavelengths is None:
            self._wavelengths = wl.copy()
        else:
            if wl.shape != self._wavelengths.shape or not np.allclose(
                wl, self._wavelengths, atol=1e-12
            ):
                raise ValueError(
                    "wavelength grid mismatch: all entries must share the same wavelengths"
                )

        self._entries[float(radius)] = optics

    def add_from_netcdf(self, radius: float, path: str) -> None:
        """Load ``AerosolOpticsData`` from a NetCDF file and add it.

        Args:
            radius: Particle radius in nanometers.
            path: Path to the NetCDF file.
        """
        from Aerosol3D.optics.optics_export import AerosolOpticsData

        optics = AerosolOpticsData.from_netcdf(path)
        self.add(radius, optics)

    def compute(
        self,
        interpolation: str = "pchip",
        integration: str = "quad",
        n_quad: int = 256,
        check_mie_ripples: bool = False,
        refractive_index: complex | None = None,
        mie_ripple_min_points: int = 3,
        warn_on_fallback: bool = True,
    ) -> BulkAerosolOpticsData:
        """Compute bulk optical properties.

        Method 2 (continuous interpolation + integration) is used by default.
        If ``check_mie_ripples`` is enabled and the radius spacing is too
        coarse to resolve Mie oscillations, Method 1 (local bin weights) is
        used as a fallback for the affected wavelengths.

        Args:
            interpolation: Interpolation type (currently only ``"pchip"``).
            integration: Integration method — ``"quad"`` for adaptive
                Gauss-Kronrod (primary, spec-compliant) or ``"fixed_quad"``
                for fixed Gauss-Legendre quadrature.
            n_quad: Number of quadrature points for Method 2 (used only when
                ``integration="fixed_quad"``).
            check_mie_ripples: If ``True``, detect insufficient sampling for
                Mie oscillations and fall back to Method 1.
            refractive_index: Complex refractive index used for ripple
                period estimation. Required when ``check_mie_ripples=True``.
            mie_ripple_min_points: Minimum number of points per Mie
                oscillation period required to use Method 2.
            warn_on_fallback: If ``True``, log a warning when fallback
                to Method 1 occurs.

        Returns:
            ``BulkAerosolOpticsData`` with all fields populated.

        Raises:
            ValueError: If no entries have been added.
        """
        from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData
        from Aerosol3D.bulk.merge import compute_bin_weights, merge_method1, merge_method2

        if not self._entries:
            raise ValueError("No entries have been added to the builder")

        # Validate Mie ripple parameters
        if check_mie_ripples and refractive_index is None:
            # Attempt auto-extraction from first entry
            first_optics = next(iter(self._entries.values()))
            if (
                first_optics.refractive_index_real is not None
                and first_optics.refractive_index_imag is not None
            ):
                # Use the first wavelength's refractive index
                n_real = float(first_optics.refractive_index_real[0])
                n_imag = float(first_optics.refractive_index_imag[0])
                refractive_index = complex(n_real, n_imag)
                logger.debug(
                    "Auto-extracted refractive index from first entry: %s",
                    refractive_index,
                )
            else:
                raise ValueError(
                    "refractive_index is required when check_mie_ripples=True. "
                    "Either pass it explicitly, or ensure the added optics entries "
                    "have refractive_index_real and refractive_index_imag fields."
                )

        # Build sorted radius list and arrays
        radii = np.asarray(self.radii_nm, dtype=float)
        radii = np.sort(radii)
        n_radius = radii.shape[0]

        # Wavelength grid
        wavelengths = self._wavelengths
        assert wavelengths is not None
        n_wavelength = wavelengths.shape[0]

        # Build per-radius optical property arrays
        C_ext = np.empty((n_radius, n_wavelength), dtype=float)
        C_sca = np.empty((n_radius, n_wavelength), dtype=float)
        beta = np.empty((n_radius, n_wavelength, self.n_legendre), dtype=float)

        for i, r in enumerate(radii):
            optics = self._entries[float(r)]
            C_ext[i, :] = optics.C_ext
            C_sca[i, :] = optics.C_sca

            if optics.legendre_moments_beta is not None:
                # Convert from Aerosol3D convention (g_l = k_l/(2l+1))
                # to vSmartMOM convention (beta_l = k_l/k_0 = (2l+1)*g_l)
                l_vals = np.arange(self.n_legendre)
                beta[i, :, :] = optics.legendre_moments_beta * (2 * l_vals + 1)
            else:
                # Reconstruct beta in vSmartMOM convention: beta_l = (2l+1)*g_l
                # beta_0 = 1*1 = 1, beta_1 = 3*g, beta_{l>1} = 0
                beta[i, :, 0] = 1.0
                if self.n_legendre > 1:
                    beta[i, :, 1] = optics.g * 3.0
                if self.n_legendre > 2:
                    beta[i, :, 2:] = 0.0

        # Determine which wavelengths need Method 1 fallback
        fallback_mask = np.zeros(n_wavelength, dtype=bool)
        if check_mie_ripples and refractive_index is not None:
            m = complex(refractive_index)
            delta_m = abs(m - 1.0)
            if delta_m > 0.0:
                for j in range(n_wavelength):
                    wl = float(wavelengths[j])
                    period = wl / (2.0 * delta_m)
                    # Check all adjacent dr
                    drs = np.diff(radii)
                    if drs.size == 0:
                        # Single radius — no ripple concern
                        continue
                    min_period_needed = period / mie_ripple_min_points
                    if np.any(drs > min_period_needed):
                        fallback_mask[j] = True
                        if warn_on_fallback:
                            logger.warning(
                                "Mie ripple fallback to Method 1 for wavelength %.1f nm "
                                "(period=%.1f nm, max dr=%.1f nm > %.1f nm)",
                                wl,
                                period,
                                float(np.max(drs)),
                                min_period_needed,
                            )

        # Pre-compute bin weights for potential Method 1 fallback
        if n_radius == 1:
            weights = np.ones(1, dtype=float)
        elif np.any(fallback_mask):
            weights = compute_bin_weights(radii, self.size_distribution)
        else:
            weights = None

        # Allocate bulk result arrays
        bulk_C_ext = np.empty(n_wavelength, dtype=float)
        bulk_C_sca = np.empty(n_wavelength, dtype=float)
        bulk_beta = np.empty((n_wavelength, self.n_legendre), dtype=float)

        fallback_wavelengths: list[float] = []

        for j in range(n_wavelength):
            if fallback_mask[j] or n_radius == 1:
                # Method 1 fallback for this wavelength (or monodisperse limit)
                m1_C_ext, m1_C_sca, m1_beta = merge_method1(
                    C_ext[:, j : j + 1],
                    C_sca[:, j : j + 1],
                    beta[:, j : j + 1, :],
                    weights,  # type: ignore[arg-type]
                )
                bulk_C_ext[j] = m1_C_ext[0]
                bulk_C_sca[j] = m1_C_sca[0]
                bulk_beta[j, :] = m1_beta[0, :]
                if fallback_mask[j]:
                    fallback_wavelengths.append(float(wavelengths[j]))
            else:
                # Method 2 for this wavelength
                m2_C_ext, m2_C_sca, m2_beta = merge_method2(
                    radii,
                    C_ext[:, j : j + 1],
                    C_sca[:, j : j + 1],
                    beta[:, j : j + 1, :],
                    self.size_distribution,
                    n_quad=n_quad,
                    interpolation=interpolation,
                    integration=integration,
                )
                bulk_C_ext[j] = m2_C_ext[0]
                bulk_C_sca[j] = m2_C_sca[0]
                bulk_beta[j, :] = m2_beta[0, :]

        # Derived quantities
        C_abs = bulk_C_ext - bulk_C_sca
        SSA = np.empty_like(bulk_C_ext)
        mask_positive = bulk_C_ext > 0.0
        SSA[mask_positive] = bulk_C_sca[mask_positive] / bulk_C_ext[mask_positive]
        SSA[~mask_positive] = 0.0

        g = np.empty_like(bulk_C_ext)
        if self.n_legendre > 1:
            g[:] = bulk_beta[:, 1] / 3.0
        else:
            g[:] = 0.0

        r_eff_nm = self.size_distribution.effective_radius()

        # Dual Legendre form: g_l = k_l / (2l+1)
        legendre_moments_beta = None
        if bulk_beta is not None:
            l_vals = np.arange(self.n_legendre)
            legendre_moments_beta = bulk_beta / (2.0 * l_vals + 1.0)

        # Effective (volume-weighted) density from per-entry material density
        effective_density_kg_m3 = None
        densities = [
            (r, self._entries[r].density_kg_m3)
            for r in self._entries
            if self._entries[r].density_kg_m3 is not None
        ]
        if len(densities) == len(self._entries) and densities:
            from Aerosol3D.bulk.density import compute_effective_density

            effective_density_kg_m3 = compute_effective_density(
                np.array([r for r, _ in densities]),
                np.array([d for _, d in densities]),
                self.size_distribution,
                n_quad=n_quad,
                method=integration,
            )

        return BulkAerosolOpticsData(
            wavelength_nm=wavelengths.copy(),
            C_ext=bulk_C_ext,
            C_sca=bulk_C_sca,
            C_abs=C_abs,
            SSA=SSA,
            g=g,
            beta=bulk_beta,
            n_legendre=self.n_legendre,
            size_distribution=self.size_distribution,
            radii_nm=radii.copy(),
            radii_weights=weights if weights is not None else np.ones(n_radius) / n_radius,
            per_radius_C_ext=C_ext.copy(),
            per_radius_C_sca=C_sca.copy(),
            per_radius_beta=beta.copy(),
            r_eff_nm=r_eff_nm,
            interpolation_method=interpolation,
            integration_method=integration,
            integration_n_points=n_quad,
            fallback_used=bool(np.any(fallback_mask)),
            fallback_wavelengths=fallback_wavelengths if fallback_wavelengths else None,
            legendre_moments_beta=legendre_moments_beta,
            effective_density_kg_m3=effective_density_kg_m3,
        )
