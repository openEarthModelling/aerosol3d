"""Thin aerosol3d wrapper over :mod:`pyradtran.viz`.

Re-exports the common RT plotting functions so aerosol3d code and examples
import them from a single aerosol3d-native location. Plotting logic lives in
pyRadtran; this module only adapts naming/convenience.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pyradtran.viz import (  # noqa: E402
    get_palette,
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

__all__ = [
    "plot_spectral",
    "plot_flux_profile",
    "plot_heating_rate",
    "plot_budget",
    "plot_rt_overview",
    "plot_composite_optics",
    "plot_component_attribution",
    "set_theme",
    "get_palette",
    "save",
    "build_comparison_figure",
]


def build_comparison_figure(bulk_ds, mono_ds, *, variable: str = "edir"):
    """Two-panel comparison: surface downward flux (bulk vs monodisperse) + rel. diff.

    Pure function (no I/O) so it is unit-testable without running libRadtran.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    set_theme()
    bulk_y = np.asarray(bulk_ds[variable].values, dtype=float).ravel()
    mono_y = np.asarray(mono_ds[variable].values, dtype=float).ravel()
    wl = np.asarray(bulk_ds["wavelength"].values, dtype=float)
    if bulk_y.size != mono_y.size:
        # interpolate monodisperse onto the bulk grid
        mono_y = np.interp(wl, np.asarray(mono_ds["wavelength"].values, dtype=float), mono_y)
    rel = (bulk_y - mono_y) / (np.abs(mono_y) + 1e-12) * 100.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_top.plot(wl, bulk_y, "b-", label="Bulk (lognormal)", linewidth=1.5)
    ax_top.plot(wl, mono_y, "r--", label="Monodisperse", linewidth=1.5)
    ax_top.set_ylabel(f"{variable} (surface)")
    ax_top.legend(loc="best")
    ax_bot.plot(wl, rel, "g-", linewidth=1.0)
    ax_bot.axhline(0, color="k", linewidth=0.5)
    ax_bot.set_xlabel("Wavelength (nm)")
    ax_bot.set_ylabel("Relative diff. (%)")
    fig.tight_layout()
    return fig
