"""Tests for the aerosol3d rt_visualization thin layer (headless)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("pyradtran.viz")  # needs pyradtran with the viz layer (sibling pyRadtran repo)
from Aerosol3D.optics.rt_visualization import build_comparison_figure, plot_spectral


def _ds(down: float) -> xr.Dataset:
    wl = np.array([300.0, 500.0, 800.0])
    return xr.Dataset(
        {"edir": ("wavelength", np.array([down, down * 0.9, down * 0.8]))},
        coords={"wavelength": wl},
    )


def test_reexport_plot_spectral_callable():
    assert callable(plot_spectral)


def test_build_comparison_figure_returns_figure_with_two_curves():
    fig = build_comparison_figure(_ds(1.0), _ds(0.95), variable="edir")
    axes = fig.get_axes()
    # top panel: 2 curves (bulk, mono); bottom panel: rel-diff curve + zero ref (axhline).
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 2
    plt.close(fig)
