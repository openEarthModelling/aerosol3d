"""Stage 3: Compare bulk vs. monodisperse radiative-transfer results.

Loads the two NetCDF outputs from Stage 2 (``rt_bulk.nc`` and
``rt_monodisperse.nc``), plots total downward irradiance (edir + edn) for
both runs plus the relative difference, and saves a single summary PNG.

Usage:
    python compare_results.py
"""

import logging

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / remote runs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import COMPARE_PNG, RT_BULK_NC, RT_MONODISPERSE_NC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def rel_diff(a, b):
    """Symmetric-ish relative difference in percent."""
    return (a - b) / (np.abs(b) + 1e-12) * 100.0


def _total_downward(ds: xr.Dataset) -> np.ndarray:
    """Return edir + edn as a 1-D array indexed by wavelength."""
    edir = ds["edir"].values if "edir" in ds else None
    edn = ds["edn"].values if "edn" in ds else None

    if edir is None and edn is None:
        raise KeyError("Dataset has neither 'edir' nor 'edn' variable.")

    # Some pyRadtran outputs are 2-D (wavelength, altitude); take the
    # surface value (last row) for a spectral comparison.
    def _surface(v):
        v = np.asarray(v, dtype=float)
        return v[-1] if v.ndim >= 2 else v

    total = np.zeros(
        max(edir.size if edir is not None else 0, edn.size if edn is not None else 0), dtype=float
    )
    if edir is not None:
        total = total + _surface(edir)
    if edn is not None:
        total = total + _surface(edn)
    return total


def _wavelengths_nm(ds: xr.Dataset) -> np.ndarray:
    """Pull the wavelength axis in nm from a pyRadtran result."""
    if "wavelength" in ds:
        return np.asarray(ds["wavelength"].values, dtype=float)
    if "lambda" in ds:
        return np.asarray(ds["lambda"].values, dtype=float)
    raise KeyError("Could not find wavelength coordinate in dataset.")


def main():
    if not RT_BULK_NC.exists():
        raise FileNotFoundError(f"Missing {RT_BULK_NC}; run run_radiative_transfer.py --tag bulk")
    if not RT_MONODISPERSE_NC.exists():
        raise FileNotFoundError(
            f"Missing {RT_MONODISPERSE_NC}; run run_radiative_transfer.py --tag monodisperse"
        )

    bulk = xr.open_dataset(RT_BULK_NC)
    mono = xr.open_dataset(RT_MONODISPERSE_NC)

    wl_bulk = _wavelengths_nm(bulk)
    wl_mono = _wavelengths_nm(mono)

    # Interpolate monodisperse onto the bulk wavelength grid if they differ.
    if not np.allclose(wl_bulk, wl_mono):
        logger.info(
            "Interpolating monodisperse (%d wl) onto bulk grid (%d wl)",
            wl_mono.size,
            wl_bulk.size,
        )
        mono_total_raw = _total_downward(mono)
        mono_total = np.interp(wl_bulk, wl_mono, mono_total_raw)
    else:
        mono_total = _total_downward(mono)
    bulk_total = _total_downward(bulk)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_top.plot(wl_bulk, bulk_total, "b-", label="Bulk (lognormal)", linewidth=1.5)
    ax_top.plot(wl_bulk, mono_total, "r--", label="Monodisperse", linewidth=1.5)
    ax_top.set_ylabel("Total downward irradiance\n(edir + edn)")
    ax_top.set_title("Bulk vs. Monodisperse: surface downward irradiance")
    ax_top.legend(loc="best")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(wl_bulk, rel_diff(bulk_total, mono_total), "g-", linewidth=1.0)
    ax_bot.axhline(0, color="k", linewidth=0.5)
    ax_bot.set_xlabel("Wavelength (nm)")
    ax_bot.set_ylabel("Relative diff. (%)")
    ax_bot.set_title("Bulk relative to monodisperse")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(COMPARE_PNG)
    plt.close(fig)
    logger.info("Saved comparison plot -> %s", COMPARE_PNG)

    # Also log a small numeric summary at 550 nm.
    idx_550 = int(np.argmin(np.abs(wl_bulk - 550.0)))
    logger.info(
        "At 550 nm: bulk=%.4e, mono=%.4e, rel_diff=%+.2f%%",
        float(bulk_total[idx_550]),
        float(mono_total[idx_550]),
        float(rel_diff(bulk_total[idx_550], mono_total[idx_550])),
    )

    bulk.close()
    mono.close()
    logger.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
