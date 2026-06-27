"""Shared configuration for the bulk aerosol -> pyRadtran DISORT pipeline.

This example demonstrates Tasks 1-8 of the bulk-pyradtran integration:
- Stage 1: build bulk aerosol optics from a size distribution (aerosol3D)
- Stage 2: feed the bulk optics into pyRadtran's DISORT solver
- Stage 3: compare bulk vs. monodisperse RT results
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Material (must be a key in Aerosol3D.materials.REFRACTIVE_INDEX)
# ---------------------------------------------------------------------------
MATERIAL = "black_carbon"  # Bond & Bergstrom 2006, m=1.95+0.79i, rho=1.8 g/cm^3

# ---------------------------------------------------------------------------
# Size distribution (lognormal, geometric-mean radius)
# ---------------------------------------------------------------------------
RG_NM = 100.0  # geometric mean radius (nm)
SIGMA_LN = 0.4  # geometric standard deviation (ln-space)

# Discrete radii sampled from the distribution for per-particle Mie solves.
# Must span the bulk of the PDF and be dense enough to resolve Mie oscillations.
RADII_NM = [40.0, 60.0, 80.0, 100.0, 130.0, 170.0, 220.0, 290.0]

# Monodisperse reference (single radius used for the comparison run)
R_EFF_MONODISPERSE_NM = 100.0

# ---------------------------------------------------------------------------
# Wavelength grid (nm)
# ---------------------------------------------------------------------------
WAVELENGTHS_NM = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]

# ---------------------------------------------------------------------------
# Legendre expansion order
# ---------------------------------------------------------------------------
N_LEGENDRE = 32

# ---------------------------------------------------------------------------
# Vertical profile (exponential, defined by scale height + total OD@550)
# ---------------------------------------------------------------------------
TOTAL_OPTICAL_DEPTH_550 = 0.3
SCALE_HEIGHT_KM = 1.5
# Layer boundaries, descending (TOA -> surface) — required by pyRadtran
ALTITUDE_GRID_KM = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0]

# ---------------------------------------------------------------------------
# RT scene configuration (pyRadtran Scene builder)
# ---------------------------------------------------------------------------
SZA_DEG = 30.0
ALBEDO = 0.1
N_STREAMS = 16

# ---------------------------------------------------------------------------
# libRadtran data path (override via PYRADTRAN_DATA_PATH env var)
# ---------------------------------------------------------------------------
LIBRADTRAN_DATA = "/home/zhangfan/Project/20260319_SPEMBSSBDART/Radiation/libRadtran-2.0.6/data"

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------
BULK_OPTICS_NC = OUTPUT_DIR / "bulk_optics.nc"
MONODISPERSE_OPTICS_NC = OUTPUT_DIR / "monodisperse_optics.nc"
RT_BULK_NC = OUTPUT_DIR / "rt_bulk.nc"
RT_MONODISPERSE_NC = OUTPUT_DIR / "rt_monodisperse.nc"
COMPARE_PNG = OUTPUT_DIR / "compare_bulk_vs_monodisperse.png"

# ---------------------------------------------------------------------------
# Scene configuration dictionary (mirrors the existing example's SCENE_CONFIG)
# ---------------------------------------------------------------------------
SCENE_CONFIG = {
    "atmosphere": {"profile": "us", "altitude": 0.0},
    "source": {"sza": SZA_DEG},
    "wavelength": {"min_nm": 401.0, "max_nm": 699.0},
    "solver": {
        "method": "disort",
        "streams": N_STREAMS,
        "disort_intcor": "moments",
        "pseudospherical": True,
    },
    "surface": {"albedo": ALBEDO},
    "output": {
        "quantities": ["lambda", "edir", "edn", "eup"],
        "quantity": "transmittance",
        "format": "ascii",
    },
}
