"""Shared configuration for the multi-component (LEGO) aerosol -> pyRadtran demo.

Three aerosol blocks with contrasting optics are built as independent
``BulkSpecies``, placed in the column, externally mixed into one
``CompositeAerosol``, and run through DISORT. ``run_comprehensive.py`` then
exercises the full ``pyradtran.viz`` plot surface and the component-attribution
workflow (leave-one-out).

Blocks:
  - black_carbon : strong absorber (low SSA)
  - sulfate      : near-pure scatterer (high SSA)
  - mineral_dust : coarse scatterer (size-dependent optics)
"""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Three LEGO blocks (name, material preset, lognormal size dist, per-block targets) ---
BLOCKS = [
    {
        "name": "black_carbon",
        "material": "black_carbon",
        "rg_nm": 100.0,
        "sigma_ln": 0.4,
        "radii_nm": [40.0, 60.0, 80.0, 100.0, 130.0, 170.0, 220.0, 290.0],
        "tau_550": 0.15,
        "scale_height_km": 1.5,
    },
    {
        "name": "sulfate",
        "material": "sulfate",
        "rg_nm": 150.0,
        "sigma_ln": 0.5,
        "radii_nm": [60.0, 100.0, 150.0, 200.0, 260.0, 330.0, 400.0, 500.0],
        "tau_550": 0.15,
        "scale_height_km": 2.0,
    },
    {
        "name": "mineral_dust",
        "material": "mineral_dust",
        "rg_nm": 500.0,
        "sigma_ln": 0.6,
        "radii_nm": [200.0, 350.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0],
        "tau_550": 0.20,
        "scale_height_km": 3.0,
    },
]

# --- Spectral + grid ---
WAVELENGTHS_NM = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
N_LEGENDRE = 32
ALTITUDE_GRID_KM = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0]  # descending (TOA -> surface)
REF_NM = 550.0

# --- RT scene ---
SZA_DEG = 30.0
ALBEDO = 0.1
N_STREAMS = 16
# Output levels (km; "toa" resolved by resolve_zout_tokens). Multi-level so that
# vertical flux / heating-rate profiles have a physical altitude axis.
ZOUT_LEVELS = [0, 1, 2, 4, 6, 8, 10, "toa"]

# libRadtran data dir (override via PYRADTRAN_DATA_PATH)
LIBRADTRAN_DATA = "/home/zhangfan/Project/20260319_SPEMBSSBDART/Radiation/libRadtran-2.0.6/data"

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
        "format": "ascii",
        "zout": ZOUT_LEVELS,
        "heating_rate": "local",
    },
}


def bulk_nc(name: str) -> Path:
    return OUTPUT_DIR / f"bulk_{name}.nc"
