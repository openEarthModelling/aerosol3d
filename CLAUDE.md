# Aerosol3D

Light scattering simulations for aerosol particles using DDA (Discrete Dipole Approximation) and Mie theory.

## Commands

```bash
python -m pytest tests/ -v                    # Run tests
python -m pytest -m "not slow" -v             # Skip slow DDA tests (~1-10 min each)
python -m pytest --cov=Aerosol3D              # Run with coverage (CI command)
python -m ruff check src/                     # Lint
python -m ruff format src/ tests/             # Format
pip install -e ".[dev]"                       # Dev install
pip install -e ".[docs]"                      # Docs dependencies
cd docs && make html                          # Build documentation
pre-commit run --all-files                    # Run pre-commit hooks manually
```

DDA solver requires Julia with CoupledElectricMagneticDipoles.jl installed.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SKIP_JULIA_TESTS=1` | Skip Julia DDA tests (used in CI) |
| `PYVISTA_OFF_SCREEN=true` | Enable headless PyVista rendering (CI/remote) |
| `PYRADTRAN_DATA_PATH` | Path to libRadtran data directory (for RT examples) |

## Architecture

```
src/Aerosol3D/
  bulk/           # Bulk aerosol optics (size distribution, binning, merging)
    builder.py      # BulkOpticsBuilder — construct from size distributions
    datastructs.py  # BulkAerosolOpticsData, SizeDistribution
    merge.py        # merge_method1, merge_method2, compute_bin_weights
    io.py           # NetCDF I/O for bulk optics
  core/           # Particle, Material, Aggregate dataclasses
    ema.py          # Effective Medium Approximation mixing rules
  factory/        # Mesh creation (from_file, from_fractal)
  geometry/       # Primitives (sphere, cube, ellipsoid), boolean ops, voxelization
  io/             # Voxel/VTP export
  materials.py    # Preset material database (soot, sulfate, dust, etc.)
  modeling/       # Coating models (CAM, CCM, distance, potential-edge, potential-void)
  optics/         # Optical property computation
    datastructs.py    # OpticalResult, CrossSections, PhaseFunction, SimulationConfig
    dda_solver.py     # solve_optics() — DDA + Mie dispatch
    mie_solver.py     # Mie theory solver
    bridge.py         # PyJulia bridge for DDA
    legendre.py       # compute_legendre_moments() — k_l = (2l+1)*integral convention
    optics_export.py   # AerosolOpticsData dataclass + from_optical_results() + NetCDF I/O
    visualization.py  # Plotting: spectral, phase function, comparison, Legendre diagnostics
  physics/        # Unit handling
  utils/          # Plotting utilities
```

## Key Patterns

- `solve_optics(particle, config, solver="DDA"|"MIE")` dispatches to DDA or Mie solver
- `from_optical_results(results, n_legendre=32)` builds `AerosolOpticsData` with auto-computed Legendre moments
- `AerosolOpticsData.to_netcdf()` / `.from_netcdf()` for persistence
- `OpticalResult` = single wavelength; `AerosolOpticsData` = multi-wavelength export container
- `BulkOpticsBuilder` = construct bulk optics from `SizeDistribution` + per-particle optics
- `preset_material(name)` = lookup refractive index data for common aerosol species

## Gotchas

- **DISORT PMOM format**: `compute_legendre_moments()` returns k_l = (2l+1)*integral (coefficient form). DISORT/libRadtran expects beta_l = k_l/(2l+1). Divide by (2l+1) before passing to pyRadtran.
- **DDA solver is slow**: Each wavelength takes 1-10 minutes depending on particle size. Mie is near-instant.
- **Julia bridge**: First call initializes Julia runtime (~30s). DDA requires CoupledElectricMagneticDipoles.jl.
- **Phase function angles**: DDA and Mie produce different theta grids. Comparison functions must interpolate.
- **`docs/superpowers/`**: Never git-commit files under this directory. They are internal workflow artifacts.

## Example Pipeline

`examples/bulk_pyradtran_pipeline/` — three-stage bulk aerosol optics → pyRadtran DISORT pipeline:

1. `compute_bulk_optics.py` — Build bulk aerosol optics from a size distribution (per-radius Mie via `BulkOpticsBuilder`), save to NetCDF via `BulkAerosolOpticsData` (auto-fills `effective_density_kg_m3` and dual-form Legendre moments)
2. `run_radiative_transfer.py` — Feed bulk optics into pyRadtran DISORT via `BulkSpecies` (requires libRadtran)
3. `compare_results.py` — Compare bulk (size-distribution-integrated) vs monodisperse@r_eff RT results

Set `PYRADTRAN_DATA_PATH` to libRadtran data directory before running RT.
