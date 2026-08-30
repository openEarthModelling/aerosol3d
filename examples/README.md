# Aerosol3D Examples

## Core Examples

### `black_carbon_sphere.py`
Basic DDA computation for a spherical soot particle.

### `black_carbon_fractal.py`
Fractal aggregate generation and optical properties.

### `coated_particle.py`
Apply coating models to a spherical core-shell particle.

### `coated_fractal_aggregate.py`
Coating algorithms on fractal aggregates with Mie comparison.

### `validate_mie_vs_dda.py`
Validate DDA against Mie theory for spherical particles.

## Bulk Optics Examples

### `bulk_method_comparison.py`
Cross-validation of Method 1 vs Method 2 for bulk asymmetry factor.

### `bulk_cross_validation.py`
Comprehensive validation suite for bulk optics computation.

## Pipeline Examples

### `bulk_pyradtran_pipeline/`
Three-stage pipeline (bulk aerosol optics → pyRadtran DISORT RT → comparison).

### `multicomponent_pipeline/`
Multi-component LEGO-block DISORT pipeline with visualization suite and component attribution.

## vSmartMOM Radiative Transfer

### `vsmartmom_rt_demo.py`
Run column radiative transfer using vSmartMOM.jl from Aerosol3D bulk optics.

Requires Julia with vSmartMOM.jl installed. Optionally specify a Julia project:

```bash
python vsmartmom_rt_demo.py --julia-project /path/to/vSmartMOM-env
```
