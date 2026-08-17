Installation
============

Requirements
------------

- Python >= 3.10
- Julia >= 1.9 (optional, required for DDA optical computation)
- CUDA-capable GPU (optional, for GPU-accelerated DDA)

Basic Installation
------------------

Install from PyPI::

    pip install Aerosol3D

For development (editable install)::

    git clone https://github.com/openEarthModelling/Aerosol3D.git
    cd Aerosol3D
    pip install -e ".[dev]"

Julia Backend (for DDA)
-----------------------

The DDA solver requires Julia and the ``CoupledElectricMagneticDipoles.jl``
package::

    pip install pyjulia
    python -c "import julia; julia.install()"

Then in a Python session::

    from julia import CoupledElectricMagneticDipoles as CEMD

GPU Acceleration
----------------

GPU-accelerated DDA requires CUDA and the Julia CUDA packages. See the
CEMD.jl documentation for setup instructions.

Additional Optional Dependencies
---------------------------------

- **Mie solver** — Requires ``PyMieScatt`` (included with Aerosol3D)
- **Optical visualization** — Requires ``matplotlib`` and ``numpy``
- **NetCDF I/O** — Requires ``xarray`` and ``netCDF4`` for
  ``AerosolOpticsData.to_netcdf()`` / ``.from_netcdf()``
- **Radiative transfer pipeline** — Requires ``pyRadtran`` and libRadtran
  (see the ``examples/bulk_pyradtran_pipeline/`` directory).
  Set ``PYRADTRAN_DATA_PATH`` to the libRadtran data directory.
- **Parallel orientation averaging** — ``tqdm`` and ``joblib`` are installed
  automatically with Aerosol3D.  They provide progress bars and multi-core
  dispatch for DDA orientational averaging (``n_jobs > 1``).
