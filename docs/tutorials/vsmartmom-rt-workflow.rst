vSmartMOM RT Workflow
=====================

.. note::
   This tutorial requires Julia and vSmartMOM.jl. If they are not
   installed, you can still read through the code to understand the
   workflow.

Learning Objectives
-------------------

By the end of this tutorial you will know how to:

* Load pre-computed :class:`BulkAerosolOpticsData` from NetCDF
* Build a vertical concentration profile and height layers
* Execute column radiative transfer with :class:`VSmartMOMRunner`
* Inspect :class:`VSmartMOMResult` output

Prerequisites
-------------

1. Julia installed with ``julia`` on ``$PATH``
2. vSmartMOM.jl installed in a Julia project
3. A pre-computed bulk optics NetCDF file (e.g. from
   :doc:`bulk-optics-workflow`)

Step 1 — Load Bulk Optics
-------------------------

Load a :class:`BulkAerosolOpticsData` object that was previously saved
from a bulk optics computation:

.. code-block:: python

   from Aerosol3D import BulkAerosolOpticsData

   bulk = BulkAerosolOpticsData.from_netcdf("soot_bulk.nc")

   print(f"Wavelengths: {bulk.wavelength_nm} nm")
   print(f"SSA at {bulk.wavelength_nm[0]:.0f} nm: {bulk.SSA[0]:.4f}")

Step 2 — Define Vertical Profile
--------------------------------

Define layer boundaries (``heights``) and per-layer number concentrations
(``number_conc``):

.. code-block:: python

   # 5 layers: surface → 500 → 1000 → 2000 → 5000 → 10000 m
   heights = [0, 500, 1000, 2000, 5000, 10000]

   # Number concentration per layer [cm^-3]
   number_conc = [1200, 900, 600, 300, 100]

   # Verify layer count
   n_layers = len(number_conc)
   print(f"Number of layers: {n_layers}")

Step 3 — Configure Viewing Geometry
-----------------------------------

Set solar and viewing angles:

.. code-block:: python

   import numpy as np

   sza = 30.0                        # Solar zenith angle [deg]
   vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0])  # Viewing zenith [deg]

Step 4 — Run RT
---------------

Instantiate the runner and call :meth:`run_rt()`:

.. code-block:: python

   from Aerosol3D.vsmartmom import VSmartMOMRunner

   runner = VSmartMOMRunner(julia_project="~/Radiation/vSmartMOM-env")

   result = runner.run_rt(
       bulk=bulk,
       heights=heights,
       number_conc=number_conc,
       sza=sza,
       vza=vza,
   )

   print("RT complete!")

Step 5 — Inspect Results
------------------------

The :class:`VSmartMOMResult` contains TOA reflectance (``R``), BOA
transmittance (``T``), and per-layer optical depths:

.. code-block:: python

   # TOA reflectance at nadir, first wavelength
   R_nadir = result.R[0, 0, 0]
   print(f"TOA R (nadir, {result.wavelengths[0]:.0f} nm): {R_nadir:.4f}")

   # Total column optical depth at first wavelength
   tau_total = result.tau_per_layer[0, :].sum()
   print(f"Total column τ: {tau_total:.4f}")

   # Optical depth profile
   for i in range(n_layers):
       print(f"  Layer {i}: τ = {result.tau_per_layer[0, i]:.4f}")

Step 6 — Save Results
---------------------

Persist the full result to NetCDF for later analysis:

.. code-block:: python

   result.to_netcdf("vsmartmom_rt_result.nc")

Complete Script
---------------

.. code-block:: python

   import numpy as np
   from Aerosol3D import BulkAerosolOpticsData
   from Aerosol3D.vsmartmom import VSmartMOMRunner

   # Step 1 — Load bulk optics
   bulk = BulkAerosolOpticsData.from_netcdf("soot_bulk.nc")

   # Step 2 — Define profile
   heights = [0, 500, 1000, 2000, 5000, 10000]
   number_conc = [1200, 900, 600, 300, 100]
   n_layers = len(number_conc)

   # Step 3 — Viewing geometry
   sza = 30.0
   vza = np.array([0.0, 15.0, 30.0, 45.0, 60.0])

   # Step 4 — Run RT
   runner = VSmartMOMRunner(julia_project="~/Radiation/vSmartMOM-env")
   result = runner.run_rt(
       bulk=bulk,
       heights=heights,
       number_conc=number_conc,
       sza=sza,
       vza=vza,
   )

   # Step 5 — Inspect
   print(f"TOA R (nadir): {result.R[0, 0, 0]:.4f}")
   print(f"Total τ: {result.tau_per_layer[0, :].sum():.4f}")

   # Step 6 — Save
   result.to_netcdf("vsmartmom_rt_result.nc")

See Also
--------

* :doc:`../user-guide/vsmartmom-integration` — Full API reference and
  architecture documentation
* ``examples/bulk_pyradtran_pipeline/`` — Alternative RT pipeline using
  pyRadtran/DISORT
* :doc:`bulk-optics-workflow` — How to generate the ``BulkAerosolOpticsData``
  input used in this tutorial
