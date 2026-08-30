Mie vs DDA Validation
=====================

This tutorial validates DDA results against exact Mie theory for a spherical
particle. The example compares DDA accuracy at multiple precision levels
(low, medium, high) using **LDR polarizability** (Draine & Goodman 1993).
Mie theory provides the analytical reference solution for spheres.

Full Script
-----------

.. literalinclude:: ../../examples/validate_mie_vs_dda.py
   :language: python
   :linenos:

Step-by-Step Explanation
------------------------

1. **Create a spherical particle**

   A 100 nm radius sphere with refractive index m = 1.5 + 0.01i is created.
   The equivalent diameter and refractive index parameters are printed.

2. **Run Mie solver** (exact reference)

   ``solve_optics(particle, config, solver="MIE")`` computes the exact Mie
   solution, providing Q_ext, Q_sca, g, and the full phase function P11 as
   ground truth.

3. **Run DDA at multiple precision levels**

   DDA is run at three precision levels: ``"low"``, ``"medium"``, ``"high"``.
   Each level controls the dipole spacing via the :math:`|m|kd` convergence
   criterion. Lower values produce more dipoles and better accuracy.

   .. code-block:: python

       for prec in ["low", "medium", "high"]:
           config = SimulationConfig(wavelength=550.0, precision=prec)
           result = solve_optics(particle, config, solver="DDA")

4. **Compare and visualize**

   A formatted table prints Q_ext, Q_sca, g and their relative errors
   against Mie for each precision level. Key metrics (n_dipoles, :math:`|m|kd`)
   show how precision controls accuracy.

   If matplotlib is available, two plots are saved:

   - **Phase function comparison** — semi-log P11 for Mie and each DDA
     precision level
   - **Convergence plot** — relative error vs. number of dipoles, showing
     how DDA converges toward Mie as precision increases

Further Analysis
-----------------

For multi-wavelength comparison with the full visualization toolkit, use
``Aerosol3D.optics.visualization``, which provides
``plot_optical_comparison()``, ``plot_phase_function_comparison()``, and
``generate_comparison_summary()`` for detailed DDA vs Mie analysis.
