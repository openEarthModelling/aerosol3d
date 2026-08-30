"""DDA optical property computation via CoupledElectricMagneticDipoles.jl."""

import scipy.integrate

# PyMieScatt<=1.8.1.1 imports scipy.integrate.trapz, removed in SciPy 1.14.
# Arm the alias at package import so PyMieScatt imports cleanly regardless of
# who imports it first (e.g. pytest.importorskip in tests).
if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from .datastructs import OpticalResult, SimulationConfig  # noqa: F401
from .dda_solver import solve_optics  # noqa: F401
from .legendre import compute_legendre_moments  # noqa: F401
from .optics_export import AerosolOpticsData, from_optical_results  # noqa: F401
from .visualization import plot_near_field, plot_phase_function_2d, print_macroscopic  # noqa: F401
