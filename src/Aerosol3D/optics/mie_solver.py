"""MIE optical solver using PyMieScatt."""

import numpy as np

from .datastructs import CrossSections, OpticalResult, PhaseFunction, SimulationConfig

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _mie_phase_function(m, d, wavelength, n_theta=181) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase function P11(theta) using PyMieScatt.ScatteringFunction."""
    import PyMieScatt as pms  # noqa: N813

    angular_resolution = 180.0 / (n_theta - 1) if n_theta > 1 else 1.0
    theta_rad, _, _, SU = pms.ScatteringFunction(
        m,
        wavelength,
        d,
        nMedium=1.0,
        minAngle=0,
        maxAngle=180,
        angularResolution=angular_resolution,
    )
    sin_theta = np.sin(theta_rad)
    norm = 2 * np.pi * _trapz(SU * sin_theta, theta_rad)
    P11 = SU / norm if norm > 0 else SU
    return theta_rad, P11


def solve_mie(
    particle,
    config: SimulationConfig,
    compute_phase_func: bool = False,
    n_theta: int = 181,
    ema_method: str = "volume_weighted",
    verbose: bool = True,
) -> OpticalResult:
    """Solve optics using Mie theory (PyMieScatt)."""
    import PyMieScatt as pms  # noqa: N813

    m = particle.effective_refractive_index(method=ema_method) / config.n_host
    d = particle.equivalent_diameter
    wavelength = config.wavelength

    if verbose:
        print(f"{'=' * 52}")
        print("  MIE Simulation Configuration")
        print(f"{'=' * 52}")
        print(f"  wavelength     = {wavelength:.1f} nm")
        print(f"  n_host         = {config.n_host}")
        print(f"  d_ve           = {d:.2f} nm")
        print(f"  m              = {m}")
        print(f"  x              = {np.pi * d / wavelength:.4f} (size parameter)")
        print(f"{'=' * 52}")

    Qext, Qsca, Qabs, g, _, _, _ = pms.MieQ(m, wavelength, d, nMedium=config.n_host)

    r_eff = d / 2.0
    geo_cs = np.pi * r_eff**2

    C_ext = Qext * geo_cs
    C_sca = Qsca * geo_cs
    C_abs = Qabs * geo_cs
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    cross_sections = CrossSections(
        wavelength=wavelength,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        Q_ext=Qext,
        Q_sca=Qsca,
        Q_abs=Qabs,
        SSA=SSA,
        g=g,
        r_eff=r_eff,
    )

    phase_function = None
    if compute_phase_func:
        theta_rad, P11 = _mie_phase_function(m, d, wavelength, n_theta=n_theta)
        phi = np.array([0.0])
        P11_2d = P11[:, np.newaxis]
        phase_function = PhaseFunction(theta=theta_rad, phi=phi, P11=P11_2d)

    return OpticalResult(
        config=config,
        cross_sections=cross_sections,
        phase_function=phase_function,
        voxel_grid=None,
        n_dipoles=0,
        validity=None,
        solve_time=0.0,
        solver="MIE",
    )


def _coreshell_phase_function(
    mCore,  # noqa: N803
    mShell,  # noqa: N803
    wavelength,
    dCore,  # noqa: N803
    dShell,  # noqa: N803
    n_theta=181,  # noqa: N803
) -> tuple[np.ndarray, np.ndarray]:
    """Compute core-shell phase function P11(theta)."""
    import PyMieScatt as pms  # noqa: N813

    angular_resolution = 180.0 / (n_theta - 1) if n_theta > 1 else 1.0
    theta_rad, _, _, SU = pms.CoreShellScatteringFunction(
        mCore,
        mShell,
        wavelength,
        dCore,
        dShell,
        nMedium=1.0,
        minAngle=0,
        maxAngle=180,
        angularResolution=angular_resolution,
    )
    # CoreShellScatteringFunction may return n_theta-1 points; interpolate
    # to ensure exactly n_theta uniformly spaced angles.
    if theta_rad.shape[0] != n_theta:
        theta_uniform = np.linspace(0, np.pi, n_theta)
        SU = np.interp(theta_uniform, theta_rad, SU)
        theta_rad = theta_uniform
    sin_theta = np.sin(theta_rad)
    norm = 2 * np.pi * _trapz(SU * sin_theta, theta_rad)
    P11 = SU / norm if norm > 0 else SU
    return theta_rad, P11


def solve_mie_coreshell(
    particle,
    config: SimulationConfig,
    compute_phase_func: bool = False,
    n_theta: int = 181,
    verbose: bool = True,
) -> OpticalResult:
    """Solve optics using core-shell Mie theory (PyMieScatt)."""
    import PyMieScatt as pms  # noqa: N813

    d_core, d_outer, m_core, m_shell = particle.coreshell_geometry
    wavelength = config.wavelength

    if verbose:
        x_core = np.pi * d_core / wavelength
        x_outer = np.pi * d_outer / wavelength
        print(f"{'=' * 52}")
        print("  MIE CORE-SHELL Simulation Configuration")
        print(f"{'=' * 52}")
        print(f"  wavelength     = {wavelength:.1f} nm")
        print(f"  n_host         = {config.n_host}")
        print(f"  d_core         = {d_core:.2f} nm")
        print(f"  d_outer        = {d_outer:.2f} nm")
        print(f"  m_core         = {m_core}")
        print(f"  m_shell        = {m_shell}")
        print(f"  x_core         = {x_core:.4f}")
        print(f"  x_outer        = {x_outer:.4f}")
        print(f"{'=' * 52}")

    Qext, Qsca, Qabs, g, _, _, _ = pms.MieQCoreShell(
        m_core, m_shell, wavelength, d_core, d_outer, nMedium=config.n_host
    )

    r_eff = d_outer / 2.0
    geo_cs = np.pi * r_eff**2

    C_ext = Qext * geo_cs
    C_sca = Qsca * geo_cs
    C_abs = Qabs * geo_cs
    SSA = C_sca / C_ext if C_ext > 0 else 0.0

    cross_sections = CrossSections(
        wavelength=wavelength,
        C_ext=C_ext,
        C_sca=C_sca,
        C_abs=C_abs,
        Q_ext=Qext,
        Q_sca=Qsca,
        Q_abs=Qabs,
        SSA=SSA,
        g=g,
        r_eff=r_eff,
    )

    phase_function = None
    if compute_phase_func:
        theta_rad, P11 = _coreshell_phase_function(
            m_core, m_shell, wavelength, d_core, d_outer, n_theta=n_theta
        )
        phi = np.array([0.0])
        P11_2d = P11[:, np.newaxis]
        phase_function = PhaseFunction(theta=theta_rad, phi=phi, P11=P11_2d)

    return OpticalResult(
        config=config,
        cross_sections=cross_sections,
        phase_function=phase_function,
        voxel_grid=None,
        n_dipoles=0,
        validity=None,
        solve_time=0.0,
        solver="MIE_CORESHELL",
    )
