"""Validate DDA against Mie theory for a spherical particle.

Compares DDA accuracy at multiple precision levels (low, medium, high) using
LDR polarizability (Draine & Goodman 1993). Mie theory provides the exact
reference solution.
"""

import numpy as np

from Aerosol3D import AerosolParticle, Material, SimulationConfig, create_sphere, solve_optics


def run_mie(particle, wavelength=550.0):
    """Run Mie solver (exact reference)."""
    config = SimulationConfig(wavelength=wavelength, n_host=1.0)
    return solve_optics(particle, config, solver="MIE", compute_phase_func=True, verbose=False)


def run_dda(particle, wavelength=550.0, precision="high"):
    """Run DDA solver with auto voxel_size for given precision level."""
    config = SimulationConfig(wavelength=wavelength, n_host=1.0, precision=precision)
    return solve_optics(particle, config, solver="DDA", compute_phase_func=True, verbose=False)


def print_comparison(mie, dda_results, precisions):
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print(f"{'':>12} {'MIE':>10} | " + " | ".join(f"DDA-{p:>6}" for p in precisions))
    print("-" * 72)

    for label, getter in [
        ("Q_ext", lambda r: r.cross_sections.Q_ext),
        ("Q_sca", lambda r: r.cross_sections.Q_sca),
        ("g", lambda r: r.cross_sections.g),
    ]:
        mie_val = getter(mie)
        vals = [getter(d) for d in dda_results]
        row = f"{label:>12} {mie_val:10.4f} | "
        row += " | ".join(f"{v:10.4f}" for v in vals)
        print(row)

    print("-" * 72)
    for label, getter in [
        ("rel err Q_ext", lambda r: r.cross_sections.Q_ext),
        ("rel err Q_sca", lambda r: r.cross_sections.Q_sca),
        ("rel err g", lambda r: r.cross_sections.g),
    ]:
        mie_val = getter(mie)
        rel_errs = [abs(getter(d) - mie_val) / abs(mie_val) * 100 for d in dda_results]
        row = f"{label:>12} {'':>10} | "
        row += " | ".join(f"{e:8.2f} %" for e in rel_errs)
        print(row)

    print("-" * 72)
    for label, getter in [
        ("n_dipoles", lambda r: r.n_dipoles),
        ("|m|kd", lambda r: r.validity["m_k_d"] if r.validity else float("nan")),
    ]:
        vals = [getter(d) for d in dda_results]
        row = f"{label:>12} {'':>10} | "
        row += " | ".join(f"{v:10.4f}" for v in vals)
        print(row)

    print("=" * 72)


def plot_results(mie, dda_results, precisions):
    """Plot phase function comparison."""
    import matplotlib.pyplot as plt

    theta_mie = np.degrees(mie.phase_function.theta)
    P11_mie = mie.phase_function.P11[:, 0]

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phase function comparison
    ax = axes[0]
    ax.semilogy(theta_mie, P11_mie, "k-", label="MIE (exact)", linewidth=2)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for dda, prec, color in zip(dda_results, precisions, colors):
        theta_dda = np.degrees(dda.phase_function.theta)
        P11_dda = np.mean(dda.phase_function.P11, axis=1)
        ax.semilogy(theta_dda, P11_dda, color=color, label=f"DDA {prec}", linewidth=1.5)

    ax.set_xlabel("Scattering angle (degrees)")
    ax.set_ylabel("P11")
    ax.legend()
    ax.set_title("Phase Function: MIE vs DDA (LDR)")

    # Convergence: relative error vs precision
    ax = axes[1]
    mie_g = mie.cross_sections.g
    mie_Qext = mie.cross_sections.Q_ext
    mie_Qsca = mie.cross_sections.Q_sca

    err_Qext = [abs(d.cross_sections.Q_ext - mie_Qext) / mie_Qext * 100 for d in dda_results]
    err_Qsca = [abs(d.cross_sections.Q_sca - mie_Qsca) / mie_Qsca * 100 for d in dda_results]
    err_g = [abs(d.cross_sections.g - mie_g) / abs(mie_g) * 100 for d in dda_results]
    n_dips = [d.n_dipoles for d in dda_results]

    ax.semilogy(n_dips, err_Qext, "o-", label="Q_ext error (%)")
    ax.semilogy(n_dips, err_Qsca, "s-", label="Q_sca error (%)")
    ax.semilogy(n_dips, err_g, "^-", label="g error (%)")
    ax.set_xlabel("Number of dipoles")
    ax.set_ylabel("Relative error (%)")
    ax.legend()
    ax.set_title("DDA Convergence with LDR Polarizability")

    for i, prec in enumerate(precisions):
        ax.annotate(
            prec, (n_dips[i], err_g[i]), textcoords="offset points", xytext=(10, 5), fontsize=9
        )

    plt.tight_layout()
    plt.savefig("mie_vs_dda_phase_function.png", dpi=150)
    print("\nPlot saved: mie_vs_dda_phase_function.png")


def main():
    material = Material(name="test", refractive_index=complex(1.5, 0.01), density=1.0)

    particle = AerosolParticle(name="Sphere100")
    particle.add_mesh("core", create_sphere((0, 0, 0), 100.0), material)

    print(f"Sphere radius: 100 nm, diameter: {particle.equivalent_diameter:.1f} nm")
    print(f"Refractive index: m = {material.refractive_index}")
    wavelength = 550.0
    print(f"Wavelength: {wavelength} nm")
    m = abs(material.refractive_index)
    k = 2.0 * np.pi / wavelength
    print(f"|m| = {m:.2f}, k = {k:.5f} nm⁻¹")

    # Mie reference
    print("\n--- MIE solve (exact) ---")
    mie_result = run_mie(particle, wavelength)
    print(f"Q_ext = {mie_result.cross_sections.Q_ext:.4f}")
    print(f"Q_sca = {mie_result.cross_sections.Q_sca:.4f}")
    print(f"g     = {mie_result.cross_sections.g:.4f}")

    # DDA at multiple precision levels
    precisions = ["low", "medium", "high"]
    dda_results = []
    for prec in precisions:
        print(f"\n--- DDA solve (precision={prec}) ---")
        result = run_dda(particle, wavelength, precision=prec)
        dda_results.append(result)
        mkd = result.validity["m_k_d"] if result.validity else float("nan")
        print(
            f"n_dipoles={result.n_dipoles}, |m|kd={mkd:.3f}, "
            f"Q_ext={result.cross_sections.Q_ext:.4f}, "
            f"Q_sca={result.cross_sections.Q_sca:.4f}, "
            f"g={result.cross_sections.g:.4f}"
        )

    # Comparison table
    print_comparison(mie_result, dda_results, precisions)

    # Plot
    try:
        plot_results(mie_result, dda_results, precisions)
    except ImportError:
        print("\nmatplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
