"""Example: Coated black carbon fractal aggregate — Mie approximation comparison.

Demonstrates:
1. Generate a BC fractal aggregate with sulfate coating
2. Compare four Mie scattering approximations:
   - Homogeneous sphere with volume-weighted EMA
   - Homogeneous sphere with Maxwell-Garnett EMA
   - Homogeneous sphere with Bruggeman EMA
   - Core-shell sphere (BC core + sulfate shell)
3. Plot optical property comparison across wavelengths

Usage:
    python coated_fractal_aggregate.py
    python coated_fractal_aggregate.py --no-video
    python coated_fractal_aggregate.py --no-video --wavelengths 450 550 650
"""

import argparse
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"

WAVELENGTHS = [450.0, 550.0, 650.0, 800.0]


def build_coated_particle():
    """Generate a BC fractal aggregate with potential void-filling coating."""
    from pyFracAggregate import Monodisperse
    from pyFracAggregate import generate as gen_fractal

    from Aerosol3D import (
        apply_potential_void_coating,
        from_fractal,
        preset_material,
    )

    soot = preset_material("black_carbon")
    sulfate = preset_material("sulfate")

    print("Generating BC fractal aggregate (30 monomers, PCA, Df=1.8)...")
    agg = gen_fractal(
        n_particles=30,
        df=1.8,
        kf=1.2,
        method="pca",
        particle_dist=Monodisperse(25.0),
    )
    print(f"  Generated: {agg.current_size} monomers")

    fractal = from_fractal(agg, soot)
    particle = fractal.to_particle()
    print(f"  Particle: {particle}")

    print("Applying potential void-filling coating...")
    apply_potential_void_coating(
        particle,
        coated_area_fraction=0.8,
        dp_dc_ratio=2.0,
        material=sulfate,
        resolution=48,
    )
    print(f"  Coated particle: {particle}")

    return particle, soot, sulfate


def compute_mie_approximations(particle, wavelengths):
    """Compute optical properties with four Mie approximation methods."""
    from Aerosol3D.optics.datastructs import SimulationConfig
    from Aerosol3D.optics.dda_solver import solve_optics
    from Aerosol3D.optics.optics_export import from_optical_results

    methods = [
        ("Mie (VW)", "MIE", {"ema_method": "volume_weighted"}),
        ("Mie (MG)", "MIE", {"ema_method": "maxwell_garnett"}),
        ("Mie (BG)", "MIE", {"ema_method": "bruggeman"}),
        ("Mie (CS)", "MIE_CORESHELL", {}),
    ]

    datasets = []
    for label, solver, kwargs in methods:
        print(f"\n  Computing {label}...")
        config = SimulationConfig(wavelength=wavelengths)
        results = solve_optics(
            particle,
            config,
            solver=solver,
            compute_phase_func=True,
            verbose=False,
            **kwargs,
        )
        if not isinstance(results, list):
            results = [results]

        data = from_optical_results(results, material_name=label)
        data.solver = label
        datasets.append(data)

        cs = results[0].cross_sections
        print(f"    @ 550nm: Q_ext={cs.Q_ext:.4f}, SSA={cs.SSA:.4f}, g={cs.g:.4f}")

    return datasets


def plot_comparison(datasets, labels, output_dir):
    """Plot comparison of optical properties across Mie approximations."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    props = [
        ("Extinction cross section", "C_ext", "nm²"),
        ("Scattering cross section", "C_sca", "nm²"),
        ("Absorption cross section", "C_abs", "nm²"),
        ("Single scattering albedo", "SSA", ""),
        ("Asymmetry parameter", "g", ""),
    ]

    # --- Figure 1: Scalar properties comparison ---
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for col, (title, attr, unit) in enumerate(props):
        ax = axes[col]
        for i, (ds, name) in enumerate(zip(datasets, labels)):
            y = getattr(ds, attr)
            ax.plot(
                ds.wavelength_nm,
                y,
                color=colors[i],
                marker=markers[i],
                label=name,
                linewidth=1.5,
                markersize=6,
            )
        ylabel = f"{title} ({unit})" if unit else title
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Mie Approximation Comparison: Coated BC Fractal Aggregate", fontsize=13)
    fig.tight_layout()
    path = output_dir / "mie_comparison_properties.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Figure 2: Relative difference vs volume-weighted baseline ---
    ref = datasets[0]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for col, (title, attr, unit) in enumerate(props):
        ax = axes[col]
        ref_vals = getattr(ref, attr)
        for i in range(1, len(datasets)):
            vals = getattr(datasets[i], attr)
            rel = (vals - ref_vals) / (np.abs(ref_vals) + 1e-12) * 100
            ax.plot(
                datasets[i].wavelength_nm,
                rel,
                color=colors[i],
                marker=markers[i],
                label=f"{labels[i]} vs {labels[0]}",
                linewidth=1.5,
                markersize=6,
            )
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Relative diff. (%)")
        ax.set_title(f"Δ{title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Relative Difference vs Volume-Weighted Baseline", fontsize=13)
    fig.tight_layout()
    path = output_dir / "mie_comparison_relative_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Figure 3: Phase function comparison per wavelength ---
    n_wl = len(datasets[0].wavelength_nm)
    for i_wl in range(n_wl):
        wl = datasets[0].wavelength_nm[i_wl]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale
        ax = axes[0]
        for i, (ds, name) in enumerate(zip(datasets, labels)):
            if ds.P11 is not None:
                P11 = np.mean(ds.P11[i_wl], axis=1)
                theta_deg = np.degrees(ds.theta_rad)
                ax.plot(theta_deg, P11, color=colors[i], label=name, linewidth=1.5)
        ax.set_xlabel("Scattering angle (°)")
        ax.set_ylabel("P11")
        ax.set_title(f"Phase function @ {wl:.0f} nm")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Log scale
        ax = axes[1]
        for i, (ds, name) in enumerate(zip(datasets, labels)):
            if ds.P11 is not None:
                P11 = np.mean(ds.P11[i_wl], axis=1)
                theta_deg = np.degrees(ds.theta_rad)
                ax.semilogy(
                    theta_deg, np.maximum(P11, 1e-30), color=colors[i], label=name, linewidth=1.5
                )
        ax.set_xlabel("Scattering angle (°)")
        ax.set_ylabel("P11 (log)")
        ax.set_title(f"Phase function @ {wl:.0f} nm (log)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Phase Function Comparison @ {wl:.0f} nm", fontsize=13)
        fig.tight_layout()
        path = output_dir / f"mie_comparison_phase_{wl:.0f}nm.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Figure 4: Summary bar chart at 550nm ---
    # Find the wavelength index closest to 550nm
    ref_wl = 550.0
    i_550 = np.argmin(np.abs(datasets[0].wavelength_nm - ref_wl))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    bar_attrs = [
        ("C_ext", "Extinction (nm²)"),
        ("C_abs", "Absorption (nm²)"),
        ("SSA", "SSA"),
        ("g", "Asymmetry parameter"),
    ]

    x = np.arange(len(labels))
    width = 0.6

    for col, (attr, title) in enumerate(bar_attrs):
        ax = axes[col]
        vals = [getattr(ds, attr)[i_550] for ds in datasets]
        bars = ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_ylabel(title)
        ax.set_title(f"{title} @ {ref_wl:.0f} nm")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"Mie Approximation Summary @ {ref_wl:.0f} nm", fontsize=13)
    fig.tight_layout()
    path = output_dir / "mie_comparison_bar_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_visualizations(particle, soot, sulfate, args):
    """Generate 3D screenshots and rotation videos."""
    from Aerosol3D import (
        save_particle_voxel_screenshot,
        save_particle_voxel_video,
        save_rotation_video,
        save_screenshot,
    )

    print("\nGenerating 3D visualizations...")
    save_screenshot(
        particle,
        str(OUTPUT_DIR / "coated_aggregate.png"),
        colors={"aggregate": "black", "coating": "dodgerblue"},
        opacity={"aggregate": 0.9, "coating": 0.5},
    )
    save_particle_voxel_screenshot(
        particle,
        voxel_size=5.0,
        path=str(OUTPUT_DIR / "coated_aggregate_voxels.png"),
        colors={soot.id: "black", sulfate.id: "dodgerblue"},
        opacity={soot.id: 0.9, sulfate.id: 0.3},
    )

    if not args.no_video:
        save_rotation_video(
            particle,
            str(OUTPUT_DIR / "coated_rotation.mp4"),
            colors={"aggregate": "black", "coating": "dodgerblue"},
            opacity={"aggregate": 0.9, "coating": 0.5},
            n_frames=72,
            fps=24,
        )
        save_particle_voxel_video(
            particle,
            voxel_size=5.0,
            path=str(OUTPUT_DIR / "coated_rotation_voxels.mp4"),
            colors={soot.id: "black", sulfate.id: "dodgerblue"},
            opacity={soot.id: 0.9, sulfate.id: 0.3},
            n_frames=72,
            fps=24,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument(
        "--wavelengths",
        nargs="+",
        type=float,
        default=WAVELENGTHS,
        help=f"Wavelengths in nm (default: {WAVELENGTHS})",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Build particle ---
    particle, soot, sulfate = build_coated_particle()

    # --- 3D visualizations ---
    generate_visualizations(particle, soot, sulfate, args)

    # --- Mie approximation comparison ---
    wavelengths = args.wavelengths
    print(f"\nComputing Mie approximations at {wavelengths} nm...")
    datasets = compute_mie_approximations(particle, wavelengths)

    labels = [ds.solver for ds in datasets]
    print("\nGenerating comparison plots...")
    plot_comparison(datasets, labels, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
