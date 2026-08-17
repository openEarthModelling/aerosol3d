"""Example: Black carbon fractal aggregate — full pipeline.

Demonstrates the complete Aerosol3D workflow with pyFracAggregate:
1. Generate a BC fractal aggregate using pyFracAggregate
2. Convert to AerosolParticle via from_fractal()
3. 3D screenshot and rotation video
4. DDA optical computation

Usage:
    # Visualization only:
    python black_carbon_fractal.py --no-optics

    # Full pipeline:
    python black_carbon_fractal.py

    # Save all outputs:
    python black_carbon_fractal.py --save
"""

import argparse
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-optics", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Generate fractal aggregate ---
    from pyFracAggregate import Monodisperse
    from pyFracAggregate import generate as gen_fractal

    print("Generating BC fractal aggregate (50 monomers, PCA, Df=1.8)...")
    agg = gen_fractal(
        n_particles=50,
        df=1.8,
        kf=1.2,
        method="pca",
        particle_dist=Monodisperse(25.0),
    )
    print(f"  Generated: {agg.current_size} monomers, unit={agg.length_unit}")

    # --- Step 2: Convert to AerosolParticle ---
    from Aerosol3D import from_fractal, preset_material, save_rotation_video, save_screenshot

    soot = preset_material("black_carbon")
    fractal = from_fractal(agg, soot)
    particle = fractal.to_particle()
    print(f"  Particle: {particle}")

    # --- Step 3: 3D visualization ---
    save_screenshot(
        particle,
        str(OUTPUT_DIR / "bc_fractal_3d.png"),
        colors={"aggregate": "black"},
        opacity={"aggregate": 0.9},
    )
    print(f"  Screenshot saved: {OUTPUT_DIR / 'bc_fractal_3d.png'}")

    save_rotation_video(
        particle,
        str(OUTPUT_DIR / "bc_fractal_rotation.mp4"),
        colors={"aggregate": "black"},
        opacity={"aggregate": 0.9},
        n_frames=72,
        fps=24,
    )
    print(f"  Rotation video saved: {OUTPUT_DIR / 'bc_fractal_rotation.mp4'}")

    # --- Step 4: DDA optical computation ---
    if args.no_optics:
        print("Optical computation skipped (--no-optics).")
        return

    from Aerosol3D import SimulationConfig, solve_optics
    from Aerosol3D.optics.visualization import plot_phase_function_2d, print_macroscopic

    config = SimulationConfig(
        wavelength=550.0,
        source="solar",
    )

    print("\nRunning DDA solve ...")
    result = solve_optics(
        particle,
        config,
        compute_near_field=True,
        compute_phase_func=True,
    )
    print_macroscopic(result)

    if args.save and result.phase_function is not None:
        phase_path = str(OUTPUT_DIR / "bc_fractal_phase.png")
        plot_phase_function_2d(result, save_path=phase_path, plane="xz", log_scale=True)
        print(f"Phase function saved: {phase_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
