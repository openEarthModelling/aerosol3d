"""Example: Uncoated black carbon sphere — 3D visualization and DDA optical computation.

Usage:
    python black_carbon_sphere.py --no-optics
    python black_carbon_sphere.py
    python black_carbon_sphere.py --save
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

    # --- Step 1: Create bare BC sphere ---
    from Aerosol3D import (
        AerosolParticle,
        MixingState,
        create_sphere,
        preset_material,
        save_screenshot,
    )

    soot = preset_material("black_carbon")
    radius_nm = 50.0

    particle = AerosolParticle(
        name="bare_bc_sphere",
        mixing_state=MixingState.INTERNAL,
        unit="nm",
    )
    particle.add_mesh("core", create_sphere((0, 0, 0), radius_nm), soot)

    print(f"Particle: {particle}")
    print(f"Material: {soot}")

    # --- Step 2: 3D visualization ---
    save_screenshot(
        particle,
        str(OUTPUT_DIR / "bc_sphere_3d.png"),
        colors={"core": "black"},
        opacity={"core": 0.9},
    )
    print(f"3D screenshot saved: {OUTPUT_DIR / 'bc_sphere_3d.png'}")

    # --- Step 3: DDA optical computation ---
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
        plot_phase_function_2d(
            result,
            save_path=str(OUTPUT_DIR / "bc_phase_function.png"),
            plane="xz",
            log_scale=True,
        )
        print(f"Phase function saved: {OUTPUT_DIR / 'bc_phase_function.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
