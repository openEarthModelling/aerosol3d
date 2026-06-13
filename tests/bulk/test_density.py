# tests/bulk/test_density.py
import numpy as np
import pytest
from scipy.interpolate import interp1d

from Aerosol3D.bulk.datastructs import SizeDistribution
from Aerosol3D.bulk.density import compute_effective_density
from Aerosol3D.bulk.integration import integrate_distribution


def test_single_material_returns_material_density():
    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
    radii = np.array([50.0, 100.0, 200.0])
    dens = np.array([1800.0, 1800.0, 1800.0])
    assert compute_effective_density(radii, dens, sd) == pytest.approx(1800.0)


def test_varying_density_matches_manual_volume_weighting():
    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
    radii = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    dens = np.array([1000.0, 1500.0, 2000.0, 2400.0, 2600.0])
    rho = interp1d(radii, dens, kind="linear", fill_value="extrapolate")
    num = integrate_distribution(lambda r: rho(r) * r**3, sd)
    den = sd.moment(3)
    expected = num / den
    assert compute_effective_density(radii, dens, sd) == pytest.approx(expected)
    # sanity: strictly between min and max input density
    rho_eff = compute_effective_density(radii, dens, sd)
    assert 1000.0 < rho_eff < 2600.0


def test_fixed_quad_method_matches_quad():
    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
    radii = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    dens = np.array([1000.0, 1500.0, 2000.0, 2400.0, 2600.0])
    quad = compute_effective_density(radii, dens, sd, method="quad")
    fixed = compute_effective_density(radii, dens, sd, method="fixed_quad", n_quad=128)
    assert fixed == pytest.approx(quad, rel=2e-2)


def test_nonpositive_moment_raises():
    # Subclass overrides moment(3) to return a negative value, tripping the
    # ``den <= 0`` guard without mutating density.py or patching the instance.
    # (monkeypatch.setattr failed at teardown due to an editable-install
    # isinstance-mismatch when restoring the attribute.)
    class NegativeMomentSD(SizeDistribution):
        def moment(self, order: float, method: str = "quad") -> float:
            return -1.0

    sd = NegativeMomentSD.lognormal(rg_nm=100.0, sigma_ln=0.5)
    radii = np.array([50.0, 100.0])  # varying -> skips constant fast path
    dens = np.array([1000.0, 2600.0])
    with pytest.raises(ValueError, match="non-positive"):
        compute_effective_density(radii, dens, sd)
