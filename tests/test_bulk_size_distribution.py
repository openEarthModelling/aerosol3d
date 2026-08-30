"""Tests for Aerosol3D.bulk.datastructs.SizeDistribution."""

from __future__ import annotations

import numpy as np
import pytest


class TestSizeDistributionFactory:
    def test_factory_creates_pdf(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        assert sd.dist_type == "lognormal"
        assert sd.params == {"rg_nm": 100.0, "sigma_ln": 0.5}
        assert sd.r_min_nm == 1.0
        assert sd.r_max_nm == 1e5

    def test_gamma_factory_creates_pdf(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        sd = SizeDistribution.gamma(reff_nm=200.0, veff=0.1)
        assert sd.dist_type == "gamma"
        assert sd.params == {"reff_nm": 200.0, "veff": 0.1}


class TestSizeDistributionPDF:
    def test_pdf_integrates_to_one(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        r = np.logspace(0, 4, 2000)
        vals = sd.pdf_values(r)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        integral = _trapz(vals, r)
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_pdf_zero_at_non_positive(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        assert sd.pdf_values(np.array([-1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)


class TestSizeDistributionMoments:
    def test_moment_analytic_lognormal(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        rg = 100.0
        sigma = 0.5
        sd = SizeDistribution.lognormal(rg_nm=rg, sigma_ln=sigma)
        m3 = sd.moment(3.0)
        expected = rg**3.0 * np.exp(9.0 * sigma**2 / 2.0)
        assert m3 == pytest.approx(expected, rel=1e-6)

    def test_effective_radius_analytic(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        rg = 100.0
        sigma = 0.5
        sd = SizeDistribution.lognormal(rg_nm=rg, sigma_ln=sigma)
        reff = sd.effective_radius()
        expected = rg * np.exp(2.5 * sigma**2)
        assert reff == pytest.approx(expected, rel=1e-6)

    def test_gamma_moment_numerical(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        reff = 200.0
        veff = 0.1
        sd = SizeDistribution.gamma(reff_nm=reff, veff=veff)
        m3 = sd.moment(3.0)
        m2 = sd.moment(2.0)
        assert m3 > 0.0
        assert m2 > 0.0
        assert sd.effective_radius() == pytest.approx(m3 / m2, rel=1e-9)


class TestSizeDistributionNumberInInterval:
    def test_number_in_interval_sums_to_one(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        total = sd.number_in_interval(0.0, np.inf)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_number_in_interval_partial(self):
        from Aerosol3D.bulk.datastructs import SizeDistribution

        sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.5)
        left = sd.number_in_interval(0.0, 100.0)
        right = sd.number_in_interval(100.0, np.inf)
        assert left + right == pytest.approx(1.0, abs=1e-6)
