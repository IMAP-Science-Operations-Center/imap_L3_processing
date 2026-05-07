"""Tests for `uncertainties.py`.

All tests are pure math — no SPICE, no calibration files.
`derive_velocity_angles` calls `rotate_rtn_to_dps` which uses SPICE, so those
tests use SpiceTestCase. The remaining helpers are free of external deps.
"""

import unittest

import numpy as np
from uncertainties import ufloat
from uncertainties import covariance_matrix

from imap_l3_processing.swapi.l3a.science.solar_wind.proton.uncertainties import (
    derive_velocity_angles,
    make_correlated_velocity,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    SolarWindParams,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    OptimizeSolarWindParamsResult,
)
from tests.spice_test_case import SpiceTestCase


def _make_result(residuals, jacobian, density=5.0, temperature=1e5, velocity=None):
    """Build a minimal OptimizeSolarWindParamsResult for testing."""
    if velocity is None:
        velocity = np.array([400.0, 0.0, 0.0])
    from imap_l3_processing.constants import PROTON_MASS_KG

    sw = SolarWindParams(
        density=density,
        bulk_velocity_rtn=velocity,
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )
    return OptimizeSolarWindParamsResult(
        sw_params=sw,
        residuals=np.asarray(residuals, dtype=float),
        jacobian=np.asarray(jacobian, dtype=float),
        success=True,
    )


class TestMakeCorrelatedVelocityHappyPath(unittest.TestCase):
    """make_correlated_velocity with a positive-definite covariance returns
    correlated UFloat components whose covariance matrix matches the input."""

    def test_psd_covariance_round_trips(self):
        """Round-trip: build correlated velocity and recover covariance from uncertainties."""
        nominal = np.array([400.0, 10.0, -5.0])
        cov = np.array(
            [
                [100.0, 20.0, -10.0],
                [20.0, 50.0, 5.0],
                [-10.0, 5.0, 25.0],
            ]
        )
        vr, vt, vn = make_correlated_velocity(nominal, cov)

        # Nominals should match.
        np.testing.assert_allclose(
            [vr.nominal_value, vt.nominal_value, vn.nominal_value],
            nominal,
            rtol=1e-12,
        )

        # Recovered covariance should match input within float precision.
        recovered = np.array(covariance_matrix([vr, vt, vn]))
        np.testing.assert_allclose(recovered, cov, rtol=1e-10, atol=1e-10)


class TestMakeCorrelatedVelocityNonPSD(unittest.TestCase):
    """make_correlated_velocity with a non-PSD covariance falls back to ufloats
    whose std_dev is NaN, preserving the nominal values."""

    def test_non_psd_covariance_returns_nan_std_dev(self):
        """A matrix with a negative eigenvalue → NaN std_devs, finite nominals."""
        nominal = np.array([400.0, 20.0, -5.0])
        # Covariance with off-diagonal 2 → eigenvalues approximately {-1, 3, 1};
        # the minimum eigenvalue is negative.
        non_psd_cov = np.array(
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        vr, vt, vn = make_correlated_velocity(nominal, non_psd_cov)

        # Nominals are preserved.
        np.testing.assert_allclose(
            [vr.nominal_value, vt.nominal_value, vn.nominal_value],
            nominal,
            rtol=1e-12,
        )

        # std_devs are NaN.
        for v in (vr, vt, vn):
            self.assertTrue(
                np.isnan(v.std_dev), msg=f"expected NaN std_dev, got {v.std_dev}"
            )

    def test_non_finite_covariance_returns_nan_std_dev(self):
        """inf or NaN anywhere in the covariance → NaN std_devs (non-finite check)."""
        nominal = np.array([400.0, 0.0, 0.0])
        bad_cov = np.array([[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        vr, vt, vn = make_correlated_velocity(nominal, bad_cov)
        for v in (vr, vt, vn):
            self.assertTrue(np.isnan(v.std_dev))


class TestDeriveVelocityAngles(SpiceTestCase):
    """derive_velocity_angles happy-path and NaN-cov fallback.

    These tests require SPICE because the function calls rotate_rtn_to_dps.
    The angle assertions use approximate answers — exact values depend on the
    SPICE frame transform at the test epoch, which we hold constant via SpiceTestCase.
    """

    # A fixed TT2000 epoch within the furnished SPICE kernels.
    _EPOCH_TT2000 = 808902069184000000  # 2025-08-26T12:01:09 TT2000 ns

    def test_finite_result_for_nominal_velocity(self):
        """Happy path: radial solar wind → finite speed, clock angle, deflection angle."""
        nominal = np.array([400.0, 0.0, 0.0])
        cov = np.diag([100.0, 25.0, 25.0])
        vr, vt, vn = make_correlated_velocity(nominal, cov)
        velocity = (vr, vt, vn)

        speed, clock, deflection = derive_velocity_angles(velocity, self._EPOCH_TT2000)

        # Speed magnitude must be close to 400 km/s (rotation preserves length).
        self.assertAlmostEqual(speed.nominal_value, 400.0, delta=1.0)

        # All three outputs must have finite nominals and positive std_devs.
        for name, qty in [
            ("speed", speed),
            ("clock", clock),
            ("deflection", deflection),
        ]:
            with self.subTest(name=name):
                self.assertTrue(
                    np.isfinite(qty.nominal_value), f"{name} nominal is not finite"
                )
                self.assertGreater(qty.std_dev, 0.0, f"{name} std_dev is not positive")

    def test_nan_cov_velocity_yields_nan_std_devs(self):
        """When velocity components carry NaN std_devs, output angles also have NaN std_devs."""
        # Build velocity with NaN std_devs.
        vr = ufloat(400.0, np.nan)
        vt = ufloat(0.0, np.nan)
        vn = ufloat(0.0, np.nan)

        speed, clock, deflection = derive_velocity_angles(
            (vr, vt, vn), self._EPOCH_TT2000
        )

        for name, qty in [
            ("speed", speed),
            ("clock", clock),
            ("deflection", deflection),
        ]:
            with self.subTest(name=name):
                self.assertTrue(
                    np.isnan(qty.std_dev),
                    f"{name} std_dev should be NaN, got {qty.std_dev}",
                )


if __name__ == "__main__":
    unittest.main()
