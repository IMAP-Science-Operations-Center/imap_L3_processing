import unittest

import numpy as np

from imap_l3_processing.swapi.response.deadtime import (
    SWAPI_DEADTIME_S,
    deadtime_factor,
)


class TestDeadtimeFactor(unittest.TestCase):
    def test_zero_rate_produces_unit_factor(self):
        self.assertEqual(deadtime_factor(0.0), 1.0)

    def test_factor_is_bounded_in_unit_interval(self):
        # For physical (non-negative) rates the factor must lie in (0, 1].
        for r in [0.0, 1e3, 1e6, 1e9, 1e12]:
            f = deadtime_factor(float(r))
            self.assertGreater(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_factor_is_monotonically_decreasing(self):
        # Higher count rates lose more events to deadtime — the factor is a
        # strictly decreasing function of rate. Sample across six decades to
        # catch any sign or denominator typo.
        rates = np.logspace(0, 9, 10)
        factors = np.array([deadtime_factor(float(r)) for r in rates])
        self.assertTrue(np.all(np.diff(factors) < 0))

    def test_recovers_documentation_example(self):
        true_rate_hz = 35_226
        measured_rate_hz = true_rate_hz * deadtime_factor(true_rate_hz)
        np.testing.assert_allclose(measured_rate_hz, 35_000, rtol=1e-4)

    def test_special_rate_matches_closed_form(self):
        # At R = 1 / tau, the factor must equal exactly 0.5 (formula 1/(1+1)).
        rate = 1.0 / SWAPI_DEADTIME_S
        np.testing.assert_allclose(deadtime_factor(rate), 0.5, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
