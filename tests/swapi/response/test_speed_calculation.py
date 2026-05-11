import unittest

import numpy as np
from uncertainties import UFloat, ufloat

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.response.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_DISCARDED_BIN,
    SWAPI_FINE_SWEEP_BINS,
    SWAPI_K_FACTOR,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
    calculate_sw_speed,
    calculate_sw_speed_h_plus,
    esa_voltage_to_proton_speed,
)

# Independent reference values for v = sqrt(2 · K · e · V / m_p) at K = 1.89,
# computed with astropy units + CODATA constants.
# Astropy used CODATA 2018 m_p (1.67262192369e-27 kg) vs the repo's CODATA 2022
# value (1.67262192595e-27 kg); the resulting ~7e-10 relative drift sets rtol below.
_REFERENCE_PROTON_SPEEDS_KM_PER_S = {
    100.0: 190.283970237818,
    500.0: 425.487892480309,
    1000.0: 601.730748171198,
    4000.0: 1203.4614963424,
}


class TestConstants(unittest.TestCase):
    """Just because they're so subtly tricky..."""

    def test_discarded_bin_is_zero(self):
        self.assertEqual(SWAPI_DISCARDED_BIN, 0)

    def test_science_bins_excludes_discarded_bin(self):
        self.assertEqual(SWAPI_SCIENCE_BINS, slice(1, 72))

    def test_coarse_and_fine_partition_science_bins(self):
        # Coarse and fine bin slices must tile the science range with no overlap and no gap.
        self.assertEqual(SWAPI_COARSE_SWEEP_BINS.start, SWAPI_SCIENCE_BINS.start)
        self.assertEqual(SWAPI_COARSE_SWEEP_BINS.stop, SWAPI_FINE_SWEEP_BINS.start)
        self.assertEqual(SWAPI_FINE_SWEEP_BINS.stop, SWAPI_SCIENCE_BINS.stop)

    def test_coarse_sweep_has_62_bins_and_fine_has_9(self):
        self.assertEqual(
            SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start, 62
        )
        self.assertEqual(SWAPI_FINE_SWEEP_BINS.stop - SWAPI_FINE_SWEEP_BINS.start, 9)

    def test_simion_k_factor(self):
        self.assertAlmostEqual(SWAPI_K_FACTOR, 1.89)

    def test_l2_label_k_factor_differs_from_simion(self):
        # L2 esa_energy = SWAPI_L2_K_FACTOR × |V|, divided out by L3 to recover voltage.
        self.assertAlmostEqual(SWAPI_L2_K_FACTOR, 1.93)
        self.assertNotAlmostEqual(SWAPI_K_FACTOR, SWAPI_L2_K_FACTOR)


class TestEsaVoltageToProtonSpeed(unittest.TestCase):
    def test_matches_independent_reference_for_typical_solar_wind(self):
        # 1000 V → ~602 km/s for protons in the SWAPI ESA. Reference values were
        # computed independently with astropy units; the 1e-8 tolerance absorbs
        # the CODATA 2018 vs 2022 m_p drift between astropy and this repo.
        for V, expected_km_per_s in _REFERENCE_PROTON_SPEEDS_KM_PER_S.items():
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_proton_speed(V),
                    expected_km_per_s,
                    rtol=1e-8,
                )

    def test_handles_negative_voltage_via_absolute_value(self):
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(-2500.0),
            esa_voltage_to_proton_speed(2500.0),
        )

    def test_array_input_returns_elementwise_speeds(self):
        voltages = np.array([100.0, 1000.0, 4000.0])
        expected = np.array(
            [_REFERENCE_PROTON_SPEEDS_KM_PER_S[v] for v in voltages.tolist()]
        )
        result = esa_voltage_to_proton_speed(voltages)
        np.testing.assert_allclose(result, expected, rtol=1e-8)


class TestCalculateSwSpeed(unittest.TestCase):
    def test_2d_array_matches_analytic_formula_per_element(self):
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        expected = (
            np.sqrt(2 * E * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        np.testing.assert_allclose(result, expected)

    def test_2d_array_input_preserves_shape(self):
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertEqual(result.shape, E.shape)

    def test_empty_array_input_returns_empty_array(self):
        result = calculate_sw_speed(
            PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, np.array([])
        )
        self.assertEqual(result.size, 0)

    def test_ufloat_scalar_propagates_uncertainty(self):
        E = ufloat(1.0e-16, 1.0e-18)
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertIsInstance(result, UFloat)
        # σ_v = v · σ_E / (2 E)
        expected_nom = (
            np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        expected_sigma = expected_nom * E.std_dev / (2 * E.nominal_value)
        self.assertAlmostEqual(result.nominal_value, expected_nom)
        self.assertAlmostEqual(result.std_dev, expected_sigma)

    def test_ufloat_array_input_propagates_uncertainty_per_element(self):
        # Array of UFloat scalars takes the `unumpy.sqrt` branch — different
        # code path from the float-array branch above. Each element should
        # propagate its own σ_E.
        E_values = np.array([ufloat(1.0e-16, 1.0e-18), ufloat(4.0e-16, 2.0e-18)])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E_values)
        self.assertEqual(result.shape, E_values.shape)
        for r, E in zip(result, E_values):
            self.assertIsInstance(r, UFloat)
            expected_nom = (
                np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
                / METERS_PER_KILOMETER
            )
            self.assertAlmostEqual(r.nominal_value, expected_nom)
            self.assertAlmostEqual(
                r.std_dev, expected_nom * E.std_dev / (2 * E.nominal_value)
            )


class TestCalculateSwSpeedHPlus(unittest.TestCase):
    def test_consistent_with_calculate_sw_speed_for_protons(self):
        E = np.array([1.0e-16, 4.0e-16])
        np.testing.assert_allclose(
            calculate_sw_speed_h_plus(E),
            calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E),
        )


if __name__ == "__main__":
    unittest.main()
