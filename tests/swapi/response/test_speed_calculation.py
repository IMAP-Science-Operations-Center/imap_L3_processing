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
    calculate_combined_sweeps,
    calculate_sw_speed,
    calculate_sw_speed_h_plus,
    esa_voltage_to_proton_speed,
    extract_coarse_sweep,
    times_for_sweep,
)


def _analytic_speed_km_per_s(
    voltage: float, mass_kg: float, charge_c: float
) -> float:
    return float(
        np.sqrt(2 * SWAPI_K_FACTOR * charge_c * abs(voltage) / mass_kg)
        / METERS_PER_KILOMETER
    )


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
    def test_matches_analytical_formula_for_typical_solar_wind(self):
        # 1000 V → ~570 km/s for protons in the SWAPI ESA.
        for V in [100.0, 500.0, 1000.0, 4000.0]:
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_proton_speed(V),
                    _analytic_speed_km_per_s(V, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS),
                    rtol=1e-12,
                )

    def test_handles_negative_voltage_via_absolute_value(self):
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(-2500.0),
            esa_voltage_to_proton_speed(2500.0),
        )

    def test_array_input_returns_elementwise_speeds(self):
        voltages = np.array([100.0, 1000.0, 4000.0])
        result = esa_voltage_to_proton_speed(voltages)
        expected = np.array(
            [
                _analytic_speed_km_per_s(v, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS)
                for v in voltages
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestTimesForSweep(unittest.TestCase):
    def test_returns_72_entries_with_uniform_step(self):
        start = 1234.5
        times = times_for_sweep(start)
        self.assertEqual(len(times), 72)
        # Each bin is 12 s / 72 = 1/6 s wide.
        np.testing.assert_allclose(times[0], start)
        np.testing.assert_allclose(np.diff(times), 1.0 / 6.0, rtol=1e-12)

    def test_total_sweep_duration_just_under_12_s(self):
        times = times_for_sweep(0.0)
        self.assertAlmostEqual(times[-1] - times[0], 71.0 / 6.0)


class TestExtractCoarseSweep(unittest.TestCase):
    def test_2d_input_returns_columns_1_to_63(self):
        data = np.arange(72 * 5, dtype=float).reshape(5, 72)
        result = extract_coarse_sweep(data)
        np.testing.assert_array_equal(result, data[:, 1:63])
        self.assertEqual(result.shape, (5, 62))

    def test_1d_input_returns_indices_1_to_63(self):
        data = np.arange(72, dtype=float)
        result = extract_coarse_sweep(data)
        np.testing.assert_array_equal(result, data[1:63])
        self.assertEqual(result.shape, (62,))


class TestCalculateCombinedSweeps(unittest.TestCase):
    def test_returns_average_count_rate_and_mean_energy_over_coarse_bins(self):
        n_sweeps, n_bins = 5, 72
        rng = np.random.default_rng(seed=0)
        rates = rng.uniform(50.0, 200.0, size=(n_sweeps, n_bins))
        per_sweep_energy = np.logspace(np.log10(20_000.0), np.log10(50.0), n_bins)
        # Per-sweep energies vary slightly so the column mean is a meaningful test.
        energies = per_sweep_energy * (
            1.0 + 0.001 * rng.standard_normal((n_sweeps, n_bins))
        )

        avg_rate, avg_energy = calculate_combined_sweeps(rates, energies)

        self.assertEqual(avg_rate.shape, (62,))
        self.assertEqual(avg_energy.shape, (62,))
        np.testing.assert_allclose(avg_rate, rates[:, 1:63].mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(
            avg_energy, energies[:, 1:63].mean(axis=0), rtol=1e-12
        )


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
