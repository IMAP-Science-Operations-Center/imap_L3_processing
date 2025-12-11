import math
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spiceypy
from scipy.optimize import curve_fit
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray, std_devs, nominal_values

import imap_l3_processing
from imap_l3_processing.constants import METERS_PER_KILOMETER, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_proton_solar_wind_speed, \
    get_peak_indices, find_peak_center_of_mass_index, interpolate_energy, fit_energy_per_charge_peak_variations, \
    calculate_sw_speed_h_plus, get_proton_peak_indices, interpolate_angle, get_angle, \
    get_spin_angle_from_swapi_axis_in_despun_frame
from tests.spice_test_case import SpiceTestCase


class TestCalculateProtonSolarWindSpeed(SpiceTestCase):
    def test_calculate_solar_wind_speed(self):
        energies = np.array([[0, 1000, 750, 500, 400, 300, 200, 100, 50]] * 5)
        count_rates = np.array([[0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 15, 16, 17, 16, 15, 14, 0]])
        count_rates_with_uncertainties = uarray(count_rates, np.full_like(count_rates, 1.0))

        times = [datetime(2025, 6, 6, 12, i) for i in range(5)]
        epochs_in_terrestrial_time = spiceypy.datetime2et(times) * ONE_SECOND_IN_NANOSECONDS
        speed, a, phi, b = calculate_proton_solar_wind_speed(count_rates_with_uncertainties, energies,
                                                             epochs_in_terrestrial_time)

        proton_charge = 1.602176634e-19
        proton_mass = 1.67262192595e-27
        expected_speed = math.sqrt(2 * 400 * proton_charge / proton_mass) / METERS_PER_KILOMETER
        self.assertAlmostEqual(expected_speed, speed.n, 0)

    def test_calculate_solar_wind_speed_throws_exception_with_too_few_count_rates(self):
        energies = np.array([[0, 1000, 750, 500, 400, 300, 200, 100, 50]] * 5)
        count_rates = np.array([[0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 15, 16, 17, 16, 15, 14, 0],
                                [0, 14, 14, 14, 0, 0, 0, 0, 0],
                                [0, 14, 14, 14, 0, 0, 0, 0, 0]])
        count_rates_with_uncertainties = uarray(count_rates, np.full_like(count_rates, 1.0))

        times = [datetime(2025, 6, 6, 12, i) for i in range(5)]
        epochs_in_terrestrial_time = spiceypy.datetime2et(times) * ONE_SECOND_IN_NANOSECONDS
        with self.assertRaises(Exception):
            calculate_proton_solar_wind_speed(count_rates_with_uncertainties, energies,
                                              epochs_in_terrestrial_time)

    def test_calculate_solar_wind_speed_from_model_data(self):
        file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / 'imap_swapi_l2_fake-menlo-5-sweeps_20250606_v001.cdf'
        with CDF(str(file_path)) as cdf:
            epoch = cdf.raw_var("epoch")[...]
            energy = cdf["esa_energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]
        speed, a, phi, b = calculate_proton_solar_wind_speed(uarray(count_rate, count_rate_delta), energy,
                                                             epoch)

        self.assertAlmostEqual(speed.n, 497.930, 3)
        self.assertAlmostEqual(speed.s, 0.4615, 3)
        self.assertAlmostEqual(a.n, 32.033, 3)
        self.assertAlmostEqual(a.s, 3.4696, 2)
        self.assertAlmostEqual(phi.n, 10.6, 1)
        self.assertAlmostEqual(phi.s, 5.862, 2)
        self.assertAlmostEqual(b.n, 1294, 0)
        self.assertAlmostEqual(b.s, 2.4, 1)

    def test_get_peak_indices(self):
        test_cases = [
            ("one clear peak", [0, 0, 5, 10, 5, 0, 0], 2, [0, 5, 10, 5, 0]),
            ("narrow width", [0, 0, 5, 10, 5, 0, 0], 1, [5, 10, 5]),
            ("wide peak", [0, 0, 5, 10, 10, 2, 0, 0], 2, [0, 5, 10, 10, 2, 0]),
            ("At left edge", [5, 10, 5, 0, 0], 2, [5, 10, 5, 0]),
            ("At right edge", [0, 0, 5, 10, ], 2, [0, 5, 10]),
        ]

        for name, count_rates, width, expected_peak_values in test_cases:
            with self.subTest(name):
                peak_indices = get_peak_indices(count_rates, width)
                extracted_peak = count_rates[peak_indices]
                self.assertEqual(expected_peak_values, extracted_peak)

    def test_get_proton_peak_indices(self):
        test_cases = [
            ("one clear peak", [0, 0, 0, 1, 0, 5, 10, 5, 0, 0, 1, 1, 1], [0, 1, 0, 5, 10, 5, 0, 0, 1]),
            ("tied for peak", [0, 0, 0, 1, 0, 5, 10, 10, 0, 0, 1, 1, 1], [0, 1, 0, 5, 10, 10, 0, 0, 1, 1]),
        ]

        for name, count_rates, expected_peak_values in test_cases:
            with self.subTest(name):
                peak_indices = get_proton_peak_indices(count_rates)
                extracted_peak = count_rates[peak_indices]
                self.assertEqual(expected_peak_values, extracted_peak)

    def test_an_exception_is_raised_when_count_rates_contains_multiple_peaks(self):
        test_cases = [
            ("two clear peak", [0, 0, 5, 10, 5, 0, 0, 2, 10, 2, 0, 0]),
            ("peak is too wide", [0, 0, 5, 10, 10, 10, 2, 0, 0]),
        ]

        for name, count_rates in test_cases:
            with self.subTest(name):
                with self.assertRaises(Exception) as cm:
                    get_peak_indices(count_rates, 2)
                self.assertEqual(str(cm.exception), "Count rates contains multiple distinct peaks")

    def test_find_peak_center_of_mass_index(self):
        test_cases = [
            ("symmetrical peak", [0, 0, 5, 10, 5, 0, 0], slice(2, 5), 0, 3),
            ("asymmetrical peak", [0, 0, 5, 15, 0, 0, 0], slice(2, 5), 0, 2.75),
            ("excluding low rates", [0, 2, 5, 10, 5, 2, 1], slice(0, 7), 2, 3),
        ]
        for name, count_rates, peak_slice, minimum_count_rate_inclusive, expected_peak_index in test_cases:
            with self.subTest(name):
                self.assertEqual(expected_peak_index,
                                 find_peak_center_of_mass_index(peak_slice, count_rates, minimum_count_rate_inclusive))

    def test_find_peak_center_of_mass_index_filters_correctly_with_uncertain_count_rates(self):
        test_cases = [
            ("uses nominal values to filter", uarray([0, 2, 2, 2, 6, 0, 0], [1, 1, 1, 1, 1, 1, 1]), slice(0, 7), 2, 3),
        ]
        for name, count_rates, peak_slice, minimum_count_rate_inclusive, expected_peak_index in test_cases:
            with self.subTest(name):
                self.assertEqual(expected_peak_index,
                                 find_peak_center_of_mass_index(peak_slice, count_rates,
                                                                minimum_count_rate_inclusive).nominal_value)

    def test_uncertainty_calculation_in_find_peak_center_of_mass_index(self):
        test_cases = [
            ("with uncertainties", uarray([0, 0, 3, 5, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0]), slice(2, 5), 0, 2.625),
            ("with sqrt uncertainties", uarray([0, 0, 6, 600, 6, 0, 0], [0, 0, 6, 60, 6, 0, 0]), slice(2, 5), 0, 3),
        ]
        for name, count_rates, peak_slice, minimum_count_rate_inclusive, expected_peak_index in test_cases:
            with self.subTest(name):
                count_rate_std_devs = std_devs(count_rates)

                nominal_count_rates = nominal_values(count_rates)
                base_result = find_peak_center_of_mass_index(peak_slice, nominal_count_rates,
                                                             minimum_count_rate_inclusive)
                deviations_for_single_inputs = []
                for i in range(len(count_rate_std_devs)):
                    DELTA = 1e-8
                    modified_count_rates = nominal_count_rates.copy()
                    modified_count_rates[i] += DELTA
                    partial_derivative = (
                                                 find_peak_center_of_mass_index(peak_slice, modified_count_rates,
                                                                                minimum_count_rate_inclusive)
                                                 - base_result) / DELTA
                    deviations_for_single_inputs.append(count_rate_std_devs[i] * partial_derivative)
                final_uncertainty = math.sqrt(sum(n * n for n in deviations_for_single_inputs))
                result_with_uncertainty = find_peak_center_of_mass_index(peak_slice, count_rates,
                                                                         minimum_count_rate_inclusive)

                self.assertAlmostEqual(final_uncertainty, result_with_uncertainty.std_dev, 5)
                self.assertEqual(base_result, result_with_uncertainty.n)

    def test_find_peak_center_of_mass_index_raises_error_if_too_few_bins_remain(self):
        test_cases = [
            ("raises error", [0, 0, 5, 10, 1, 0, 0], slice(2, 5), 2, 3, True),
            ("no error", [0, 0, 5, 15, 1, 0, 0], slice(2, 5), 2, 2, False),
        ]
        for name, count_rates, peak_slice, minimum_count_rate_inclusive, minimum_bin_count, expect_error in test_cases:
            with self.subTest(name):
                if expect_error:
                    with self.assertRaises(Exception) as cm:
                        find_peak_center_of_mass_index(peak_slice, count_rates, minimum_count_rate_inclusive,
                                                       minimum_bin_count)
                else:
                    find_peak_center_of_mass_index(peak_slice, count_rates, minimum_count_rate_inclusive,
                                                   minimum_bin_count)

    def test_interpolates_to_find_energy_at_center_of_mass(self):
        test_cases = [
            ("Fractional Center of mass index", 1.5, [10, 100, 10000, 100000], 1000),
            ("Fractional Center of mass index 2", 2 / 3, [2, 16, 1024], 8),
            ("Whole number Center of mass index", 1, [3, 9, 27], 9),
        ]

        for name, center_of_mass_index, energies, expected_interpolated_energy in test_cases:
            with self.subTest(name):
                self.assertAlmostEqual(expected_interpolated_energy, interpolate_energy(center_of_mass_index, energies))

    def test_interpolates_to_find_angle_at_center_of_mass(self):
        test_cases = [
            ("interpolates", [110, 230, 350, 110], 1.5, 290),
            ("interpolates correctly across period", [110, 230, 350, 110], 2.5, 50)
        ]

        for name, angles, center_of_mass_index, expected_interpolated_angle in test_cases:
            with self.subTest(name):
                angle = interpolate_angle(center_of_mass_index, angles)

                self.assertAlmostEqual(angle, expected_interpolated_angle)

    def test_interpolates_with_uncertainty_to_find_energy_at_center_of_mass(self):
        test_cases = [
            ("Fractional Center of mass index", ufloat(1.5, 0.5), [10, 100, 10000, 100000], 1000),
        ]

        for name, center_of_mass_index, energies, expected_interpolated_energy in test_cases:
            with self.subTest(name):
                DELTA = 1e-8
                deriv = (interpolate_energy(1.5 + DELTA, energies) - interpolate_energy(1.5, energies)) / DELTA

                expected_uncertainty = center_of_mass_index.std_dev * deriv

                result = interpolate_energy(center_of_mass_index, energies)
                self.assertAlmostEqual(expected_interpolated_energy, result.n)
                self.assertAlmostEqual(expected_uncertainty, result.std_dev, 3)

    @patch('imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed.scipy.optimize.curve_fit')
    def test_curve_fit_is_initialized_correctly(self, mock_curve_fit):
        test_cases = [
            ([30, 60, 90, 120], uarray([1319, 1328, 1329, 1323], 1), 5, 180, 1324.75),
            ([30, 60, 90, 120], uarray([975, 956, 950, 957], 1), 12.5, 120, 959.5),
        ]

        mock_curve_fit.side_effect = curve_fit

        for angles, energies, expected_initial_a, expected_initial_phi, expected_initial_b in test_cases:
            with self.subTest():
                a, phi, b = fit_energy_per_charge_peak_variations(energies, angles)

                curve_fit_parameters = mock_curve_fit.call_args.kwargs

                self.assertEqual(expected_initial_a, curve_fit_parameters["p0"][0])
                self.assertEqual(expected_initial_phi, curve_fit_parameters["p0"][1])
                self.assertEqual(expected_initial_b, curve_fit_parameters["p0"][2])

                a_lower_bound, phi_lower_bound, b_lower_bound = curve_fit_parameters["bounds"][0]
                self.assertEqual(0, a_lower_bound)
                self.assertEqual(-np.inf, phi_lower_bound)
                self.assertEqual(0, b_lower_bound)

                a_upper_bound, phi_upper_bound, b_upper_bound = curve_fit_parameters["bounds"][1]
                self.assertEqual(np.inf, a_upper_bound)
                self.assertEqual(np.inf, phi_upper_bound)
                self.assertEqual(np.inf, b_upper_bound)

    def test_throws_error_when_reduced_chi_squared_greater_than_10(self):
        test_cases = [
            ("throws error", [30, 60, 90, 120], uarray([1319, 1103, 1110, 1323], 1), True, 13.512723754922165),
            ("doesn't throw error", [30, 60, 90, 120], uarray([975, 956, 950, 971], 1), False, 9.071796769724411),
        ]

        for name, angles, energies, error_flag, expected_chi_squared in test_cases:
            with self.subTest(name):
                try:
                    fit_energy_per_charge_peak_variations(energies, angles)
                    did_error = False
                except ValueError as e:
                    did_error = True
                    exception = e
                self.assertEqual(error_flag, did_error)
                if did_error:
                    self.assertEqual("Failed to fit - chi-squared too large", exception.args[0])
                    self.assertAlmostEqual(expected_chi_squared, exception.args[1])

    def test_curve_fit(self):
        test_cases = [
            (10, 500, 25, 120),
            (30, 1000, 270, 2)
        ]

        for a, b, phi, initial_angle in test_cases:
            with self.subTest():
                angles = [initial_angle - 72 * i for i in range(5)]
                centers_of_mass = [b + a * np.sin(np.deg2rad(phi - angle)) for angle in angles]
                center_of_mass_uncertainties = np.full_like(centers_of_mass, 1e-6)
                centers_of_mass = uarray(centers_of_mass, center_of_mass_uncertainties)
                result_a, result_phi, result_b = fit_energy_per_charge_peak_variations(centers_of_mass, angles)

                self.assertAlmostEqual(a, result_a.nominal_value)
                self.assertAlmostEqual(phi, result_phi.nominal_value)
                self.assertAlmostEqual(b, result_b.nominal_value)

    def test_converts_proton_energies_to_speeds(self):
        test_cases = [
            (1000, 437.6947142244463),
            (800, 391.48605375928245),
            (2000, 618.993801035228),
        ]

        for ev_q, expected_speed_km_s in test_cases:
            with self.subTest(ev_q):
                self.assertAlmostEqual(expected_speed_km_s, calculate_sw_speed_h_plus(ev_q))

    def test_converts_proton_energies_with_uncertainties(self):
        test_cases = [
            (ufloat(1000, 10), ufloat(437.6947142244463, 2.1884735711222315)),
            (ufloat(800, 100), ufloat(391.48605375928245, 24.46787835995515)),
            (ufloat(2000, 2), ufloat(618.993801035228, 0.30949690051761405)),
        ]

        for ev_q, expected_speed_km_s in test_cases:
            with self.subTest(ev_q):
                sw_speed = calculate_sw_speed_h_plus(ev_q)
                self.assertAlmostEqual(expected_speed_km_s.n, sw_speed.n)
                self.assertAlmostEqual(expected_speed_km_s.s, sw_speed.s)

    def test_calculate_sw_speed_on_array_with_uncertainties(self):
        energy = np.array([
            ufloat(1000.0, 10.0),
            ufloat(800.0, 100.0),
            ufloat(2000.0, 2.0)
        ], dtype=object)
        expected = [ufloat(437.6947142244463, 2.1884735711222315), ufloat(391.48605375928245, 24.46787835995515),
                    ufloat(618.993801035228, 0.30949690051761405)]
        np.testing.assert_array_equal(nominal_values(expected), nominal_values(calculate_sw_speed_h_plus(energy)))
        np.testing.assert_array_almost_equal(std_devs(expected), std_devs(calculate_sw_speed_h_plus(energy)))

    def test_calculate_sw_speed_on_array_without_uncertainties(self):
        energy = np.array([1000.0, 800.0, 2000.0], dtype=float)
        expected = [437.6947142244463, 391.48605375928245, 618.993801035228]
        result = calculate_sw_speed_h_plus(energy)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(float, result.dtype)
        np.sqrt(result)
        np.testing.assert_array_equal(expected, result)

    def test_calculate_sw_speed_on_2d_array(self):
        energy = uarray([[1000]], [[10]])
        expected = uarray([[437.6947142244463]], [[2.1884735711222315]])
        result = calculate_sw_speed_h_plus(energy)
        np.testing.assert_array_almost_equal(nominal_values(expected), nominal_values(result))
        np.testing.assert_array_almost_equal(std_devs(expected), std_devs(result))

    def test_calculate_sw_speed_on_empty_array(self):
        energy = []
        expected = []
        result = calculate_sw_speed_h_plus(energy)
        np.testing.assert_array_equal(expected, result)

    @patch('imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed.spiceypy.pxform')
    @patch('imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed.spiceypy.unitim')
    def test_get_angle(self, mock_unitim, mock_pxform):
        mock_pxform.return_value = [[1, 0, 0], [0, 0, -1], [0, -1, 0]]

        epoch = 123

        actual_angle = get_angle(epoch)

        mock_unitim.assert_called_with((epoch / ONE_SECOND_IN_NANOSECONDS), "TT", "ET")
        mock_pxform.assert_called_with("IMAP_SWAPI", "IMAP_DPS", mock_unitim.return_value)

        self.assertEqual(0, actual_angle)

    def test_get_spin_angle_from_swapi_axis_in_despun_frame(self):
        cases = [
            ([1, 0, 0], 270),
            ([0, 1, 0], 0),
            ([0, -1, 0], 180),
            ([-1, 0, 0], 90),
            ([-1, 0, 50], 90),
        ]
        for swapi_axis, expected_angle in cases:
            with self.subTest(swapi_axis):
                self.assertAlmostEqual(expected_angle,
                                       get_spin_angle_from_swapi_axis_in_despun_frame(np.array(swapi_axis)))
