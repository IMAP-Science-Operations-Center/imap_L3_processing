import math
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray, std_devs, nominal_values

import imap_processing
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_proton_solar_wind_speed, \
    get_peak_indices, find_peak_center_of_mass_index, interpolate_energy, fit_energy_per_charge_peak_variations, \
    calculate_sw_speed_h_plus, get_proton_peak_indices


class TestCalculateProtonSolarWindSpeed(TestCase):
    def test_calculate_solar_wind_speed(self):
        spin_angles = np.array([[0, 4, 8, 12], [16, 20, 24, 28], [32, 36, 40, 44], [48, 52, 56, 60], [64, 68, 72, 76]])
        energies = np.array([0, 1000, 750, 500])
        count_rates = np.array([[0, 0, 10, 0], [0, 0, 10, 0], [0, 0, 10, 0], [0, 0, 10, 0], [0, 0, 10, 0]])
        count_rates_with_uncertainties = uarray(count_rates, np.full_like(count_rates, 1.0))

        speed, a, phi, b = calculate_proton_solar_wind_speed(count_rates_with_uncertainties, spin_angles, energies,
                                                             [0, 0, 0, 0, 0])

        proton_charge = 1.602176634e-19
        proton_mass = 1.67262192595e-27
        expected_speed = math.sqrt(2 * 750 * proton_charge / proton_mass)
        self.assertAlmostEqual(speed.n, expected_speed, 0)

    def test_calculate_solar_wind_speed_from_model_data(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            epoch = cdf.raw_var("epoch")[...]
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            spin_angles = cdf["spin_angles"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        speed, a, phi, b = calculate_proton_solar_wind_speed(uarray(count_rate, count_rate_delta), spin_angles, energy,
                                                             epoch)

        self.assertAlmostEqual(speed.n, 497919, 0)
        self.assertAlmostEqual(speed.s, 464, 0)
        self.assertAlmostEqual(a.n, 32.234, 3)
        self.assertAlmostEqual(a.s, 3.48, 2)
        self.assertAlmostEqual(phi.n, 252.2, 1)
        self.assertAlmostEqual(phi.s, 5.85, 2)
        self.assertAlmostEqual(b.n, 1294, 0)
        self.assertAlmostEqual(b.s, 2.4, 1)

    def test_get_peak_indices(self):
        test_cases = [
            ("one clear peak", [0, 0, 5, 10, 5, 0, 0], 2, [0, 5, 10, 5, 0]),
            ("narrow width", [0, 0, 5, 10, 5, 0, 0], 1, [5, 10, 5]),
            ("wide peak", [0, 0, 5, 10, 10, 2, 0, 0], 2, [0, 5, 10, 10, 2, 0]),
        ]

        for name, count_rates, width, expected_peak_values in test_cases:
            with self.subTest(name):
                peak_indices = get_peak_indices(count_rates, width)
                extracted_peak = count_rates[peak_indices]
                self.assertEqual(expected_peak_values, extracted_peak)


    def test_get_proton_peak_indices(self):
        test_cases = [
            ("one clear peak", [0,0,0, 1, 0, 5, 10, 5, 0, 0, 1, 1, 1], [0, 1, 0, 5, 10, 5, 0, 0, 1]),
            ("tied for peak", [0,0,0, 1, 0, 5, 10, 10, 0, 0, 1, 1, 1], [0, 1, 0, 5, 10, 10, 0, 0, 1, 1]),
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
            ("symmetrical peak", [0, 0, 5, 10, 5, 0, 0], slice(2, 5), 3),
            ("asymmetrical peak", [0, 0, 5, 15, 0, 0, 0], slice(2, 5), 2.75),
        ]
        for name, count_rates, peak_slice, expected_peak_index in test_cases:
            with self.subTest(name):
                self.assertEqual(expected_peak_index,
                                 find_peak_center_of_mass_index(peak_slice, count_rates))

    def test_uncertainty_calculation_in_find_peak_center_of_mass_index(self):
        test_cases = [
            ("with uncertainties", uarray([0, 0, 3, 5, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0]), slice(2, 5), 2.625),
            ("with sqrt uncertainties", uarray([0, 0, 6, 600, 6, 0, 0], [0, 0, 6, 60, 6, 0, 0]), slice(2, 5), 3),

        ]
        for name, count_rates, peak_slice, expected_peak_index in test_cases:
            with self.subTest(name):
                count_rate_std_devs = std_devs(count_rates)

                nominal_count_rates = nominal_values(count_rates)
                base_result = find_peak_center_of_mass_index(peak_slice, nominal_count_rates)
                deviations_for_single_inputs = []
                for i in range(len(count_rate_std_devs)):
                    DELTA = 1e-8
                    modified_count_rates = nominal_count_rates.copy()
                    modified_count_rates[i] += DELTA
                    partial_derivative = (
                                                 find_peak_center_of_mass_index(peak_slice, modified_count_rates)
                                                 - base_result) / DELTA
                    deviations_for_single_inputs.append(count_rate_std_devs[i] * partial_derivative)
                final_uncertainty = math.sqrt(sum(n * n for n in deviations_for_single_inputs))
                result_with_uncertainty = find_peak_center_of_mass_index(peak_slice, count_rates)

                self.assertAlmostEqual(final_uncertainty, result_with_uncertainty.std_dev, 5)
                self.assertEqual(base_result, result_with_uncertainty.n)

    def test_interpolates_to_find_energy_at_center_of_mass(self):
        test_cases = [
            ("Fractional Center of mass index", 1.5, [10, 100, 10000, 100000], 1000),
            ("Fractional Center of mass index 2", 2 / 3, [2, 16, 1024], 8),
            ("Whole number Center of mass index", 1, [3, 9, 27], 9),
        ]

        for name, center_of_mass_index, energies, expected_interpolated_energy in test_cases:
            with self.subTest(name):
                self.assertAlmostEqual(expected_interpolated_energy, interpolate_energy(center_of_mass_index, energies))

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

    def test_curve_fit(self):
        test_cases = [
            (10, 500, 25, 120),
            (30, 1000, 270, 2),
        ]

        for a, b, phi, initial_angle in test_cases:
            with self.subTest():
                angles = [initial_angle - 72 * i for i in range(5)]
                centers_of_mass = [b + a * np.sin(np.deg2rad(phi + angle)) for angle in angles]
                center_of_mass_uncertainties = np.full_like(centers_of_mass, 1e-6)
                centers_of_mass = uarray(centers_of_mass, center_of_mass_uncertainties)
                result_a, result_phi, result_b = fit_energy_per_charge_peak_variations(centers_of_mass, angles)

                self.assertAlmostEqual(a, result_a.nominal_value)
                self.assertAlmostEqual(phi, result_phi.nominal_value)
                self.assertAlmostEqual(b, result_b.nominal_value)

    def test_converts_proton_energies_to_speeds(self):
        test_cases = [
            (1000, 437694.7142244463),
            (800, 391486.05375928245),
            (2000, 618993.801035228),
        ]

        for ev_q, expected_speed_m_s in test_cases:
            with self.subTest(ev_q):
                self.assertAlmostEqual(expected_speed_m_s, calculate_sw_speed_h_plus(ev_q))

    def test_converts_proton_energies_with_uncertainties(self):
        test_cases = [
            (ufloat(1000, 10), ufloat(437694.7142244463, 2188.4735711222315)),
            (ufloat(800, 100), ufloat(391486.05375928245, 24467.87835995515)),
            (ufloat(2000, 2), ufloat(618993.801035228, 309.49690051761405)),
        ]

        for ev_q, expected_speed_m_s in test_cases:
            with self.subTest(ev_q):
                sw_speed = calculate_sw_speed_h_plus(ev_q)
                self.assertAlmostEqual(expected_speed_m_s.n, sw_speed.n)
                self.assertAlmostEqual(expected_speed_m_s.s, sw_speed.s)
