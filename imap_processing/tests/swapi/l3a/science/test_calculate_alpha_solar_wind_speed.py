from pathlib import Path
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray

import imap_processing
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed, \
    calculate_alpha_center_of_mass, get_alpha_peak_indices, calculate_sw_speed_alpha


class TestCalculateAlphaSolarWindSpeed(TestCase):
    def test_get_alpha_peak_indices(self):
        test_cases = [
            ("one clear peak", [0, 2, 3, 2, 0, 0, 0, 5, 10, 5, 0], [0, 2, 3, 2, 0]),
            ("wide peak", [0, 2, 3, 3, 2, 0, 0, 0, 5, 10, 5, 0], [0, 2, 3, 3, 2, 0]),
        ]

        for name, count_rates, expected_peak_values in test_cases:
            with self.subTest(name):
                peak_indices = get_alpha_peak_indices(count_rates)
                extracted_peak = count_rates[peak_indices]
                self.assertEqual(expected_peak_values, extracted_peak)

    def test_an_exception_is_raised_when_no_alpha_peak(self):
        test_cases = [
            ("no clear peak", [0, 0, 1, 1, 1, 2,2 , 2, 10, 2, 0, 0], "Alpha peak not found"),
            ("peak is too wide", [0, 4, 4, 4, 0, 0, 0, 0, 0, 5, 10, 10, 3, 2, 0, 0], "Count rates contains multiple distinct peaks"),
        ]

        for name, count_rates, exception_text in test_cases:
            with self.subTest(name):
                with self.assertRaises(Exception) as cm:
                    get_alpha_peak_indices(count_rates)
                self.assertEqual(str(cm.exception), exception_text)

    def test_convert_energy_to_alpha_solar_wind_speed(self):
        test_cases = [
            (ufloat(1000, 200), 310562, 31056),
            (ufloat(1300, 10), 354096, 1362),
            (ufloat(1800, 5), 416663, 579),
            (ufloat(2500, 600), 491042, 58925),
            (ufloat(5000, 25), 694439, 1736)
        ]

        for energy, expected_speed, expected_uncertainty in test_cases:
            with self.subTest(f"converting energy of {energy} to speed"):
                alpha_sw_speed = calculate_sw_speed_alpha(energy)

                self.assertAlmostEqual(expected_speed, alpha_sw_speed.n, 0)
                self.assertAlmostEqual(expected_uncertainty, alpha_sw_speed.s, 0)

    def test_calculate_alpha_solar_wind_speed(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        alpha_solar_wind_center_of_mass = calculate_alpha_solar_wind_speed(uarray(count_rate, count_rate_delta), energy)

        self.assertAlmostEqual(496490, alpha_solar_wind_center_of_mass.nominal_value, 0)
        self.assertAlmostEqual(2811, alpha_solar_wind_center_of_mass.std_dev, 0)

    def test_calculate_alpha_center_of_mass_with_fake_data(self):
        test_cases = [
            (925, 1850, 50),
            (950, 1900, 51),
            (1000, 2000, 54),
            (1050, 2100, 56),
            (1100, 2200, 59)
        ]
        for modeled_proton_peak_energy, modeled_alpha_particle_peak_energy, expected_uncertainty in test_cases:
            with self.subTest(f"alpha particle peak energy is: {modeled_alpha_particle_peak_energy}"):
                count_rates, energies = generate_sweep_data(modeled_proton_peak_energy, modeled_alpha_particle_peak_energy)
                count_rate_delta = synthesize_uncertainties(count_rates)

                alpha_energy_center_of_mass = calculate_alpha_center_of_mass(uarray(count_rates, count_rate_delta), energies)

                self.assertAlmostEqual(modeled_alpha_particle_peak_energy, alpha_energy_center_of_mass.nominal_value, delta=5)
                self.assertAlmostEqual(expected_uncertainty, alpha_energy_center_of_mass.std_dev,0)

def get_sweep_voltages(sweep_table_id=0):
    energies = np.geomspace(19000, 100, 62)
    return energies / get_k_factor()

def synthesize_uncertainties(count_rates):
    return np.sqrt(6 * count_rates)

def generate_sweep_data(proton_center, alpha_center) -> tuple[np.ndarray, np.ndarray]:
    voltages = get_sweep_voltages()
    energies = voltages * get_k_factor()
    background = 0.1
    proton_peak = generate_peak(energies, 1200, proton_center, 100)
    alpha_peak = generate_peak(energies, 20, alpha_center, 100)

    return (proton_peak + alpha_peak + background, energies)

def get_k_factor():
    return 1.8

def generate_peak(energies, height, center, narrowness):
    return np.exp(np.log(height) - narrowness * np.square(np.log(energies) - np.log(center)))