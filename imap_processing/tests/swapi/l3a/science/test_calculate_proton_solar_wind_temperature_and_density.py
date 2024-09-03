from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF
from uncertainties.unumpy import uarray

import imap_processing
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    calculate_proton_solar_wind_temperature_and_density, proton_count_rate_model, \
    calculate_proton_solar_wind_temperature_and_density_for_one_sweep
from imap_processing.tests.swapi.l3a.science.test_calculate_alpha_solar_wind_speed import synthesize_uncertainties


class TestCalculateProtonSolarWindTemperatureAndDensity(TestCase):
    def test_calculate_a_single_sweep_from_example_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        temperature, density = calculate_proton_solar_wind_temperature_and_density_for_one_sweep(
            uarray(count_rate, count_rate_delta)[4], energy)

        self.assertAlmostEqual(101734, temperature.nominal_value, 0)
        self.assertAlmostEqual(538, temperature.std_dev, 0)
        self.assertAlmostEqual(3.76, density.nominal_value, 2)
        self.assertAlmostEqual(8.64e-3, density.std_dev, 5)

    def test_calculate_using_five_sweeps_from_example_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        temperature, density = calculate_proton_solar_wind_temperature_and_density(
            uarray(count_rate, count_rate_delta), energy)

        # self.assertAlmostEqual(100476, temperature.nominal_value, 0)
        # self.assertAlmostEqual(265, temperature.std_dev, 0)
        # self.assertAlmostEqual(5.102, density.nominal_value, 3)
        # self.assertAlmostEqual(6.15e-3, density.std_dev, 5)

        self.assertAlmostEqual(100622, temperature.nominal_value, 0)
        self.assertAlmostEqual(245, temperature.std_dev, 0)
        self.assertAlmostEqual(4.674, density.nominal_value, 3)
        self.assertAlmostEqual(5.00e-3, density.std_dev, 5)

    def test_can_recover_density_and_temperature_from_model_data(self):
        test_cases = [
            (5, 100e3, 450),
            (3, 100e3, 450),
            (3, 80e3, 550),
            (3, 80e3, 750),
            (3, 200e3, 750),
            (0.05, 1e6, 450),
            (300, 200e3, 750),
        ]

        energy = np.geomspace(100, 19000, 62)
        for density, temperature, speed in test_cases:
            with self.subTest(f"{density}cm^-3, {temperature}K, {speed}km/s"):
                count_rates = proton_count_rate_model(energy, density, temperature, speed)
                fake_uncertainties = synthesize_uncertainties(count_rates)
                count_rates_with_uncertainties = uarray(count_rates, fake_uncertainties)
                fit_temperature, fit_density = calculate_proton_solar_wind_temperature_and_density_for_one_sweep(
                    count_rates_with_uncertainties, energy)
                self.assertAlmostEqual(density, fit_density.nominal_value, 6)
                self.assertAlmostEqual(temperature, fit_temperature.nominal_value, 0)
