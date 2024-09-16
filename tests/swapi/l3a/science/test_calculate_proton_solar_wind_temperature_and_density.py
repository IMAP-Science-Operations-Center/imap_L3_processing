import itertools
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray

import imap_processing
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    proton_count_rate_model, \
    calculate_proton_solar_wind_temperature_and_density_for_one_sweep, TemperatureAndDensityCalibrationTable, \
    calculate_uncalibrated_proton_solar_wind_temperature_and_density, \
    calculate_proton_solar_wind_temperature_and_density
from tests.swapi.l3a.science.test_calculate_alpha_solar_wind_speed import synthesize_uncertainties


class TestCalculateProtonSolarWindTemperatureAndDensity(TestCase):
    def test_uncalibrated_calculate_a_single_sweep_from_example_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        temperature, density = calculate_proton_solar_wind_temperature_and_density_for_one_sweep(
            uarray(count_rate, count_rate_delta)[4], energy)

        self.assertAlmostEqual(102267, temperature.nominal_value, 0)
        self.assertAlmostEqual(6376, temperature.std_dev, 0)
        self.assertAlmostEqual(3.76, density.nominal_value, 2)
        self.assertAlmostEqual(0.1555, density.std_dev, 4)

    def test_uncalibrated_calculate_using_five_sweeps_from_example_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        temperature, density = calculate_uncalibrated_proton_solar_wind_temperature_and_density(
            uarray(count_rate, count_rate_delta), energy)

        self.assertAlmostEqual(100109, temperature.nominal_value, 0)
        self.assertAlmostEqual(2379, temperature.std_dev, 0)
        self.assertAlmostEqual(4.953, density.nominal_value, 3)
        self.assertAlmostEqual(8.028e-2, density.std_dev, 5)

    def test_calibrate_density_and_temperature_using_lookup_table(self):
        speed_values = [250, 1000]
        deflection_angle_values = [0, 6]
        clock_angle_values = [0, 360]
        density_values = [1, 10]
        temperature_values = [1000, 100000]

        lookup_table = self.generate_lookup_table(speed_values, deflection_angle_values, clock_angle_values,
                                                  density_values, temperature_values)

        temperature = lookup_table.calibrate_temperature(ufloat(450, 2), ufloat(3, 0.1),
                                                         ufloat(1, 1), ufloat(4, 0.1),
                                                         ufloat(50000, 10000))
        density = lookup_table.calibrate_density(ufloat(450, 2), ufloat(3, 0.1), ufloat(1, 1),
                                                 ufloat(4, 0.1), ufloat(50000, 10000))

        self.assertAlmostEqual(50000 * 0.97561, temperature.n, 3)
        self.assertAlmostEqual(10000 * 0.97561, temperature.s, 3)

        self.assertAlmostEqual(4 * 1.021, density.n)
        self.assertAlmostEqual(0.1 * 1.021, density.s)

    def test_calculate_temperature_and_density(self):
        speed_values = [250, 1000]
        deflection_angle_values = [0, 6]
        clock_angle_values = [0, 360]
        density_values = [1, 10]
        temperature_values = [1000, 200000]

        lookup_table = self.generate_lookup_table(speed_values, deflection_angle_values, clock_angle_values,
                                                  density_values, temperature_values)

        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        temperature, density = calculate_proton_solar_wind_temperature_and_density(lookup_table, ufloat(450, 2),
                                                                                   ufloat(3, 0.1), ufloat(1, 1),
                                                                                   uarray(count_rate, count_rate_delta),
                                                                                   energy)

        self.assertAlmostEqual(100109 * 0.97561, temperature.nominal_value, 0)
        self.assertAlmostEqual(2379 * 0.97561, temperature.std_dev, 0)

        self.assertAlmostEqual(4.953 * 1.021, density.nominal_value, 3)
        self.assertAlmostEqual(8.028e-2 * 1.021, density.std_dev, 5)

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

    def test_throws_error_when_chi_squared_over_ten(self):
        test_cases = [
            ["throws error", uarray([100, 200, 300, 150, 50], 8), np.array([1000, 900, 800, 700, 600]), True,
             10.959014166245833],
            ["does not throw error", uarray([100, 200, 300, 150, 50], 9), np.array([1000, 900, 800, 700, 600]), False,
             8.658974156048295]
        ]

        for name, coincident_count_rates, energy, error_flag, expected_chi_squared in test_cases:
            with self.subTest(name):
                try:
                    calculate_proton_solar_wind_temperature_and_density_for_one_sweep(coincident_count_rates, energy)
                    did_error = False
                except ValueError as e:
                    did_error = True
                    exception = e
                if did_error:
                    self.assertEqual("Failed to fit - chi-squared too large", exception.args[0])
                    self.assertAlmostEqual(expected_chi_squared, exception.args[1])
                self.assertEqual(error_flag, did_error)

    def generate_lookup_table(self, speed_values, deflection_angle_values, clock_angle_values, density_values,
                              temperature_values):

        coords = (speed_values, deflection_angle_values, clock_angle_values, density_values, temperature_values)

        lut_rows = []
        for (speed, deflection, clock_angle, density, temperature) in itertools.product(*coords):
            output_density = 1.021 * density
            output_temperature = 0.97561 * temperature
            lut_rows.append([speed, deflection, clock_angle, density, output_density, temperature, output_temperature])
        return TemperatureAndDensityCalibrationTable(np.array(lut_rows))
