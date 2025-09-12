import itertools
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray

import imap_l3_processing
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    proton_count_rate_model, \
    calculate_proton_solar_wind_temperature_and_density_for_one_sweep, ProtonTemperatureAndDensityCalibrationTable, \
    calculate_uncalibrated_proton_solar_wind_temperature_and_density, \
    calculate_proton_solar_wind_temperature_and_density
from tests.swapi.l3a.science.test_calculate_alpha_solar_wind_speed import synthesize_uncertainties
from tests.test_helpers import get_test_data_path


class TestCalculateProtonSolarWindTemperatureAndDensity(TestCase):
    def test_uncalibrated_calculate_a_single_sweep_from_example_file(self):
        file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        efficiency = 0.1

        temperature, density = calculate_proton_solar_wind_temperature_and_density_for_one_sweep(
            uarray(count_rate, count_rate_delta)[4], energy, efficiency)

        self.assertAlmostEqual(102267, temperature.nominal_value, 0)
        self.assertAlmostEqual(6376, temperature.std_dev, 0)
        self.assertAlmostEqual(8.829e-2 / efficiency, density.nominal_value, 2)
        self.assertAlmostEqual(3.6515e-3 / efficiency, density.std_dev, 3)

    def test_uncalibrated_calculate_using_five_sweeps_from_example_file(self):
        file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / 'imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        efficiency = 0.8
        temperature, density = calculate_uncalibrated_proton_solar_wind_temperature_and_density(
            uarray(count_rate, count_rate_delta), energy, efficiency)

        np.testing.assert_allclose(100109, temperature.nominal_value, rtol=1e-5)
        np.testing.assert_allclose(2379, temperature.std_dev, rtol=1e-3)
        np.testing.assert_allclose(0.1163 / efficiency, density.nominal_value, rtol=1e-4)
        np.testing.assert_allclose(1.8852e-3 / efficiency, density.std_dev, rtol=1e-4)

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

    def test_density_temperature_lookup_table_works_with_clock_angles_outside_0_to_360(self):
        speed_values = [250, 1000]
        deflection_angle_values = [0, 6]
        clock_angle_values = [0, 360]
        density_values = [1, 10]
        temperature_values = [1000, 100000]

        lookup_table = self.generate_lookup_table(speed_values, deflection_angle_values, clock_angle_values,
                                                  density_values, temperature_values)

        temperature = lookup_table.calibrate_temperature(ufloat(450, 2), ufloat(3, 0.1),
                                                         ufloat(0.0, 1), ufloat(4, 0.1),
                                                         ufloat(50000, 10000))

        density = lookup_table.calibrate_density(ufloat(450, 2), ufloat(3, 0.1), ufloat(360, 1),
                                                 ufloat(4, 0.1), ufloat(50000, 10000))

        self.assertAlmostEqual(0.97561 * 50000, temperature.n, places=3)
        self.assertAlmostEqual(0.97561 * 10000, temperature.s, places=3)

        self.assertAlmostEqual(1.021 * 4, density.n)
        self.assertAlmostEqual(1.021 * 0.1, density.s)

    def test_calculate_temperature_and_density(self):
        speed_values = [250, 1000]
        deflection_angle_values = [0, 6]
        clock_angle_values = [0, 360]
        density_values = [1, 10]
        temperature_values = [1000, 200000]
        efficiency = 0.1

        lookup_table = self.generate_lookup_table(speed_values, deflection_angle_values, clock_angle_values,
                                                  density_values, temperature_values)

        file_path = get_test_data_path('swapi/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf')
        with CDF(str(file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

        temperature, density = calculate_proton_solar_wind_temperature_and_density(lookup_table, ufloat(450, 2),
                                                                                   ufloat(3, 0.1), ufloat(1, 1),
                                                                                   uarray(count_rate, count_rate_delta),
                                                                                   energy, efficiency)

        np.testing.assert_allclose(100109 * 0.97561, temperature.nominal_value, rtol=1e-4)
        np.testing.assert_allclose(2379 * 0.97561, temperature.std_dev, rtol=2e-4)

        np.testing.assert_allclose(0.1163 * 1.021 / efficiency, density.nominal_value, rtol=1e-4)
        np.testing.assert_allclose(1.8852e-3 * 1.021 / efficiency, density.std_dev, rtol=1e-4)

    def test_proton_count_rate_model_accounts_for_efficiency(self):
        efficiency_factor = 0.5

        cases = [
            (np.array([800, 900, 1000]), 1, 2, 1e6, 700, np.array([9.406743, 25.85924, 62.665463])),
            (np.array([800, 900, 1000]), efficiency_factor, 2, 1e6, 700, np.array([9.406743, 25.85924, 62.665463]) * efficiency_factor),
        ]

        for ev_per_q, efficiency, density_per_cm3, temperature, bulk_flow_speed_km_per_s, expected in cases:
            result = proton_count_rate_model(efficiency, ev_per_q, density_per_cm3, temperature, bulk_flow_speed_km_per_s)
            np.testing.assert_array_almost_equal(result, expected)

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
        efficiency = 1
        for density, temperature, speed in test_cases:
            with self.subTest(f"{density}cm^-3, {temperature}K, {speed}km/s"):
                count_rates = proton_count_rate_model(efficiency, energy, density, temperature, speed)
                fake_uncertainties = synthesize_uncertainties(count_rates)
                count_rates_with_uncertainties = uarray(count_rates, fake_uncertainties)
                fit_temperature, fit_density = calculate_proton_solar_wind_temperature_and_density_for_one_sweep(
                    count_rates_with_uncertainties, energy, efficiency)
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
                    calculate_proton_solar_wind_temperature_and_density_for_one_sweep(coincident_count_rates, energy, 1)
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
        return ProtonTemperatureAndDensityCalibrationTable(np.array(lut_rows))
