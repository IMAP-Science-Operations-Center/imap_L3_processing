from _ast import Slice
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray

import imap_processing
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable, calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps


class TestCalculateAlphaSolarWindTemperatureAndDensity(TestCase):
    def setUp(self) -> None:
        data_file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf"
        with CDF(str(data_file_path)) as cdf:
            self.energy = cdf["energy"][...]
            self.count_rate = cdf["swp_coin_rate"][...]
            self.count_rate_delta = cdf["swp_coin_unc"][...]

        lookup_table_file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf"
        self.calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(lookup_table_file_path)

    def test_temperature_and_density_calibration_table_from_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240920_v004.cdf"

        calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(file_path)

        self.assertEqual((15, 16, 23), (
            len(calibration_table.grid[0]), len(calibration_table.grid[1]), len(calibration_table.grid[2])))
        self.assertEqual((15, 16, 23), calibration_table.density_grid.shape)
        self.assertEqual((15, 16, 23), calibration_table.temperature_grid.shape)

        sw_speed = ufloat(460, 10)
        density = ufloat(0.25, 0.03)
        temperature = ufloat(6.0000e+05, 100)

        self.assertAlmostEqual(0.39999999,
                               calibration_table.lookup_density(sw_speed, density, temperature).nominal_value)
        self.assertEqual(7.2e+05, calibration_table.lookup_temperature(sw_speed, density, temperature).nominal_value)

    def test_calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(self):
        speed = ufloat(496.490, 2.811)

        actual_temperature, actual_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
            self.calibration_table, speed,
            uarray(self.count_rate, self.count_rate_delta), self.energy)

        self.assertAlmostEqual(493686.26052628754, actual_temperature.nominal_value, 2)
        self.assertAlmostEqual(156296.1278672896, actual_temperature.std_dev, 2)
        self.assertAlmostEqual(0.164806556, actual_density.nominal_value)
        self.assertAlmostEqual(0.02097422, actual_density.std_dev)

    @patch(
        'imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density.get_alpha_peak_indices')
    @patch(
        'imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density.calculate_combined_sweeps')
    def test_raises_error_when_chi_squared_over_ten(self, mock_calculate_combine_sweeps, mock_get_alpha_peak_indices):
        speed = ufloat(496.490, 2.811)
        peak_energies = np.array([2944, 2705, 2485, 2281, 2094])
        peak_coincidence_rates = np.array([
            ufloat(20.2, 4.2708313008125245),
            ufloat(10.8, 5.5641710972974225),
            ufloat(606.0, 5.585696017507577),
            ufloat(10.4, 4.298837052040936),
            ufloat(5.8, 2.638181191654584),
        ])

        mock_calculate_combine_sweeps.return_value = peak_coincidence_rates, peak_energies
        mock_get_alpha_peak_indices.return_value = slice(0, 5)
        with self.assertRaises(ValueError) as e:
            calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(self.calibration_table, speed,
                                                                                   uarray(self.count_rate,
                                                                                          self.count_rate_delta),
                                                                                   self.energy)
        self.assertEqual(str(e.exception.args[0]), "Failed to fit - chi-squared too large")
        self.assertAlmostEqual(e.exception.args[1], 13.6018326)
