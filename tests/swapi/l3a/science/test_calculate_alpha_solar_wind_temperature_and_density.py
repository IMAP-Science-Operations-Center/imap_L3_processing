from pathlib import Path
from unittest import TestCase

import numpy as np
from uncertainties import ufloat

import imap_processing
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable


class TestCalculateAlphaSolarWindTemperatureAndDensity(TestCase):
    def test_temperature_and_density_calibration_table_from_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240905_v002.cdf"

        calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(file_path)

        self.assertEqual((2, 11, 15), (len(calibration_table.grid[0]), len(calibration_table.grid[1]), len(calibration_table.grid[2])))
        self.assertEqual((2, 11, 15), calibration_table.density_grid.shape)
        self.assertEqual((2, 11, 15), calibration_table.temperature_grid.shape)

        sw_speed = ufloat(460, 10)
        density = ufloat(2.5, 0.5)
        temperature = ufloat(6.0000e+04, 100)

        self.assertEqual(2.5525, calibration_table.lookup_density(sw_speed, density, temperature).nominal_value)
        self.assertEqual(5.8537e+04, calibration_table.lookup_temperature(sw_speed, density, temperature).nominal_value)
