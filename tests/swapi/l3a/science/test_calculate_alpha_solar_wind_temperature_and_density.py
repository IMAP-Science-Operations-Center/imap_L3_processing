from pathlib import Path
from unittest import TestCase

import numpy as np
from uncertainties import ufloat

import imap_processing
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable, calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps


class TestCalculateAlphaSolarWindTemperatureAndDensity(TestCase):
    def test_temperature_and_density_calibration_table_from_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_alpha-density-temperature-lut-text-not-cdf_20240905_v002.cdf"

        calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(file_path)

        self.assertEqual((2, 11, 15), (
            len(calibration_table.grid[0]), len(calibration_table.grid[1]), len(calibration_table.grid[2])))
        self.assertEqual((2, 11, 15), calibration_table.density_grid.shape)
        self.assertEqual((2, 11, 15), calibration_table.temperature_grid.shape)

        sw_speed = ufloat(460, 10)
        density = ufloat(2.5, 0.5)
        temperature = ufloat(6.0000e+04, 100)

        self.assertEqual(2.5525, calibration_table.lookup_density(sw_speed, density, temperature).nominal_value)
        self.assertEqual(5.8537e+04, calibration_table.lookup_temperature(sw_speed, density, temperature).nominal_value)

    def test_calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(self):
        peak_energies = np.array([2944, 2705, 2485, 2281, 2094])
        speed = ufloat(496.490, 2.811)
        peak_coincidence_rates = np.array([
            ufloat(15.2, 4.2708313008125245),
            ufloat(25.8, 5.5641710972974225),
            ufloat(26.0, 5.585696017507577),
            ufloat(15.4, 4.298837052040936),
            ufloat(5.8, 2.638181191654584),
        ])

        actual_temperature, actual_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
            peak_coincidence_rates, peak_energies, speed)

        self.assertAlmostEqual(411405.2171052396, actual_temperature.nominal_value)
        self.assertEqual(130246.773340626, actual_temperature.std_dev)
        self.assertEqual(0.10300409775543028, actual_density.nominal_value)
        self.assertEqual(0.013108888715912525, actual_density.std_dev)
