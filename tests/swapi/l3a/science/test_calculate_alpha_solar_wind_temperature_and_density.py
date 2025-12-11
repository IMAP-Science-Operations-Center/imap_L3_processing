from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray

import imap_l3_processing
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable, calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps
from tests.test_helpers import get_test_data_path


class TestCalculateAlphaSolarWindTemperatureAndDensity(TestCase):
    def setUp(self) -> None:
        data_file_path = get_test_data_path("swapi/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf")
        with CDF(str(data_file_path)) as cdf:
            self.energy = cdf["esa_energy"][...]
            self.count_rate = cdf["swp_coin_rate"][...]
            self.count_rate_delta = cdf["swp_coin_unc"][...]

        lookup_table_file_path = get_test_data_path("swapi/imap_swapi_alpha-density-temperature-lut_20240920_v000.dat")
        self.calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(lookup_table_file_path)

    def test_temperature_and_density_calibration_table_from_file(self):
        file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / "imap_swapi_alpha-density-temperature-lut_20240920_v000.dat"

        calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(file_path)

        sw_speed = ufloat(460, 10)
        density = ufloat(0.25, 0.03)
        temperature = ufloat(6.0000e+05, 100)

        self.assertAlmostEqual(0.39999999,
                               calibration_table.lookup_density(sw_speed, density, temperature).nominal_value)
        self.assertAlmostEqual(7.2e+05,
                               calibration_table.lookup_temperature(sw_speed, density, temperature).nominal_value)

    def test_calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(self):
        speed = ufloat(496.490, 2.811)

        efficiency = 0.08

        actual_temperature, actual_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
            self.calibration_table, speed,
            uarray(self.count_rate, self.count_rate_delta), self.energy, efficiency)

        np.testing.assert_allclose(493916.041942, actual_temperature.nominal_value)
        np.testing.assert_allclose(155210.29054, actual_temperature.std_dev)
        np.testing.assert_allclose(3.8704e-3 / efficiency, actual_density.nominal_value, rtol=1e-4)
        np.testing.assert_allclose(4.9177e-4 / efficiency, actual_density.std_dev, rtol=1e-4)

    @patch(
        'imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density.get_alpha_peak_indices')
    @patch(
        'imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density.calculate_combined_sweeps')
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
        efficiency = 0.7

        mock_calculate_combine_sweeps.return_value = peak_coincidence_rates, peak_energies
        mock_get_alpha_peak_indices.return_value = slice(0, 5)
        with self.assertRaises(ValueError) as e:
            calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(self.calibration_table, speed,
                                                                                   uarray(self.count_rate,
                                                                                          self.count_rate_delta),
                                                                                   self.energy, efficiency)
        self.assertEqual(str(e.exception.args[0]), "Failed to fit - chi-squared too large")
        self.assertAlmostEqual(e.exception.args[1], 11.018918806664246)

    @patch(
        'imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density.get_alpha_peak_indices')
    @patch(
        'imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density.calculate_combined_sweeps')
    def test_excludes_zero_count_rates_from_fit(self, mock_calculate_combine_sweeps, mock_get_alpha_peak_indices):
        speed = ufloat(496.490, 2.811)
        peak_energies = np.array([2944, 2705, 2485, 2281, 2094])
        peak_coincidence_rates = np.array([
            ufloat(15.2, 4.2708313008125245),
            ufloat(30.8, 5.5641710972974225),
            ufloat(0, 5.585696017507577),
            ufloat(30.4, 4.298837052040936),
            ufloat(8.8, 2.638181191654584),
        ]) * 10
        efficiency = 0.7

        mock_calculate_combine_sweeps.return_value = peak_coincidence_rates, peak_energies
        mock_get_alpha_peak_indices.return_value = slice(0, 5)
        with assert_does_not_error():
            calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(self.calibration_table, speed,
                                                                                   uarray(self.count_rate,
                                                                                          self.count_rate_delta),
                                                                                   self.energy, efficiency)


@contextmanager
def assert_does_not_error():
    yield
