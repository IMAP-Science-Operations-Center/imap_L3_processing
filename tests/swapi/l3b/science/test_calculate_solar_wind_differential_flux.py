from unittest import TestCase
from unittest.mock import create_autospec

import numpy as np

from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import \
    calculate_combined_solar_wind_differential_flux
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable


class TestCalculateSolarWindDifferentialFlux(TestCase):
    def test_calculate_combined_differential_flux(self):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        efficiency = 0.0882
        mock_geometric_factor_table = create_autospec(GeometricFactorCalibrationTable)
        mock_geometric_factor_table.lookup_geometric_factor.return_value = np.array([1e-12, 1e-13, 1e-14])

        differential_flux = calculate_combined_solar_wind_differential_flux(energies, count_rates, efficiency,
                                                                            mock_geometric_factor_table)
        expected_flux = np.array([11.337868480725623, 302.3431594860166, 6802.721088435374])
        np.testing.assert_array_almost_equal(expected_flux, differential_flux)
        mock_geometric_factor_table.lookup_geometric_factor.assert_called_with(energies)
