from unittest import TestCase
from unittest.mock import create_autospec

import numpy as np

from imap_processing.swapi.l3b.science.calculate_solar_wind_vdf import calculate_proton_solar_wind_vdf, \
    GeometricFactorCalibrationTable, calculate_alpha_solar_wind_vdf, calculate_pui_solar_wind_vdf, \
    calculate_delta_minus_plus


class TestCalculateSolarWindVDF(TestCase):
    def test_calculate_proton_solar_wind_vdf(self):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        efficiency = 0.0882
        mock_geometric_factor_table = create_autospec(GeometricFactorCalibrationTable)
        mock_geometric_factor_table.lookup_geometric_factor.return_value = np.array([1e-12, 1e-13, 1e-14])

        expected_velocities = [437.6947142244463, 379.05474162054054, 309.496900517614]
        expected_probabilities = [14874.030602544019, 396640.8160678405, 8924418.36152641]

        velocities, probabilities = calculate_proton_solar_wind_vdf(energies, count_rates, efficiency,
                                                                    mock_geometric_factor_table)

        np.testing.assert_array_equal(velocities, expected_velocities)
        np.testing.assert_array_almost_equal(probabilities, expected_probabilities)
        mock_geometric_factor_table.lookup_geometric_factor.assert_called_with(energies)

    def test_calculate_delta_minus_plus(self):
        energies = np.array([8, 32, 128])
        delta_minus_plus = calculate_delta_minus_plus(energies)
        np.testing.assert_array_equal([4, 16, 64], delta_minus_plus.delta_minus)
        np.testing.assert_array_equal([8, 32, 128], delta_minus_plus.delta_plus)

    def test_calculate_delta_minus_plus_reversed(self):
        energies = np.array([128, 32, 8])
        delta_minus_plus = calculate_delta_minus_plus(energies)
        np.testing.assert_array_equal([64, 16, 4], delta_minus_plus.delta_minus)
        np.testing.assert_array_equal([128, 32, 8], delta_minus_plus.delta_plus)

    def test_calculate_delta_minus_plus_uneven(self):
        energies = np.array([8, 32, 288])
        delta_minus_plus = calculate_delta_minus_plus(energies)
        np.testing.assert_array_equal([4, 16, 192], delta_minus_plus.delta_minus)
        np.testing.assert_array_equal([8, 64, 576], delta_minus_plus.delta_plus)

    def test_calculate_alpha_solar_wind_vdf(self):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        efficiency = 0.0882
        mock_geometric_factor_table = create_autospec(GeometricFactorCalibrationTable)
        mock_geometric_factor_table.lookup_geometric_factor.return_value = np.array([1e-12, 1e-13, 1e-14])

        expected_velocities = [310.5624166704235, 268.95494229727456, 219.6007908093385]
        expected_probabilities = [29544.284683, 787847.591534, 17726570.809506]

        velocities, probabilities = calculate_alpha_solar_wind_vdf(energies, count_rates, efficiency,
                                                                   mock_geometric_factor_table)

        np.testing.assert_array_equal(velocities, expected_velocities)
        np.testing.assert_array_almost_equal(probabilities, expected_probabilities)
        mock_geometric_factor_table.lookup_geometric_factor.assert_called_with(energies)

    def test_calculate_pickup_ion_solar_wind_vdf(self):
        energies = np.array([1000, 750, 500])
        count_rates = np.array([10, 20, 30])
        efficiency = 0.0882
        mock_geometric_factor_table = create_autospec(GeometricFactorCalibrationTable)
        mock_geometric_factor_table.lookup_geometric_factor.return_value = np.array([1e-12, 1e-13, 1e-14])

        expected_velocities = [219.58573945228636, 190.16682867447082, 155.27056541857408]
        expected_probabilities = [59096.670015, 1575911.200407, 35458002.009151]

        velocities, probabilities = calculate_pui_solar_wind_vdf(energies, count_rates, efficiency,
                                                                 mock_geometric_factor_table)

        np.testing.assert_array_equal(velocities, expected_velocities)
        np.testing.assert_array_almost_equal(probabilities, expected_probabilities)
        mock_geometric_factor_table.lookup_geometric_factor.assert_called_with(energies)
