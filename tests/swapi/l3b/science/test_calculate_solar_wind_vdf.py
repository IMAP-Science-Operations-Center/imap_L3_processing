from pathlib import Path
from unittest import TestCase
from unittest.mock import create_autospec

import numpy as np

import imap_processing
from imap_processing.swapi.l3b.science.calculate_solar_wind_vdf import calculate_proton_solar_wind_vdf, \
    GeometricFactorCalibrationTable, calculate_alpha_solar_wind_vdf, calculate_pui_solar_wind_vdf


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

    def test_geometric_factor_table_from_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v001.cdf"

        calibration_table = GeometricFactorCalibrationTable.from_file(file_path)

        self.assertEqual(62, len(calibration_table.grid))
        self.assertEqual((62,), calibration_table.geometric_factor_grid.shape)

        known_energy = 8165.393844536367
        energy_to_interpolate = 14194.87288073211
        self.assertEqual(6.419796603112413e-13, calibration_table.lookup_geometric_factor(known_energy))
        self.assertAlmostEqual(5.711128783363629e-13, calibration_table.lookup_geometric_factor(energy_to_interpolate))
