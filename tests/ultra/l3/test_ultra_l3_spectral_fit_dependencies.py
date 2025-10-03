import unittest
from unittest.mock import patch, call, Mock

import numpy as np
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3SpectralIndexDependencies
from tests.test_helpers import get_test_data_path


class TestUltraL3SpectralFitDependencies(unittest.TestCase):
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.imap_data_access.download')
    def test_fetch_dependencies(self, mock_download, mock_rectangular_read_from_path):
        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        ancillary_file_name = 'imap_ultra_spx-energy-ranges_20250601_v000.cdf'
        expected_energy_ranges = [[5, 15], [15, 50]]
        map_input = ScienceInput(map_file_name)
        ancillary_input = AncillaryInput(ancillary_file_name)
        processing_input_collection = ProcessingInputCollection(map_input, ancillary_input)

        mock_download.side_effect = [
            "map_file",
            get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat')
        ]

        ultra_l3_dependencies = UltraL3SpectralIndexDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            call(map_file_name),
            call(ancillary_file_name)
        ])
        mock_rectangular_read_from_path.assert_called_once_with("map_file")
        self.assertEqual(ultra_l3_dependencies.map_data, mock_rectangular_read_from_path.return_value)
        np.testing.assert_array_equal(ultra_l3_dependencies.fit_energy_ranges, expected_energy_ranges)

    def test_fetch_dependencies_raises_exception_on_missing_science_file(self):
        ancillary_input = AncillaryInput('imap_ultra_spx-energy-ranges_20250601_v000.dat')
        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(ancillary_input))
        self.assertEqual("Missing Ultra L3 file", str(context.exception))

    def test_fetch_dependencies_raises_exception_on_missing_ancillary_file(self):
        science_input = ScienceInput('imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf')

        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(science_input))
        self.assertEqual("Missing fit energy ranges ancillary file", str(context.exception))

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    def test_from_file_paths(self, mock_read_from_path):
        map_file_path = Mock()
        ancillary_file_path = get_test_data_path('ultra') / 'imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat'
        expected_energy_range_values = np.loadtxt(ancillary_file_path)

        actual_dependencies = UltraL3SpectralIndexDependencies.from_file_paths(map_file_path, ancillary_file_path)
        self.assertEqual(mock_read_from_path.return_value, actual_dependencies.map_data)
        np.testing.assert_array_equal(actual_dependencies.fit_energy_ranges, expected_energy_range_values)

    def test_get_fit_energy_ranges(self):
        expected_energy_range_values = np.array([[5, 10], [15, 20]])

        dependencies = UltraL3SpectralIndexDependencies(map_data=Mock(),
                                                        fit_energy_ranges=expected_energy_range_values)
        actual_fit_energy_ranges = dependencies.get_fit_energy_ranges()

        np.testing.assert_array_equal(actual_fit_energy_ranges, expected_energy_range_values)
