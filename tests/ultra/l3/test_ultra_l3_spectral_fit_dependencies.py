import unittest
from unittest.mock import patch, call, Mock

import numpy as np
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3SpectralFitDependencies
from tests.test_helpers import get_test_data_path


class TestUltraL3SpectralFitDependencies(unittest.TestCase):
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.download')
    def test_fetch_dependencies(self, mock_download, mock_healpix_read_from_path):
        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        ancillary_file_name = 'imap_ultra_spx-energy-ranges_20250601_v000.cdf'
        expected_energy_ranges = [[5, 15], [15, 50]]
        map_input = ScienceInput(map_file_name)
        ancillary_input = AncillaryInput(ancillary_file_name)
        processing_input_collection = ProcessingInputCollection(map_input, ancillary_input)

        mock_download.side_effect = [
            "map_file",
            get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250507_v000.txt')
        ]

        ultra_l3_dependencies = UltraL3SpectralFitDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            call(map_file_name),
            call(ancillary_file_name)
        ])
        mock_healpix_read_from_path.assert_called_once_with("map_file")
        self.assertEqual(ultra_l3_dependencies.ultra_l3_data, mock_healpix_read_from_path.return_value)
        np.testing.assert_array_equal(ultra_l3_dependencies.energy_fit_ranges, expected_energy_ranges)

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_path')
    def test_from_file_paths(self, mock_read_from_path):
        map_file_path = Mock()
        ancillary_file_path = get_test_data_path('ultra') / 'imap_ultra_ulc-spx-energy-ranges_20250507_v000.txt'
        expected_energy_range_values = np.loadtxt(ancillary_file_path)

        actual_dependencies = UltraL3SpectralFitDependencies.from_file_paths(map_file_path, ancillary_file_path)
        self.assertEqual(mock_read_from_path.return_value, actual_dependencies.ultra_l3_data)
        np.testing.assert_array_equal(actual_dependencies.energy_fit_ranges, expected_energy_range_values)
