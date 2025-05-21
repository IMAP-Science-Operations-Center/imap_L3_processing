import unittest
from unittest.mock import patch, call

from imap_data_access import ScienceInput, ProcessingInputCollection

from imap_l3_processing.ultra.l3.ultra_l3_to_rectangular_dependencies import UltraL3ToRectangularDependencies


class TestUltraL3ToRectangularDependencies(unittest.TestCase):

    @patch('imap_l3_processing.ultra.l3.ultra_l3_to_rectangular_dependencies.HealPixIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_to_rectangular_dependencies.download')
    def test_fetch_dependencies(self, mock_download, mock_healpix_read_from_path):
        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        map_input = ScienceInput(map_file_name)
        processing_input_collection = ProcessingInputCollection(map_input)

        mock_download.side_effect = [
            "map_file",
        ]

        ultra_l3_to_rectangular_dependencies = UltraL3ToRectangularDependencies.fetch_dependencies(
            processing_input_collection)

        mock_download.assert_has_calls([
            call(map_file_name),
        ])

        mock_healpix_read_from_path.assert_called_once_with("map_file")

        self.assertEqual(ultra_l3_to_rectangular_dependencies.healpix_map_data,
                         mock_healpix_read_from_path.return_value)
