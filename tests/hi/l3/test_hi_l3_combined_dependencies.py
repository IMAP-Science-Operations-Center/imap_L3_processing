import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies import HiL3CombinedMapDependencies


class TestHiL3CombinedDependencies(unittest.TestCase):
    @patch("imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies.RectangularIntensityMapData.read_from_path")
    def test_from_file_paths(self, read_cdf):
        hi_map_paths = [
            Path("test_hi_l3_cdf1.cdf"),
            Path("test_hi_l3_cdf2.cdf"),
            Path("test_hi_l3_cdf3.cdf")
        ]

        expected_return_maps = [sentinel.read_data1,
                                sentinel.read_data2,
                                sentinel.read_data3]

        read_cdf.side_effect = expected_return_maps

        result = HiL3CombinedMapDependencies.from_file_paths(hi_map_paths)

        read_cdf.assert_has_calls([call(path) for path in hi_map_paths])

        self.assertEqual(read_cdf.call_count, 3)
        self.assertEqual(expected_return_maps, result.maps)

    @patch("imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies.imap_data_access.download")
    @patch("imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies.RectangularIntensityMapData.read_from_path")
    def test_fetch_dependencies(self, read_cdf, mock_download):
        file_name1 = "imap_hi_l3_h90-spx-h-hf-sp-ram-hae-4deg-6mo_20250422_v001.cdf"
        file_name2 = "imap_hi_l3_h90-spx-h-hf-sp-anti-hae-4deg-6mo_20250422_v001.cdf"
        file_name3 = "imap_hi_l3_h45-spx-h-hf-sp-ram-hae-4deg-6mo_20250422_v001.cdf"
        file_name4 = "imap_hi_l3_h45-spx-h-hf-sp-anti-hae-4deg-6mo_20250422_v001.cdf"
        file_names = [file_name1, file_name2, file_name3, file_name4]
        processing_input = ProcessingInputCollection(
            ScienceInput(file_name1),
            ScienceInput(file_name2),
            ScienceInput(file_name3),
            ScienceInput(file_name4),
        )

        downloaded_paths = [
            Path("test_hi_l3_cdf1.cdf"),
            Path("test_hi_l3_cdf2.cdf"),
            Path("test_hi_l3_cdf3.cdf"),
            Path("test_hi_l3_cdf4.cdf"),
        ]
        mock_download.side_effect = downloaded_paths

        read_maps = [
            sentinel.read_data1,
            sentinel.read_data2,
            sentinel.read_data3,
            sentinel.read_data4,
        ]
        read_cdf.side_effect = read_maps

        result = HiL3CombinedMapDependencies.fetch_dependencies(processing_input)

        mock_download.assert_has_calls([call(file_name) for file_name in file_names])
        read_cdf.assert_has_calls([call(path) for path in downloaded_paths])

        self.assertEqual(read_cdf.call_count, 4)
        self.assertEqual(mock_download.call_count, 4)

        self.assertEqual(read_maps, result.maps)
