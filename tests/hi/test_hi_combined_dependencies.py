import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.hi.hi_combined_sensor_dependencies import HiL3CombinedMapDependencies


class TestHiCombinedDependencies(unittest.TestCase):
    @patch("imap_l3_processing.hi.hi_combined_sensor_dependencies.RectangularIntensityMapData.read_from_path")
    def test_from_file_paths(self, read_cdf):
        hi_map_paths = [
            Path("test_h45_l3_cdf1.cdf"),
            Path("test_h45_l3_cdf2.cdf"),
            Path("test_h45_l3_cdf3.cdf"),
            Path("test_h45_l3_cdf4.cdf"),
            Path("test_h90_l3_cdf1.cdf"),
            Path("test_h90_l3_cdf2.cdf"),
            Path("test_h90_l3_cdf3.cdf"),
            Path("test_h90_l3_cdf4.cdf"),
        ]

        expected_45_return_maps = [sentinel.read_data45_1,
                                   sentinel.read_data45_2,
                                   sentinel.read_data45_3,
                                   sentinel.read_data45_4]

        expected_90_return_maps = [sentinel.read_data90_1,
                                   sentinel.read_data90_2,
                                   sentinel.read_data90_3,
                                   sentinel.read_data90_4]

        read_cdf.side_effect = expected_45_return_maps + expected_90_return_maps

        result = HiL3CombinedMapDependencies.from_file_paths(hi_map_paths)

        read_cdf.assert_has_calls([call(path) for path in hi_map_paths])

        self.assertEqual(read_cdf.call_count, 8)
        self.assertEqual(expected_45_return_maps, result.h45_maps)
        self.assertEqual(expected_90_return_maps, result.h90_maps)

    @patch("imap_l3_processing.hi.hi_combined_sensor_dependencies.imap_data_access.download")
    @patch("imap_l3_processing.hi.hi_combined_sensor_dependencies.RectangularIntensityMapData.read_from_path")
    def test_fetch_dependencies(self, read_cdf, mock_download):
        file_name1 = "imap_hi_l3_h45-ena-h-hf-sp-ram-hae-4deg-6mo_20251022_v001.cdf"
        file_name2 = "imap_hi_l3_h45-ena-h-hf-sp-anti-hae-4deg-6mo_20251022_v001.cdf"
        file_name3 = "imap_hi_l3_h45-ena-h-hf-sp-ram-hae-4deg-6mo_20250422_v001.cdf"
        file_name4 = "imap_hi_l3_h45-ena-h-hf-sp-anti-hae-4deg-6mo_20250422_v001.cdf"
        file_name5 = "imap_hi_l3_h90-ena-h-hf-sp-ram-hae-4deg-6mo_20251022_v001.cdf"
        file_name6 = "imap_hi_l3_h90-ena-h-hf-sp-anti-hae-4deg-6mo_20251022_v001.cdf"
        file_name7 = "imap_hi_l3_h90-ena-h-hf-sp-ram-hae-4deg-6mo_20250422_v001.cdf"
        file_name8 = "imap_hi_l3_h90-ena-h-hf-sp-anti-hae-4deg-6mo_20250422_v001.cdf"

        file_names = [file_name1, file_name2, file_name3, file_name4, file_name5, file_name6, file_name7, file_name8]
        processing_input = ProcessingInputCollection(
            ScienceInput(file_name1),
            ScienceInput(file_name2),
            ScienceInput(file_name3),
            ScienceInput(file_name4),
            ScienceInput(file_name5),
            ScienceInput(file_name6),
            ScienceInput(file_name7),
            ScienceInput(file_name8),
        )

        downloaded_paths = [
            Path("test_h45_l3_cdf1.cdf"),
            Path("test_h45_l3_cdf2.cdf"),
            Path("test_h45_l3_cdf3.cdf"),
            Path("test_h45_l3_cdf4.cdf"),
            Path("test_h90_l3_cdf5.cdf"),
            Path("test_h90_l3_cdf6.cdf"),
            Path("test_h90_l3_cdf7.cdf"),
            Path("test_h90_l3_cdf8.cdf"),
        ]
        mock_download.side_effect = downloaded_paths

        read_maps = [
            sentinel.read_data1,
            sentinel.read_data2,
            sentinel.read_data3,
            sentinel.read_data4,
            sentinel.read_data5,
            sentinel.read_data6,
            sentinel.read_data7,
            sentinel.read_data8,
        ]
        read_cdf.side_effect = read_maps

        result = HiL3CombinedMapDependencies.fetch_dependencies(processing_input)

        mock_download.assert_has_calls([call(file_name) for file_name in file_names])
        read_cdf.assert_has_calls([call(path) for path in downloaded_paths])

        self.assertEqual(read_cdf.call_count, 8)
        self.assertEqual(mock_download.call_count, 8)

        self.assertEqual([sentinel.read_data1, sentinel.read_data2, sentinel.read_data3, sentinel.read_data4],
                         result.h45_maps)
        self.assertEqual([sentinel.read_data5, sentinel.read_data6, sentinel.read_data7, sentinel.read_data8],
                         result.h90_maps)
