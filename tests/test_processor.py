import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.swapi.swapi_processor import SwapiProcessor


class TestProcessor(unittest.TestCase):
    @patch('imap_l3_processing.processor.spiceypy')
    def test_get_parent_file_names(self, mock_spiceypy):
        input_collection = ProcessingInputCollection(ScienceInput("imap_swapi_l2_science_20250417_v001.cdf"),
                                                     ScienceInput("imap_swapi_l2_science_20250418_v001.cdf"),
                                                     AncillaryInput("imap_swapi_lookup_20250417_v001.cdf"))
        mock_spiceypy.ktotal.return_value = 2
        mock_spiceypy.kdata.side_effect = [['april/kernel_1', 'type', 'source', 'handle'],
                                           ['may/kernel_2', 'type', 'source', 'handle']]

        processor = SwapiProcessor(input_collection, Mock())

        expected_parent_file_names = [
            "imap_swapi_l2_science_20250417_v001.cdf",
            "imap_swapi_l2_science_20250418_v001.cdf",
            "imap_swapi_lookup_20250417_v001.cdf",
            "kernel_1",
            "kernel_2",
        ]
        actual_parent_file_names = processor.get_parent_file_names()

        self.assertEqual(2, mock_spiceypy.kdata.call_count)
        self.assertEqual(expected_parent_file_names, actual_parent_file_names)

    @patch('imap_l3_processing.processor.spiceypy')
    def test_get_parent_files_takes_optional_parameter(self, mock_spiceypy):
        file_paths = [Path('path')]
        mock_spiceypy.ktotal.return_value = 0

        processor = SwapiProcessor(Mock(), Mock())
        actual_parent_file_names = processor.get_parent_file_names(file_paths)

        self.assertEqual(['path'], actual_parent_file_names)


if __name__ == '__main__':
    unittest.main()
