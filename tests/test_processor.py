import unittest
from unittest.mock import Mock, patch

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.processor import Processor


class TestProcessor(unittest.TestCase):
    @patch('imap_l3_processing.processor.spiceypy')
    def test_get_parent_file_names(self, mock_spicepy):
        input_collection = ProcessingInputCollection(ScienceInput("imap_swapi_l2_science_20250417_v001.cdf"),
                                                     ScienceInput("imap_swapi_l2_science_20250418_v001.cdf"),
                                                     AncillaryInput("imap_swapi_lookup_20250417_v001.cdf"))
        mock_spicepy.ktotal.return_value = 2
        mock_spicepy.kdata.side_effect = [['kernel_1', 'type', 'source', 'handle'],
                                          ['kernel_2', 'type', 'source', 'handle']]

        processor = Processor(input_collection, Mock())

        expected_parent_file_names = [
            "imap_swapi_l2_science_20250417_v001.cdf",
            "imap_swapi_l2_science_20250418_v001.cdf",
            "imap_swapi_lookup_20250417_v001.cdf",
            "kernel_1",
            "kernel_2",
        ]
        actual_parent_file_names = processor.get_parent_file_names()

        self.assertEqual(2, mock_spicepy.kdata.call_count)
        self.assertEqual(expected_parent_file_names, actual_parent_file_names)


if __name__ == '__main__':
    unittest.main()
