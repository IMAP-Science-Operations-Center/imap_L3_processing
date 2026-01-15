import unittest
from unittest.mock import patch

from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection

from imap_l3_processing.lo.l3.lo_l3_isn_background_subtracted_dependencies import \
    LoL3ISNBackgroundSubtractedDependencies


class TestLoL3ISNBackgroundSubtractedDependencies(unittest.TestCase):

    @patch("imap_l3_processing.lo.l3.lo_l3_isn_background_subtracted_dependencies.ISNRateData.read_from_path")
    @patch("imap_l3_processing.lo.l3.lo_l3_isn_background_subtracted_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_read_from_path):
        file_name = "imap_lo_l2_l090-isnnbkgnd-h-sf-nsp-ram-hae-6deg-1yr_20250422_v001.cdf"
        input = ScienceInput(file_name)
        processing_input_collection = ProcessingInputCollection(input)

        dependencies = LoL3ISNBackgroundSubtractedDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_called_once_with(file_name)

        self.assertEqual(dependencies.map_data, mock_read_from_path.return_value)
