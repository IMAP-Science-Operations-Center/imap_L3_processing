import unittest
from datetime import datetime
from unittest.mock import patch, sentinel

from imap_l3_processing.codice.codice_processor import CodiceProcessor
from imap_l3_processing.models import InputMetadata


class TestCodiceProcessor(unittest.TestCase):

    @patch("imap_l3_processing.codice.codice_processor.CodiceL3Dependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.codice_processor.CodiceProcessor.process_l3a")
    def test_process_l3a(self, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3a", start_date, end_date, 'v02')

        processor = CodiceProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(sentinel.processing_input_collection)
        mock_process_l3a.assert_called_with(mock_fetch_dependencies.return_value)

    @patch("imap_l3_processing.codice.codice_processor.CodiceL3Dependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.codice_processor.CodiceProcessor.process_l3a")
    def test_ignores_non_l3_input_metadata(self, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l2a", start_date, end_date, 'v02')

        processor = CodiceProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_not_called()
        mock_process_l3a.assert_not_called()
