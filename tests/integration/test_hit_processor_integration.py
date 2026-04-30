import logging
import unittest
from unittest.mock import patch, Mock

from imap_data_access.processing_input import ProcessingInputCollection, generate_imap_input, ScienceFilePath

import imap_l3_data_processor
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_test_data_path, get_run_local_data_path


class HitProcessorIntegration(unittest.TestCase):
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_hit_macropixel_product(self, mock_parse_cli_arguments):
        input_files = [
            get_test_data_path("hit/imap_hit_l2_macropixel-intensity_20260228_v002.cdf"),
            get_test_data_path("hit/imap_mag_l1d_norm-dsrf_20260228_v005.cdf"),
            get_test_data_path("hit/imap_mag_l2_norm-dsrf_20260228_v002.cdf"),
        ]

        output_path = get_run_local_data_path("hit_integration")

        with mock_imap_data_access(output_path, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            )

            processing_input_collection = ProcessingInputCollection(*(generate_imap_input(f.name) for f in input_files))

            mock_arguments = Mock()
            mock_arguments.instrument = "hit"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = "macropixel"
            mock_arguments.start_date = "20260228"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = processing_input_collection.serialize()
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_output_path = ScienceFilePath(
                "imap_hit_l3_macropixel_20260228_v001.cdf").construct_path()
            self.assertTrue(expected_output_path.exists(), f"Expected file {expected_output_path.name} not found")
