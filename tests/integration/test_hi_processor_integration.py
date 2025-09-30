import logging
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from imap_data_access import ScienceFilePath

import imap_l3_data_processor
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path


class TestMapIntegration(unittest.TestCase):
    INTEGRATION_DATA_DIR = get_run_local_data_path("hi/integration_data")
    INPUT_DATA_DIR = Path(__file__).parent / "test_data" / "hi"

    @patch("imap_l3_data_processor._parse_cli_arguments")
    @mock_imap_data_access(INTEGRATION_DATA_DIR, [
        INPUT_DATA_DIR / "imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v006.cdf",
        INPUT_DATA_DIR / "imap_hi_l1c_90sensor-pset_20250415-repoint01000_v001.cdf",
        INPUT_DATA_DIR / "imap_glows_l3e_survival-probability-hi-45_20250415-repoint01000_v001.cdf",
    ])
    def test_all_sp_maps(self, mock_parse_cli_arguments):
        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        mock_arguments = Mock()
        mock_arguments.instrument = "hi"
        mock_arguments.data_level = "l3"
        mock_arguments.descriptor = "all-maps"
        mock_arguments.start_date = "20250415"
        mock_arguments.end_date = None
        mock_arguments.repointing = None
        mock_arguments.version = "v001"
        mock_arguments.dependency = "[]"
        mock_arguments.upload_to_sdc = False

        mock_parse_cli_arguments.return_value = mock_arguments

        imap_l3_data_processor.imap_l3_processor()

        expected_files = [
            ScienceFilePath('imap_hi_l3_h45-ena-h-sf-sp-ram-hae-4deg-1yr_20250415_v001.cdf').construct_path()
        ]

        for expected_file in expected_files:
            self.assertTrue(expected_file.exists(), f"Expected file {expected_file} not found")
