import logging
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from imap_data_access import ProcessingInputCollection, ScienceInput, AncillaryInput
from imap_data_access.file_validation import ScienceFilePath
from spacepy.pycdf import CDF

import imap_l3_data_processor
import tests
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, get_test_data_path


class CodiceProcessorIntegration(unittest.TestCase):
    TEST_DATA_DIR = Path(tests.integration.__file__).parent / 'test_data/codice'
    OUTPUT_DATA_DIR = get_run_local_data_path('codice_lo_integration')

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_processor_integration(self, mock_parse_cli_arguments):
        energy_per_charge_path = get_test_data_path('codice/imap_codice_lo-energy-per-charge_20241110_v001.csv')
        mass_coefficient_path = get_test_data_path('codice/imap_codice_mass-coefficient-lookup_20241110_v003.csv')
        input_files = [
            self.TEST_DATA_DIR / 'imap_codice_l2_lo-direct-events_20250814_v001.cdf',
            self.TEST_DATA_DIR / 'imap_codice_l1a_lo-nsw-priority_20250814_v001.cdf',
            self.TEST_DATA_DIR / 'imap_codice_l1a_lo-sw-priority_20250814_v001.cdf',
            energy_per_charge_path,
            mass_coefficient_path,
        ]

        with mock_imap_data_access(self.OUTPUT_DATA_DIR, input_files):
            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            processing_input_collection = ProcessingInputCollection(
                ScienceInput('imap_codice_l2_lo-direct-events_20250814_v001.cdf'),
                ScienceInput('imap_codice_l1a_lo-nsw-priority_20250814_v001.cdf'),
                ScienceInput('imap_codice_l1a_lo-sw-priority_20250814_v001.cdf'),
                AncillaryInput(energy_per_charge_path.name),
                AncillaryInput(mass_coefficient_path.name)
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "codice"
            mock_arguments.data_level = "l3a"
            mock_arguments.descriptor = "lo-direct-events"
            mock_arguments.start_date = "20250814"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = processing_input_collection.serialize()
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_map_path = ScienceFilePath(
                'imap_codice_l3a_lo-direct-events_20250814_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {'imap_codice_l2_lo-direct-events_20250814_v001.cdf',
                                'imap_codice_l1a_lo-nsw-priority_20250814_v001.cdf',
                                'imap_codice_l1a_lo-sw-priority_20250814_v001.cdf',
                                'imap_codice_lo-energy-per-charge_20241110_v001.csv',
                                'imap_codice_mass-coefficient-lookup_20241110_v003.csv'}

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))
