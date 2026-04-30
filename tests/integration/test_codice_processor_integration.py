import logging
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from imap_data_access import ProcessingInputCollection
from imap_data_access.file_validation import ScienceFilePath
from imap_data_access.processing_input import generate_imap_input
from spacepy.pycdf import CDF

import imap_l3_data_processor
import tests
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, get_test_data_path, run_periodically

INTEGRATION_DATA_DIR = Path(tests.integration.__file__).parent / "test_data/codice"
CODICE_TEST_DATA_DIR = get_test_data_path("codice")
OUTPUT_DIR = get_run_local_data_path("codice_integration")


class CodiceProcessorIntegration(unittest.TestCase):
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_lo_direct_events(self, mock_parse_cli_arguments):
        energy_per_charge_path = CODICE_TEST_DATA_DIR / "imap_codice_lo-energy-per-charge_20241110_v002.csv"
        mass_coefficient_path = CODICE_TEST_DATA_DIR / "imap_codice_mass-coefficient-lookup_20241110_v003.csv"
        input_files = [
            CODICE_TEST_DATA_DIR / "imap_codice_l2_lo-direct-events_20260307_v004.cdf",
            CODICE_TEST_DATA_DIR / "imap_codice_l1a_lo-nsw-priority_20260307_v004.cdf",
            CODICE_TEST_DATA_DIR / "imap_codice_l1a_lo-sw-priority_20260307_v004.cdf",
            energy_per_charge_path,
            mass_coefficient_path,
        ]

        with mock_imap_data_access(OUTPUT_DIR, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            )

            processing_input_collection = ProcessingInputCollection(*(generate_imap_input(f.name) for f in input_files))

            mock_arguments = Mock()
            mock_arguments.instrument = "codice"
            mock_arguments.data_level = "l3a"
            mock_arguments.descriptor = "lo-direct-events"
            mock_arguments.start_date = "20260307"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = processing_input_collection.serialize()
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_output_path = ScienceFilePath(
                "imap_codice_l3a_lo-direct-events_20260307_v001.cdf").construct_path()
            self.assertTrue(expected_output_path.exists(), f"Expected file {expected_output_path.name} not found")

            expected_parents = {
                "imap_codice_l2_lo-direct-events_20260307_v004.cdf",
                "imap_codice_l1a_lo-nsw-priority_20260307_v004.cdf",
                "imap_codice_l1a_lo-sw-priority_20260307_v004.cdf",
                "imap_codice_lo-energy-per-charge_20241110_v002.csv",
                "imap_codice_mass-coefficient-lookup_20241110_v003.csv",
            }

            with CDF(str(expected_output_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_lo_partial_densities_and_sw_products(self, mock_parse_cli_arguments):
        input_files = [
            INTEGRATION_DATA_DIR / "imap_codice_l2_lo-sw-species_20260301_v001.cdf",
            INTEGRATION_DATA_DIR / "imap_codice_mass-per-charge_20241110_v003.csv",
            INTEGRATION_DATA_DIR / "imap_codice_l3a_lo-partial-densities-25ccf871_20260301_v001.json",
            INTEGRATION_DATA_DIR / "imap_codice_l3a_lo-sw-ratios-25ccf871_20260301_v001.json",
        ]
        with mock_imap_data_access(OUTPUT_DIR, input_files):
            def run_processor(descriptor: str, dependency_file: str):
                mock_arguments = Mock()
                mock_arguments.instrument = "codice"
                mock_arguments.data_level = "l3a"
                mock_arguments.descriptor = descriptor
                mock_arguments.start_date = "20260301"
                mock_arguments.end_date = None
                mock_arguments.repointing = None
                mock_arguments.version = "v001"
                mock_arguments.dependency = dependency_file
                mock_arguments.upload_to_sdc = False
                mock_parse_cli_arguments.return_value = mock_arguments
                imap_l3_data_processor.imap_l3_processor()

            run_processor(
                "lo-partial-densities",
                "imap_codice_l3a_lo-partial-densities-25ccf871_20260301_v001.json",
            )
            expected_pd_file_path = ScienceFilePath(
                "imap_codice_l3a_lo-partial-densities_20260301_v001.cdf"
            ).construct_path()
            self.assertTrue(expected_pd_file_path.exists())

            run_processor(
                "lo-sw-ratios",
                "imap_codice_l3a_lo-sw-ratios-25ccf871_20260301_v001.json",
            )
            expected_sw_ratios_file_path = ScienceFilePath(
                "imap_codice_l3a_lo-sw-ratios_20260301_v001.cdf"
            ).construct_path()
            self.assertTrue(expected_sw_ratios_file_path.exists())

            run_processor(
                "lo-sw-charge-state-distributions",
                "imap_codice_l3a_lo-sw-ratios-25ccf871_20260301_v001.json",
            )
            expected_sw_csd_file_path = ScienceFilePath(
                "imap_codice_l3a_lo-sw-charge-state-distributions_20260301_v001.cdf"
            ).construct_path()
            self.assertTrue(expected_sw_csd_file_path.exists())

    @run_periodically(timedelta(days=14))
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_lo_3d_distributions(self, mock_parse_cli_arguments):
        input_files = [
            CODICE_TEST_DATA_DIR / "imap_codice_l3a_lo-direct-events_20260307_v001.cdf",
            CODICE_TEST_DATA_DIR / "imap_codice_l1a_lo-nsw-priority_20260307_v004.cdf",
            CODICE_TEST_DATA_DIR / "imap_codice_l1a_lo-sw-priority_20260307_v004.cdf",
            CODICE_TEST_DATA_DIR / "imap_codice_lo-energy-per-charge_20241110_v002.csv",
            CODICE_TEST_DATA_DIR / "imap_codice_l2-lo-efficiency_20251008_v003.csv",
            CODICE_TEST_DATA_DIR / "imap_codice_l2-lo-gfactor_20251212_v003.csv",
            CODICE_TEST_DATA_DIR / "imap_codice_lo-mass-species-bin-lookup_20250309_v003.csv",
        ]

        processing_input = ProcessingInputCollection(*(generate_imap_input(f.name) for f in input_files))
        dependency_json = processing_input.serialize()

        with mock_imap_data_access(OUTPUT_DIR, input_files):
            for species in ("hplus", "heplus", "heplusplus", "oplus6"):
                with self.subTest(species=species):
                    descriptor = f"lo-{species}-3d-distribution"

                    mock_arguments = Mock()
                    mock_arguments.instrument = "codice"
                    mock_arguments.data_level = "l3a"
                    mock_arguments.descriptor = descriptor
                    mock_arguments.start_date = "20260307"
                    mock_arguments.end_date = None
                    mock_arguments.repointing = None
                    mock_arguments.version = "v001"
                    mock_arguments.dependency = dependency_json
                    mock_arguments.upload_to_sdc = False
                    mock_parse_cli_arguments.return_value = mock_arguments
                    imap_l3_data_processor.imap_l3_processor()

                    expected_output_path = ScienceFilePath(
                        f"imap_codice_l3a_{descriptor}_20260307_v001.cdf"
                    ).construct_path()
                    self.assertTrue(expected_output_path.exists())

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_hi_direct_events(self, mock_parse_cli_arguments):
        input_files = [
            INTEGRATION_DATA_DIR / "imap_codice_l2_hi-direct-events_20260301_v003.cdf",
            INTEGRATION_DATA_DIR / "imap_codice_l3a_hi-direct-events-e968219e_20260301_v001.json",
        ]
        with mock_imap_data_access(OUTPUT_DIR, input_files):
            mock_arguments = Mock()
            mock_arguments.instrument = "codice"
            mock_arguments.data_level = "l3a"
            mock_arguments.descriptor = "hi-direct-events"
            mock_arguments.start_date = "20260301"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = "imap_codice_l3a_hi-direct-events-e968219e_20260301_v001.json"
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_output_path = ScienceFilePath(
                "imap_codice_l3a_hi-direct-events_20260301_v001.cdf"
            ).construct_path()
            self.assertTrue(expected_output_path.exists())

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_hi_pitch_angle(self, mock_parse_cli_arguments):
        input_files = [
            INTEGRATION_DATA_DIR / "imap_codice_l2_hi-sectored_20260120_v003.cdf",
            INTEGRATION_DATA_DIR / "imap_mag_l1d_norm-dsrf_20260120_v002.cdf",
            INTEGRATION_DATA_DIR / "imap_mag_l2_norm-dsrf_20260120_v002.cdf",
        ]
        with mock_imap_data_access(OUTPUT_DIR, input_files):
            mock_arguments = Mock()
            mock_arguments.instrument = "codice"
            mock_arguments.data_level = "l3b"
            mock_arguments.descriptor = "hi-pitch-angle"
            mock_arguments.start_date = "20260120"
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            processing_input = ProcessingInputCollection(*[generate_imap_input(f.name) for f in input_files])
            mock_arguments.dependency = processing_input.serialize()
            mock_arguments.upload_to_sdc = False
            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_output_path = ScienceFilePath("imap_codice_l3b_hi-pitch-angle_20260120_v001.cdf").construct_path()
            self.assertTrue(expected_output_path.exists())
