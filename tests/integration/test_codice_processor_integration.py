import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from imap_data_access import ProcessingInputCollection, ScienceInput, AncillaryInput
from imap_data_access.file_validation import ScienceFilePath
from spacepy.pycdf import CDF

import imap_l3_data_processor
import imap_l3_processing
import tests
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, get_test_data_path


class CodiceProcessorIntegration(unittest.TestCase):
    TEST_DATA_DIR = Path(tests.integration.__file__).parent / 'test_data/codice'
    OUTPUT_DATA_DIR = get_run_local_data_path('codice_lo_integration')

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_codice_processor_integration_lo_direct_events(self, mock_parse_cli_arguments):
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

    def test_codice_lo_partial_densities_and_sw_products(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        OUTPUT_DATA_DIR = get_run_local_data_path("codice_integration")
        expected_pd_file_path = (
            OUTPUT_DATA_DIR / "imap/codice/l3a/2026/03/imap_codice_l3a_lo-partial-densities_20260301_v001.cdf"
        )
        expected_sw_ratios_file_path = (
                OUTPUT_DATA_DIR / "imap/codice/l3a/2026/03/imap_codice_l3a_lo-sw-ratios_20260301_v001.cdf"
        )
        expected_sw_csd_file_path = (
                OUTPUT_DATA_DIR / "imap/codice/l3a/2026/03/imap_codice_l3a_lo-sw-charge-state-distributions_20260301_v001.cdf"
        )

        if expected_pd_file_path.parent.exists():
            expected_pd_file_path.unlink(missing_ok=True)
            expected_sw_ratios_file_path.unlink(missing_ok=True)
            expected_sw_csd_file_path.unlink(missing_ok=True)

        input_files = [
            Path("tests/integration/test_data/codice/imap_codice_l2_lo-sw-species_20260301_v001.cdf"),
            Path("tests/integration/test_data/codice/imap_codice_mass-per-charge_20241110_v003.csv"),
            Path("tests/integration/test_data/codice/imap_codice_l3a_lo-partial-densities-25ccf871_20260301_v001.json"),
            Path("tests/integration/test_data/codice/imap_codice_l3a_lo-sw-ratios-25ccf871_20260301_v001.json"),
        ]
        os.environ["IMAP_DATA_DIR"] = str(OUTPUT_DATA_DIR)
        with mock_imap_data_access(OUTPUT_DATA_DIR, input_files):
            pd_result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-partial-densities",
                    "--start-date",
                    "20260301",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_codice_l3a_lo-partial-densities-25ccf871_20260301_v001.json",
                ],
            )

            self.assertEqual(0, pd_result.returncode)
            self.assertTrue(expected_pd_file_path.exists())

            sw_ratios_result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-sw-ratios",
                    "--start-date",
                    "20260301",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_codice_l3a_lo-sw-ratios-25ccf871_20260301_v001.json",
                ],
            )

            self.assertEqual(0, sw_ratios_result.returncode)
            self.assertTrue(expected_sw_ratios_file_path.exists())

            sw_csd_result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-sw-charge-state-distributions",
                    "--start-date",
                    "20260301",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_codice_l3a_lo-sw-ratios-25ccf871_20260301_v001.json",
                ],
            )

            self.assertEqual(0, sw_csd_result.returncode)
            self.assertTrue(expected_sw_csd_file_path.exists())

    def test_codice_hi_direct_events(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        OUTPUT_DATA_DIR = get_run_local_data_path("codice_integration")
        expected_output_path = (
            OUTPUT_DATA_DIR / "imap/codice/l3a/2026/03/imap_codice_l3a_hi-direct-events_20260301_v001.cdf"
        )
        if expected_output_path.parent.exists():
            expected_output_path.unlink(missing_ok=True)

        input_files = [
            Path("tests/integration/test_data/codice/imap_codice_l2_hi-direct-events_20260301_v003.cdf"),
            Path("tests/integration/test_data/codice/imap_codice_l3a_hi-direct-events-e968219e_20260301_v001.json"),
        ]
        os.environ["IMAP_DATA_DIR"] = str(OUTPUT_DATA_DIR)
        with mock_imap_data_access(OUTPUT_DATA_DIR, input_files):
            result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "hi-direct-events",
                    "--start-date",
                    "20260301",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_codice_l3a_hi-direct-events-e968219e_20260301_v001.json",
                ],
            )

            self.assertEqual(0, result.returncode)
            self.assertTrue(expected_output_path.exists())

    def test_codice_hi_pitch_angle(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        OUTPUT_DATA_DIR = get_run_local_data_path("codice_integration")
        expected_output_path = (
                OUTPUT_DATA_DIR / "imap/codice/l3b/2026/01/imap_codice_l3b_hi-pitch-angle_20260120_v001.cdf"
        )
        if expected_output_path.parent.exists():
            expected_output_path.unlink(missing_ok=True)

        input_files = [
            Path("tests/integration/test_data/codice/imap_codice_l2_hi-sectored_20260120_v003.cdf"),
            Path("tests/integration/test_data/codice/imap_mag_l1d_norm-dsrf_20260120_v002.cdf"),
            Path(
                "tests/integration/test_data/codice/imap_codice_l3b_hi-pitch-angle-25ccf871_20260120_v001.json"),
        ]
        os.environ["IMAP_DATA_DIR"] = str(OUTPUT_DATA_DIR)
        with mock_imap_data_access(OUTPUT_DATA_DIR, input_files):
            result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3b",
                    "--descriptor",
                    "hi-pitch-angle",
                    "--start-date",
                    "20260120",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_codice_l3b_hi-pitch-angle-25ccf871_20260120_v001.json",
                ],
            )

            self.assertEqual(0, result.returncode)
            self.assertTrue(expected_output_path.exists())
