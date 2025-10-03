import logging
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF

import imap_l3_data_processor
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path

INTEGRATION_TEST_DATA_PATH = Path(__file__).parent / "test_data"


class TestMapIntegration(unittest.TestCase):
    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_hi_all_sp_maps(self, mock_parse_cli_arguments):
        hi_test_data_dir = INTEGRATION_TEST_DATA_PATH / "hi"
        hi_imap_data_dir = get_run_local_data_path("hi/integration_data")

        input_files = [
            hi_test_data_dir / "imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v006.cdf",
            hi_test_data_dir / "imap_hi_l1c_90sensor-pset_20250415-repoint01000_v001.cdf",
            hi_test_data_dir / "imap_glows_l3e_survival-probability-hi-45_20250415-repoint01000_v001.cdf",
            hi_test_data_dir / "imap_glows_l3e_survival-probability-hi-45_20260418-repoint02000_v001.cdf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc"
        ]

        with mock_imap_data_access(hi_imap_data_dir, input_files):
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

            expected_map_path = ScienceFilePath(
                'imap_hi_l3_h45-ena-h-sf-sp-ram-hae-4deg-1yr_20250415_v001.cdf').construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {
                "imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-1yr_20250415_v006.cdf",
                "imap_hi_l1c_90sensor-pset_20250415-repoint01000_v001.cdf",
                "imap_glows_l3e_survival-probability-hi-45_20250415-repoint01000_v001.cdf",
            }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_lo_all_sp_maps(self, mock_parse_cli_arguments):
        lo_test_data_dir = INTEGRATION_TEST_DATA_PATH / "lo"
        lo_imap_data_dir = get_run_local_data_path("lo/integration_data")

        input_files = [
            lo_test_data_dir / "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20250415_v006.cdf",
            lo_test_data_dir / "imap_lo_l1c_pset_20250415-repoint01000_v001.cdf",

            lo_test_data_dir / "imap_glows_l3e_survival-probability-lo_20250415-repoint01000_v001.cdf",
            lo_test_data_dir / "imap_glows_l3e_survival-probability-lo_20260418-repoint02003_v001.cdf",

            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc"
        ]

        with mock_imap_data_access(lo_imap_data_dir, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "lo"
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

            expected_map_path = ScienceFilePath(
                "imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-12mo_20250415_v001.cdf").construct_path()
            self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")

            expected_parents = {
                "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20250415_v006.cdf",
                "imap_lo_l1c_pset_20250415-repoint01000_v001.cdf",
                "imap_glows_l3e_survival-probability-lo_20250415-repoint01000_v001.cdf",
            }

            with CDF(str(expected_map_path)) as cdf:
                self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_ultra_all_sp_maps(self, mock_parse_cli_arguments):
        ultra_test_data_dir = INTEGRATION_TEST_DATA_PATH / "ultra"
        ultra_imap_data_dir = get_run_local_data_path("ultra/integration_data")

        input_files = [
            ultra_test_data_dir / 'imap_glows_l3e_survival-probability-ul_20250415-repoint00001_v010.cdf',
            ultra_test_data_dir / 'imap_glows_l3e_survival-probability-ul_20261020-repoint00100_v010.cdf',
            ultra_test_data_dir / 'imap_ultra_l1c_45sensor-spacecraftpset_20250415-repoint00001_v010.cdf',
            ultra_test_data_dir / 'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-6mo_20250415_v010.cdf',

            INTEGRATION_TEST_DATA_PATH / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA_PATH / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc"
        ]

        with mock_imap_data_access(ultra_imap_data_dir, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "ultra"
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

            # expected_map_path = ScienceFilePath(
            #     "imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-12mo_20250415_v001.cdf").construct_path()
            # self.assertTrue(expected_map_path.exists(), f"Expected file {expected_map_path.name} not found")
            #
            # expected_parents = {
            #     "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-12mo_20250415_v006.cdf",
            #     "imap_lo_l1c_pset_20250415-repoint01000_v001.cdf",
            #     "imap_glows_l3e_survival-probability-lo_20250415-repoint01000_v001.cdf",
            # }
            #
            # with CDF(str(expected_map_path)) as cdf:
            #     self.assertEqual(expected_parents, set(cdf.attrs["Parents"]))