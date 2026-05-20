import os
import subprocess
import sys
import unittest
from datetime import timedelta
from pathlib import Path

from imap_data_access import (
    ScienceFilePath,
)

import imap_l3_processing
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, get_test_data_path, get_integration_test_data_path, \
    run_periodically

SWE_OUTPUT_DATA_DIR = get_run_local_data_path("swe_integration")


class SweProcessorIntegration(unittest.TestCase):
    @run_periodically(timedelta(days=7))
    def test_swe_processor_with_local_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)

        swe_test_data_dir = get_test_data_path("swe")
        spice_test_data_dir = get_integration_test_data_path("spice")
        input_files = [
            swe_test_data_dir / "imap_swe_l3_sci-9ad0ccff_20260120_v009.json",
            swe_test_data_dir / "imap_swe_l1b_sci_20260120_v015.cdf",
            swe_test_data_dir / "imap_swe_l2_sci_20260120_v015.cdf",
            swe_test_data_dir / "imap_swe_config_20251119_v002.json",
            swe_test_data_dir / "imap_swapi_l3a_proton-sw_20260120_v004.cdf",
            swe_test_data_dir / "imap_mag_l2_norm-dsrf_20260120_v003.cdf",
            spice_test_data_dir / "naif0012.tls",
            spice_test_data_dir / "pck00011.tpc",
            spice_test_data_dir / "imap_130.tf",
            spice_test_data_dir / "imap_science_120.tf",
            spice_test_data_dir / "imap_sclk_0171.tsc",
            spice_test_data_dir / "de440.bsp",
            spice_test_data_dir / "imap_recon_20250925_20260511_v01.bsp",
            spice_test_data_dir / "imap_dps_2025_359_2026_131_002.ah.bc"
        ]

        with mock_imap_data_access(SWE_OUTPUT_DATA_DIR, input_files):
            result = subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "swe",
                    "--data-level",
                    "l3",
                    "--descriptor",
                    "sci",
                    "--start-date",
                    "20260120",
                    "--version",
                    "v001",
                    "--dependency",
                    "imap_swe_l3_sci-9ad0ccff_20260120_v009.json",
                ],
            )

            expected_file_path = ScienceFilePath("imap_swe_l3_sci_20260120_v001.cdf").construct_path()

            self.assertEqual(0, result.returncode)
            self.assertTrue(expected_file_path.exists())

