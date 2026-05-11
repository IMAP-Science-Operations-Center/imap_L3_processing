import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

import imap_data_access
import numpy as np
from imap_data_access import (
    ScienceFilePath,
)
from spacepy.pycdf import CDF

import imap_l3_processing
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_test_data_path, get_run_local_data_path

class SweProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_swe_processor_with_production_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"
        expected_file_path = ScienceFilePath(
            "imap_swe_l3_sci_20260120_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)
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
                "imap_swe_l3_sci-e979d33c_20260120_v001.json",
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

    def test_swe_processor_with_local_data(self):
        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        OUTPUT_DATA_DIR = get_run_local_data_path("swe_integration")
        expected_file_path = (
            OUTPUT_DATA_DIR / "imap/swe/l3/2026/01/imap_swe_l3_sci_20260120_v001.cdf"
        )
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory() as augmented_inputs_dir:
            input_files = []
            for file in Path("tests/integration/test_data/swe").iterdir():
                if file.is_file():
                    staged = Path(augmented_inputs_dir, file.name)
                    shutil.copyfile(file, staged)
                    if file.name.startswith("imap_swapi_l3a_proton-sw_"):
                        _augment_swapi_fixture_with_rtn_velocity(staged)
                    input_files.append(staged)
            os.environ["IMAP_DATA_DIR"] = str(OUTPUT_DATA_DIR)
            with mock_imap_data_access(OUTPUT_DATA_DIR, input_files):
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
                        "imap_swe_l3_sci-e979d33c_20260120_v001.json",
                    ],
                )

                self.assertEqual(0, result.returncode)
                self.assertTrue(expected_file_path.exists())


def _augment_swapi_fixture_with_rtn_velocity(swapi_cdf_path: Path) -> None:
    """Inject `proton_sw_bulk_velocity_rtn_sc` into a copied SWAPI L3a CDF.

    The committed integration fixture predates the variable; tests build a
    temp-dir copy and augment that copy so the subprocess SWE processor sees
    the field it now requires."""
    with CDF(str(swapi_cdf_path), readonly=False) as cdf:
        if 'proton_sw_bulk_velocity_rtn_sc' in cdf:
            return
        speed = cdf['proton_sw_speed'][:].astype(float)
        velocity_rtn = np.stack(
            [-speed, np.zeros_like(speed), np.zeros_like(speed)], axis=-1
        ).astype(np.float32)
        cdf.new('proton_sw_bulk_velocity_rtn_sc', data=velocity_rtn)
        cdf['proton_sw_bulk_velocity_rtn_sc'].attrs['FILLVAL'] = np.float32(-1e31)
        cdf['proton_sw_bulk_velocity_rtn_sc'].attrs['UNITS'] = 'km/s'

