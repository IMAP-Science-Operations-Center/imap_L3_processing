"""End-to-end subprocess integration tests for the SWAPI L3a processors.

Mirrors test_swe_processor_integration.py: runs ``imap_l3_data_processor.py`` as
a subprocess against staged test data, exercising the full real path —
dependency manifest deserialization → SPICE furnishing → SwapiL3ADependencies
loading of all 13 ancillaries → SwapiProcessor.process() → process_l3a_*
→ save_data → CDF written to disk.

SWAPI-specific inputs live in ``tests/integration/test_data/swapi/``. SPICE
kernels are pulled from the shared ``tests/integration/test_data/spice/`` dir.
The L2 science CDF is generated on the fly by retiming an existing synthetic
spectrum into the SPICE coverage window — keeping a date-shifted copy on disk
would just be duplicate data.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import skipUnless

import imap_data_access
import numpy.testing
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF
import datetime

import imap_l3_processing
from tests.integration.integration_test_helpers import stage_input_file

SWAPI_INTEGRATION_DATA_DIR = Path(__file__).parent / "test_data" / "swapi"


class SwapiProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_proton_sw_with_production_data(self):
        """
        With real data and with full dependency setup, validate that the CDF output
        matches hardcoded expected values to check for unexpected changes
        """

        expected_values = {
            'epoch': datetime.datetime(2025, 12, 31, 23, 59, 35, 6000),
            'proton_sw_speed': 474.57455,
            'proton_sw_speed_uncert': 0.36071813,
            'proton_sw_speed_sun': 475.3688,
            'proton_sw_speed_sun_uncert': 0.3539945,
            'epoch_delta': 30000000000,
            'proton_sw_temperature': 55131.13,
            'proton_sw_temperature_uncert': 2008.7198,
            'proton_sw_density': 2.6919234,
            'proton_sw_density_uncert': 0.049627114,
            'proton_sw_bulk_velocity_rtn_sun': [474.0841, 30.623661, 16.79102],
            'proton_sw_bulk_velocity_rtn_sc': [474.14252, 1.0705476, 20.217045],
            'proton_sw_bulk_velocity_rtn_covariance': [[0.1169681, -0.0341877, 0.13108876],
                                                           [-0.0341877, 0.8606747, -0.23923519],
                                                           [0.13108876, -0.23923519, 1.3221142]],
            'swp_flags': 0
        }

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"

        dependency_filename = "imap_swapi_l3a_proton-sw_20260101_v001.json"
        stage_input_file(SWAPI_INTEGRATION_DATA_DIR / dependency_filename)

        expected_file_path = ScienceFilePath(
            "imap_swapi_l3a_proton-sw_20260101_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument", "swapi",
                "--data-level", "l3a",
                "--descriptor", "proton-sw",
                "--start-date", "20260101",
                "--version", "v001",
                "--dependency", dependency_filename,
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())


        bulk_speed_atol = 1e-3 * float(expected_values['proton_sw_speed'])

        with CDF(str(expected_file_path)) as cdf:
            for key in expected_values.keys():
                actual_value = cdf[key][0]
                if 'bulk_velocity' in key:
                    rtol, atol = 1e-3, bulk_speed_atol
                elif key.endswith('_uncert'):
                    rtol, atol = 1e-2, 0.0
                else:
                    rtol, atol = 1e-3, 0.0
                try:
                    numpy.testing.assert_allclose(
                        actual_value, expected_values[key], rtol=rtol, atol=atol, err_msg=key
                    )
                except TypeError:
                    self.assertEqual(expected_values[key], actual_value, msg=key)



    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_alpha_sw_with_production_data(self):
        """
        With real data and with full dependency setup, validate that the alpha-sw
        CDF output matches hardcoded expected values to check for unexpected changes.
        Unlike proton-sw, alpha-sw requires a MAG RTN science input (L2 preferred,
        L1D fallback) for the field-aligned drift constraint.
        """

        # Record index 0 for this date is flagged (swp_flags=4) so every alpha
        # quantity is fill; the first valid record is index 1.
        sample_index = 1
        expected_values = {
            'epoch': datetime.datetime(2026, 1, 1, 0, 0, 35, 6000),
            'epoch_delta': 30000000000,
            'alpha_sw_speed': 473.3955993652344,
            'alpha_sw_speed_uncert': 0.3668617010116577,
            'alpha_sw_density': 0.1588851362466812,
            'alpha_sw_density_uncert': 0.2096901834011078,
            'alpha_sw_temperature': 2332716.75,
            'alpha_sw_temperature_uncert': 1120345.5,
            'alpha_sw_velocity_rtn_sc': [472.94376, -10.725992, 17.678875],
            'alpha_sw_velocity_rtn_covariance': [
                [0.13366441, 0.22170407, 0.13530357],
                [0.22170407, 13.252416, 6.8162704],
                [0.13530357, 6.8162704, 4.195147],
            ],
            'swp_flags': 0,
        }

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"

        dependency_filename = "imap_swapi_l3a_alpha-sw_20260101_v001.json"
        stage_input_file(SWAPI_INTEGRATION_DATA_DIR / dependency_filename)

        expected_file_path = ScienceFilePath(
            "imap_swapi_l3a_alpha-sw_20260101_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument", "swapi",
                "--data-level", "l3a",
                "--descriptor", "alpha-sw",
                "--start-date", "20260101",
                "--version", "v001",
                "--dependency", dependency_filename,
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

        bulk_speed_atol = 1e-3 * float(expected_values['alpha_sw_speed'])

        with CDF(str(expected_file_path)) as cdf:
            for key in expected_values.keys():
                actual_value = cdf[key][sample_index]
                if 'velocity' in key:
                    rtol, atol = 1e-3, bulk_speed_atol
                elif key.endswith('_uncert'):
                    rtol, atol = 1e-2, 0.0
                else:
                    rtol, atol = 1e-3, 0.0
                try:
                    numpy.testing.assert_allclose(
                        actual_value, expected_values[key], rtol=rtol, atol=atol, err_msg=key
                    )
                except TypeError:
                    self.assertEqual(expected_values[key], actual_value, msg=key)


    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_pui_he_with_production_data(self):
        """
        With real data and with full dependency setup, validate that the pui-he
        CDF output matches hardcoded expected values to check for unexpected
        changes.
        """

        sample_index = 98
        expected_values = {
            'epoch': datetime.datetime(2026, 1, 1, 16, 24, 4, 954000),
            'epoch_delta': 300000000000,
            'pui_cooling_index': 1.9189398,
            'pui_cooling_index_uncert': 0.20612726,
            'pui_ionization_rate': 7.7820943e-08,
            'pui_ionization_rate_uncert': 4.0720134e-09,
            'pui_cutoff_speed': 481.61884,
            'pui_cutoff_speed_uncert': 3.2015524,
            'pui_background_count_rate': 0.50305182,
            'pui_background_count_rate_uncert': 0.19893467,
            'pui_density': 5.246582e-04,
            'pui_density_uncert': 2.8316077e-05,
            'pui_temperature': 21426046.0,
            'pui_temperature_uncert': 838306.625,
            'swp_flags': 0,
        }

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        imap_data_access.config["DATA_DIR"] = root_dir / "data"

        dependency_filename = "imap_swapi_l3a_pui-he_20260101_v001.json"
        stage_input_file(SWAPI_INTEGRATION_DATA_DIR / dependency_filename)

        expected_file_path = ScienceFilePath(
            "imap_swapi_l3a_pui-he_20260101_v001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument", "swapi",
                "--data-level", "l3a",
                "--descriptor", "pui-he",
                "--start-date", "20260101",
                "--version", "v001",
                "--dependency", dependency_filename,
            ]
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

        with CDF(str(expected_file_path)) as cdf:
            for key in expected_values.keys():
                actual_value = cdf[key][sample_index]
                if key.endswith('_uncert'):
                    rtol, atol = 1e-2, 0.0
                else:
                    rtol, atol = 1e-3, 0.0
                try:
                    numpy.testing.assert_allclose(
                        actual_value, expected_values[key], rtol=rtol, atol=atol, err_msg=key
                    )
                except TypeError:
                    self.assertEqual(expected_values[key], actual_value, msg=key)


if __name__ == "__main__":
    unittest.main()
