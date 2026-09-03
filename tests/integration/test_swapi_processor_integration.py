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
import numpy as np
import numpy.testing
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF
import datetime

import imap_l3_processing
from imap_l3_processing.swapi.constants import SWAPI_COARSE_SWEEP_BINS
from tests.integration.integration_test_helpers import stage_input_file

SWAPI_INTEGRATION_DATA_DIR = Path(__file__).parent / "test_data" / "swapi"


def assert_velocity_frames_are_physically_consistent(test_case, cdf, prefix, index):
    """Check the transformed vectors against the geometry expected on January 1."""
    for sun_suffix in ("", "_sun"):
        rtn = cdf[f"{prefix}_rtn{sun_suffix}"][index]
        gse = cdf[f"{prefix}_gse{sun_suffix}"][index]
        gsm = cdf[f"{prefix}_gsm{sun_suffix}"][index]
        hae = cdf[f"{prefix}_hae{sun_suffix}"][index]

        # A rotation changes coordinates, not the speed.
        for transformed in (gse, gsm, hae):
            numpy.testing.assert_allclose(
                numpy.linalg.norm(transformed),
                numpy.linalg.norm(rtn),
                rtol=1e-6,
            )

        # GSE +X points from Earth toward the Sun, opposite the predominantly
        # anti-sunward solar-wind flow. GSM is a rotation about that same X axis.
        test_case.assertLess(gse[0], 0)
        test_case.assertGreater(abs(gse[0]), numpy.linalg.norm(gse[1:]))
        numpy.testing.assert_allclose(gsm[0], gse[0], rtol=1e-6)

        # Near January 1, Earth's anti-sunward radial direction in HAE is
        # predominantly +Y (Earth's heliocentric ecliptic longitude is ~100 deg).
        test_case.assertGreater(hae[1], 0)
        test_case.assertGreater(abs(hae[1]), abs(hae[0]))
        test_case.assertGreater(abs(hae[1]), abs(hae[2]))

    rtn_covariance = cdf[f"{prefix}_rtn_covariance"][index]
    for frame in ("gse", "gsm", "hae"):
        covariance = cdf[f"{prefix}_{frame}_covariance"][index]
        numpy.testing.assert_allclose(covariance, covariance.T, rtol=1e-6, atol=1e-6)
        numpy.testing.assert_allclose(
            numpy.linalg.eigvalsh(covariance),
            numpy.linalg.eigvalsh(rtn_covariance),
            rtol=1e-5,
            atol=1e-5,
        )


class SwapiProcessorIntegration(unittest.TestCase):
    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_proton_sw_with_production_data(self):
        """
        With real data and with full dependency setup, validate that the CDF output
        matches hardcoded expected values to check for unexpected changes
        """

        expected_values = {
            'epoch': datetime.datetime(2025, 12, 31, 23, 59, 35, 6000),
            'proton_sw_speed': 474.578857421875,
            'proton_sw_speed_uncert': 0.36081013083457947,
            'proton_sw_speed_sun': 475.4219055175781,
            'proton_sw_speed_sun_uncert': 0.3551594018936157,
            'epoch_delta': 30000000000,
            'proton_sw_temperature': 55131.46875,
            'proton_sw_temperature_uncert': 2007.7718505859375,
            'proton_sw_density': 2.6932971477508545,
            'proton_sw_density_uncert': 0.049651436507701874,
            'proton_sw_velocity_rtn_sun': [474.1429748535156, 31.2506103515625, 15.421923637390137],
            'proton_sw_velocity_rtn': [474.2013854980469, 1.6974985599517822, 18.847949981689453],
            'proton_sw_velocity_gse': [-474.21169142, -0.48161806, 18.65798744],
            'proton_sw_velocity_gse_sun': [-474.09292141, -30.23247255, 18.65198256],
            'proton_sw_velocity_gsm': [-474.21169142, -5.07181642, 17.96187990],
            'proton_sw_velocity_gsm_sun': [-474.09292141, -33.90076411, 10.61306001],
            'proton_sw_velocity_hae': [-84.52581716, 466.61913283, 18.62932749],
            'proton_sw_velocity_hae_sun': [-113.78456481, 461.22903633, 18.62379766],
            'proton_sw_velocity_rtn_covariance': [[0.11774054169654846, -0.029444590210914612, 0.13579009473323822],
                                                           [-0.029444590210914612, 0.8430156111717224, -0.22263257205486298],
                                                           [0.13579009473323822, -0.22263257205486298, 1.3417274951934814]],
            'proton_sw_velocity_rtn_uncert': [0.34313341675877107, 0.9181588158764923, 1.1583296142262276],
            'swp_flags': 0
        }

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        data_dir = root_dir / "data"
        imap_data_access.config["DATA_DIR"] = data_dir

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
            ],
            env={**os.environ, "IMAP_DATA_DIR": str(data_dir)},
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())


        bulk_speed_atol = 1e-3 * float(expected_values['proton_sw_speed'])

        with CDF(str(expected_file_path)) as cdf:
            for key in expected_values.keys():
                actual_value = cdf[key][0]
                if key.endswith('_uncert'):
                    rtol, atol = 1e-2, 0.0
                elif 'velocity' in key:
                    rtol, atol = 1e-3, bulk_speed_atol
                else:
                    rtol, atol = 1e-3, 0.0
                try:
                    numpy.testing.assert_allclose(
                        actual_value, expected_values[key], rtol=rtol, atol=atol, err_msg=key
                    )
                except TypeError:
                    self.assertEqual(expected_values[key], actual_value, msg=key)
            assert_velocity_frames_are_physically_consistent(
                self, cdf, "proton_sw_velocity", 0
            )



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
            'alpha_sw_speed': 473.3949279785156,
            'alpha_sw_speed_uncert': 0.3672780692577362,
            'alpha_sw_speed_sun': 473.53363037109375,
            'alpha_sw_speed_sun_uncert': 0.4522981643676758,
            'alpha_sw_density': 0.1577976644039154,
            'alpha_sw_density_uncert': 0.2121371477842331,
            'alpha_sw_temperature': 2338303.75,
            'alpha_sw_temperature_uncert': 1141809.75,
            'alpha_sw_velocity_rtn': [473.010009765625, -9.928609848022461, 16.30108070373535],
            'alpha_sw_velocity_rtn_sun': [472.9515686035156, 19.624502182006836, 12.875075340270996],
            'alpha_sw_velocity_gse': [-473.04103169, 10.77708246, 14.79218361],
            'alpha_sw_velocity_gse_sun': [-472.92223443, -18.97376979, 14.78618060],
            'alpha_sw_velocity_gsm': [-473.04103169, 6.79540805, 16.99342908],
            'alpha_sw_velocity_gsm_sun': [-472.92223443, -22.03470465, 9.64919000],
            'alpha_sw_velocity_hae': [-73.24367583, 467.46141800, 14.76341745],
            'alpha_sw_velocity_hae_sun': [-102.50234989, 462.07093380, 14.75788951],
            'alpha_sw_velocity_rtn_covariance': [
                [0.13386771082878113, 0.23742027580738068, 0.1489003300666809],
                [0.23742027580738068, 13.923757553100586, 7.167418003082275],
                [0.1489003300666809, 7.167418003082275, 4.364877223968506],
            ],
            'alpha_sw_velocity_rtn_uncert': [0.36587936649773123, 3.7314551522295676, 2.0892288586865027],
            'swp_flags': 0,
        }

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        data_dir = root_dir / "data"
        imap_data_access.config["DATA_DIR"] = data_dir

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
            ],
            env={**os.environ, "IMAP_DATA_DIR": str(data_dir)},
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

        bulk_speed_atol = 1e-3 * float(expected_values['alpha_sw_speed'])

        with CDF(str(expected_file_path)) as cdf:
            for key in expected_values.keys():
                actual_value = cdf[key][sample_index]
                if key.endswith('_uncert'):
                    rtol, atol = 3e-2, 0.0
                elif 'velocity' in key:
                    rtol, atol = 1e-3, bulk_speed_atol
                else:
                    rtol, atol = 3e-2, 0.0
                try:
                    numpy.testing.assert_allclose(
                        actual_value, expected_values[key], rtol=rtol, atol=atol, err_msg=key
                    )
                except TypeError:
                    self.assertEqual(expected_values[key], actual_value, msg=key)
            assert_velocity_frames_are_physically_consistent(
                self, cdf, "alpha_sw_velocity", sample_index
            )


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
            'pui_cooling_index': 1.9128987789154053,
            'pui_cooling_index_uncert': 0.20524528622627258,
            'pui_ionization_rate': 7.788479194914544e-08,
            'pui_ionization_rate_uncert': 4.0779393195577995e-09,
            'pui_cutoff_speed': 481.7744140625,
            'pui_cutoff_speed_uncert': 3.204035758972168,
            'pui_background_count_rate': 0.4994434714317322,
            'pui_background_count_rate_uncert': 0.1976516991853714,
            'pui_density': 0.0005394626059569418,
            'pui_density_uncert': 2.824551847879775e-05,
            'pui_temperature': 20905214.0,
            'pui_temperature_uncert': 1058934.25,
            'swp_flags': 0,
        }

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        data_dir = root_dir / "data"
        imap_data_access.config["DATA_DIR"] = data_dir

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
            ],
            env={**os.environ, "IMAP_DATA_DIR": str(data_dir)},
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


    @skipUnless(os.environ.get("IMAP_API_KEY"), "requires production API key")
    def test_l3b_with_production_data(self):
        """
        With real data and with full dependency setup, validate that the pui-he
        CDF output matches hardcoded expected values to check for unexpected
        changes.
        """

        root_dir = Path(imap_l3_processing.__file__).parent.parent
        os.chdir(root_dir)
        data_dir = root_dir / "data"
        imap_data_access.config["DATA_DIR"] = data_dir

        dependency_filename = "imap_swapi_l3b_combined_20260101_v001.json"
        stage_input_file(SWAPI_INTEGRATION_DATA_DIR / dependency_filename)

        expected_file_path = ScienceFilePath(
            "imap_swapi_l3b_combined_20260101_v001.0001.cdf"
        ).construct_path()
        if expected_file_path.parent.exists():
            expected_file_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                sys.executable,
                "imap_l3_data_processor.py",
                "--instrument", "swapi",
                "--data-level", "l3b",
                "--start-date", "20260101",
                "--version", "v001",
                "--dependency", dependency_filename,
            ],
            env={**os.environ, "IMAP_DATA_DIR": str(data_dir)},
        )

        self.assertEqual(0, result.returncode)
        self.assertTrue(expected_file_path.exists())

        l2_file_path = ScienceFilePath(
            "imap_swapi_l2_sci_20260101_v001.0001.cdf"
        ).construct_path()
        self.assertTrue(l2_file_path.exists())

        with CDF(str(l2_file_path)) as l2_cdf, CDF(str(expected_file_path)) as l3b_cdf:
            print(l3b_cdf.keys())
            l2_coincidence_rate = l2_cdf["swp_coin_rate"][
                :50, SWAPI_COARSE_SWEEP_BINS
            ].mean(axis=0)
            combined_differential_flux = l3b_cdf["combined_differential_flux"][0]

            geometric_factor = (
                l2_coincidence_rate / combined_differential_flux
            )

            index = np.abs(l3b_cdf["combined_energy"][0] - 1000.0).argmin()
            print(l3b_cdf["combined_energy"][0][index], l3b_cdf["combined_energy"].attrs["UNITS"])
            print(l3b_cdf["combined_differential_flux"][0][index], l3b_cdf["combined_differential_flux"].attrs["UNITS"])
            numpy.testing.assert_allclose(
                geometric_factor[index],
                38.1,
                rtol=1e-2,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
