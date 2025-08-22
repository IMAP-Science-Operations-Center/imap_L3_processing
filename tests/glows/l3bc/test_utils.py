import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import patch, Mock, MagicMock, call
from zipfile import ZIP_DEFLATED

import numpy as np
from astropy.time import Time, TimeDelta
from imap_processing.spice.repoint import set_global_repoint_table_paths
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess
from imap_l3_processing.glows.l3bc.utils import read_glows_l3a_data, \
    archive_dependencies, get_pointing_date_range, get_best_ancillary, read_cdf_parents
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repoint_file_path = get_test_data_path("fake_1_day_repointing_file.csv")
        set_global_repoint_table_paths([repoint_file_path])

    def test_determine_crs_to_process_based_on_ancillary_files(self):

        start_date = datetime(2009, 12, 31)
        end_date = datetime(2010, 1, 2)

        case_1 = [
            {"file_path": "some/server/path/" + "older_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100101 00:00:00"},
            {"file_path": "some/server/path/" + "newer_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100102 00:00:00"}
        ]

        case_2 = [
            {"file_path": "some/server/path/" + "older_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100101 00:00:00"},
            {"file_path": "some/server/path/" + "newer_ancillary.dat", "start_date": "20100104", "end_date": None,
             "ingestion_date": "20100102 00:00:00"}
        ]

        case_3 = [
            {"file_path": "some/server/path/" + "older_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100101 00:00:00"},
            {"file_path": "some/server/path/" + "newer_ancillary.dat", "start_date": "20091205", "end_date": "20091206",
             "ingestion_date": "20100102 00:00:00"}
        ]

        test_cases = [
            ("picks latest ingestion date", case_1, "newer_ancillary.dat"),
            ("ignores ancillary with start date after cr", case_2, "older_ancillary.dat"),
            ("ignores ancillary with end date before cr", case_3, "older_ancillary.dat")
        ]

        for name, available_ancillaries, expected_best_ancillary_file_name in test_cases:
            with self.subTest(name):
                actual_ancillary_name = get_best_ancillary(start_date, end_date, available_ancillaries)

                self.assertEqual(expected_best_ancillary_file_name, actual_ancillary_name)

    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.download")
    def test_read_cdf_parents(self, mock_download):
        with (TemporaryDirectory() as temp_dir):
            cdf_downloaded_path = Path(temp_dir) / "l3b.cdf"

            with CDF(str(cdf_downloaded_path), masterpath='') as cdf:
                cdf.attrs["Parents"] = ["l3a_1.cdf", "l3a_2.cdf"]

            mock_download.return_value = cdf_downloaded_path

            cdf_path = "l3b.cdf"
            parents = read_cdf_parents(cdf_path)

            mock_download.assert_called_once_with(cdf_path)

            self.assertEqual({"l3a_1.cdf", "l3a_2.cdf"}, parents)

    def test_read_glows_l3a_data(self):
        cdf = CDF(str(get_test_data_path("glows/imap_glows_l3a_hist_20100101_v001.cdf")))

        actual_glows_lightcurve = read_glows_l3a_data(cdf)

        self.assertAlmostEqual(7.48702879e+01, actual_glows_lightcurve.latitude[0][0])
        self.assertAlmostEqual(154.67118388, actual_glows_lightcurve.longitude[0][0])
        self.assertEqual(datetime(2013, 9, 8, 18, 55, 14), actual_glows_lightcurve.epoch[0])
        self.assertEqual(36180, actual_glows_lightcurve.epoch_delta[0])
        self.assertEqual(802.8, actual_glows_lightcurve.exposure_times[0][0])
        self.assertEqual(65, actual_glows_lightcurve.number_of_bins)
        self.assertEqual(0.0007200144163580106, actual_glows_lightcurve.extra_heliospheric_background[0][0])
        self.assertEqual(-27.84000015258789, actual_glows_lightcurve.filter_temperature_average[0])
        self.assertEqual(0.0, actual_glows_lightcurve.filter_temperature_std_dev[0])
        self.assertEqual(1527.0999755859375, actual_glows_lightcurve.hv_voltage_average[0])
        self.assertEqual(87.94999694824219, actual_glows_lightcurve.hv_voltage_std_dev[0])
        self.assertEqual(620.9317389138017, actual_glows_lightcurve.photon_flux[0][0])
        self.assertEqual(0.8794643666117251, actual_glows_lightcurve.photon_flux_uncertainty[0][0])
        self.assertEqual(91.5780029296875, actual_glows_lightcurve.position_angle_offset_average[0])
        self.assertEqual(0.009991000406444073, actual_glows_lightcurve.position_angle_offset_std_dev[0])
        self.assertEqual(0.29899999499320984, actual_glows_lightcurve.pulse_length_average[0])
        self.assertEqual(0.017260000109672546, actual_glows_lightcurve.pulse_length_std_dev[0])
        self.assertEqual(498484, actual_glows_lightcurve.raw_histogram[0][0])
        self.assertEqual(146231104.0, actual_glows_lightcurve.spacecraft_location_average[0][0])
        self.assertEqual(142100.0, actual_glows_lightcurve.spacecraft_location_std_dev[0][0])
        self.assertEqual(6.669000148773193, actual_glows_lightcurve.spacecraft_velocity_average[0][0])
        self.assertEqual(0.11879999935626984, actual_glows_lightcurve.spacecraft_velocity_std_dev[0][0])
        self.assertEqual(2.0, actual_glows_lightcurve.spin_angle[0])
        self.assertEqual(2.0, actual_glows_lightcurve.spin_angle_delta[0])
        self.assertEqual(162.0919952392578, actual_glows_lightcurve.spin_axis_orientation_average[0][0])
        self.assertEqual(0.2345000058412552, actual_glows_lightcurve.spin_axis_orientation_std_dev[0][0])
        self.assertEqual(15.0, actual_glows_lightcurve.spin_period_average[0])
        self.assertEqual(15.236681938171387, actual_glows_lightcurve.spin_period_ground_average[0])
        self.assertEqual(0.0014979999978095293, actual_glows_lightcurve.spin_period_ground_std_dev[0])
        self.assertEqual(0.0, actual_glows_lightcurve.spin_period_std_dev[0])
        self.assertEqual(0.0, actual_glows_lightcurve.time_dependent_background[0][0])

    def test_get_pointing_date_range(self):
        repointing_number = 13
        actual_start, actual_end = get_pointing_date_range(repointing_number)
        expected_start = np.datetime64(
            datetime(year=2000, month=1, day=14, hour=12, minute=13, second=55, microsecond=816000).isoformat())
        expected_end = np.datetime64(
            datetime(year=2000, month=1, day=15, hour=11, minute=58, second=55, microsecond=816000).isoformat())

        np.testing.assert_array_equal(actual_start, expected_start)
        np.testing.assert_array_equal(actual_end, expected_end)

    def test_get_repoint_date_range_handles_no_pointing(self):
        repointing_number = 5998
        with self.assertRaises(ValueError) as err:
            _, _ = get_pointing_date_range(repointing_number)
        self.assertEqual(str(err.exception), f"No pointing found for pointing: 5998")

    @patch("imap_l3_processing.glows.l3bc.utils.json")
    @patch("imap_l3_processing.glows.l3bc.utils.ZipFile")
    def test_archive_dependencies(self, mock_zip, mock_json):
        expected_filepath = TEMP_CDF_FOLDER_PATH / "imap_glows_l3b-archive_20250314_v001.zip"
        expected_json_filename = "cr_to_process.json"

        dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                             waw_helioion_mp_path="waw_helioion",
                                                             bad_days_list="bad_days",
                                                             pipeline_settings="pipeline_settings",
                                                             lyman_alpha_path=Path("lyman_alpha"),
                                                             omni2_data_path=Path("omni"),
                                                             f107_index_file_path=Path("f107"),
                                                             initializer_time_buffer=TimeDelta(42, format="jd"),
                                                             repointing_file=Path("/path/to/repointing.csv")
                                                             )

        cr_to_process: CRToProcess = CRToProcess(cr_rotation_number=2095, l3a_paths=["file1", "file2"],
                                                 cr_start_date=Time("2025-03-14 12:34:56.789"),
                                                 cr_end_date=Time("2025-03-24 12:34:56.789"))

        expected_json_to_serialize = {"cr_rotation_number": 2095,
                                      "l3a_paths": ["file1", "file2"],
                                      "cr_start_date": "2025-03-14 12:34:56.789",
                                      "cr_end_date": "2025-03-24 12:34:56.789",
                                      "bad_days_list": dependencies.bad_days_list,
                                      "pipeline_settings": dependencies.pipeline_settings,
                                      "waw_helioion_mp": dependencies.waw_helioion_mp_path,
                                      "uv_anisotropy": dependencies.uv_anisotropy_path,
                                      "repointing_file": "repointing.csv",
                                      }

        mock_zip_file = MagicMock()
        mock_zip.return_value.__enter__.return_value = mock_zip_file

        version_number = "v001"
        actual_zip_file_name = archive_dependencies(cr_to_process, version_number, dependencies)

        self.assertEqual(expected_filepath, actual_zip_file_name)

        mock_zip.assert_called_with(expected_filepath, "w", ZIP_DEFLATED)

        mock_json.dumps.assert_called_once_with(expected_json_to_serialize)

        mock_zip_file.write.assert_has_calls([
            call(dependencies.lyman_alpha_path, "lyman_alpha_composite.nc"),
            call(dependencies.omni2_data_path, "omni2_all_years.dat"),
            call(dependencies.f107_index_file_path, "f107_fluxtable.txt"),
        ])
        mock_zip_file.writestr.assert_called_once_with(expected_json_filename, mock_json.dumps.return_value)


def create_imap_data_access_json(file_path: str, data_level: str, start_date: str,
                                 descriptor: str = "hist", version: str = "v001",
                                 repointing: Optional[int] = None) -> dict:
    return {'file_path': file_path, 'instrument': 'glows', 'data_level': data_level, 'descriptor': descriptor,
            'start_date': start_date, 'repointing': repointing, 'version': version, 'extension': 'pkts',
            'ingestion_date': '2024-10-11 15:28:32'}


def create_l3a_path_by_date(file_date: str, repointing: int) -> str:
    return f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_{file_date}-repoint{str(repointing).zfill(5)}_v001.pkts'
