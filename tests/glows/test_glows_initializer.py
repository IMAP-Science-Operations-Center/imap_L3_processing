import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch, call, sentinel, MagicMock, mock_open

import numpy as np
from astropy.time import Time, TimeDelta
from imap_data_access import ScienceInput, ScienceFilePath
from imap_processing.spice.repoint import set_global_repoint_table_paths
from spacepy.pycdf import CDF

from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess
from tests.glows.l3bc.test_utils import create_l3a_path_by_date, create_imap_data_access_json
from tests.test_helpers import get_test_data_path


class TestGlowsInitializer(unittest.TestCase):

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.determine_crs_to_process")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.group_l3a_by_cr")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_get_crs_to_process(self, mock_query, mock_group_l3a, mock_determine_crs_to_process):

        l3a_file_names = ["imap_glows_l3a_hist_20100101-repoint000001_v001.cdf", "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"]
        l3a_query_result = [{"file_path": "some/server/path/" + file_name} for file_name in l3a_file_names]

        l3b_file_names = [
            "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf",
            "imap_glows_l3b_ion-rate-profile_20100501-cr02096_v001.cdf"
        ]
        l3b_query_result = [{"file_path": "some/l3b/server/path/" + file_name} for file_name in l3b_file_names]

        mock_query.side_effect = [l3a_query_result, l3b_query_result]

        actual_crs_to_process = GlowsInitializer.get_crs_to_process()

        mock_query.assert_has_calls([
            call(instrument="glows", data_level="l3a", version="latest"),
            call(instrument="glows", data_level="l3b", descriptor="ion-rate-profile", version="latest")
        ])

        mock_group_l3a.assert_called_once_with(l3a_file_names)

        expected_l3bs_by_cr = {2091: "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf",
                               2096: "imap_glows_l3b_ion-rate-profile_20100501-cr02096_v001.cdf"}

        mock_determine_crs_to_process.assert_called_once_with(mock_group_l3a.return_value, expected_l3bs_by_cr)

        self.assertEqual(mock_determine_crs_to_process.return_value, actual_crs_to_process)

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.read_cdf_parents")
    def test_determine_crs_to_process(self, mock_read_cdf_parents):
        l3as_on_server = {
            2091: {
                "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf",
            },
            2096: {
                "imap_glows_l3a_hist_20100501-repoint000050_v001.cdf",
                "imap_glows_l3a_hist_20100502-repoint000051_v002.cdf",
            },
            2099: {
                "imap_glows_l3a_hist_20100801-repoint000100_v001.cdf",
            }
        }

        l3b_files_and_parents = {
            2091: "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf",
            2096: "imap_glows_l3b_ion-rate-profile_20100501-cr02096_v001.cdf"
        }

        mock_read_cdf_parents.side_effect = [
            {
                "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"
            },
            {
                "imap_glows_l3a_hist_20100501-repoint000050_v001.cdf",
                "imap_glows_l3a_hist_20100502-repoint000051_v001.cdf"
            }
        ]

        crs_to_process = GlowsInitializer.determine_crs_to_process(l3as_on_server, l3b_files_and_parents)

        mock_read_cdf_parents.assert_has_calls([
            call("imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf"),
            call("imap_glows_l3b_ion-rate-profile_20100501-cr02096_v001.cdf")
        ])

        expected_cr_to_reprocess = CRToProcess(
            l3a_paths=[
                "imap_glows_l3a_hist_20100501-repoint000050_v001.cdf",
                "imap_glows_l3a_hist_20100502-repoint000051_v002.cdf"
            ],
            cr_start_date=datetime(2010, 4, 22, 13, 2, 9, 600000),
            cr_end_date=datetime(2010, 5, 19, 19, 38, 35, 520000),
            cr_rotation_number=2096,
            version=2,
        )

        expected_cr_to_process = CRToProcess(
            l3a_paths=[
                "imap_glows_l3a_hist_20100801-repoint000100_v001.cdf",
            ],
            cr_start_date=datetime(2010, 7, 13, 8, 51, 27, 360000),
            cr_end_date=datetime(2010, 8, 9, 15, 27, 53, 280000),
            cr_rotation_number=2099,
            version=1
        )

        self.assertEqual([expected_cr_to_reprocess, expected_cr_to_process], crs_to_process)

    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_available_ancillary_files_from_descriptor(self, mock_query):

        expected = AvailableAncillaryFiles(
            data=[
                AncillaryFileInfo(
                    ingestion_date=datetime(2010, 1, 3),
                    start_date=datetime(2010, 1, 2),
                    end_date=None,
                    file_name="imap_glows_uv-anisotropy-1CR_20100102_v001.dat"
                ),
                AncillaryFileInfo(
                    ingestion_date=datetime(2010, 1, 1),
                    start_date=datetime(2010, 1, 1),
                    end_date=None,
                    file_name="imap_glows_uv-anisotropy-1CR_20100101_v001.dat"
                ),
            ]
        )

        available_ancillary_files = AvailableAncillaryFiles.from_query_by_descriptor("uv-anisotropy-1CR")

        mock_query.assert_called_once_with(instrument="glows", table="ancillary")


        self.assertEqual(expected, available_ancillary_files)

    def test_available_ancillary_files_filter_by_cr(self):

        (AvailableAncillaryFileInfo()
            .filter_by_cr(2096)
            .sort()
            .first())

    @patch("imap_l3_processing.glows.glows_initializer.get_pointing_date_range")
    def test_group_l3a_by_cr(self, mock_get_pointing_date_range):
        mock_get_pointing_date_range.side_effect = [
            (np.datetime64("2010-01-01T00:00:00"), np.datetime64("2010-01-02T00:00:00")),
            (np.datetime64("2010-01-02T00:00:00"), np.datetime64("2010-01-03T00:00:00")),
            (np.datetime64("2010-05-01T00:00:00"), np.datetime64("2010-05-02T00:00:00")),
            (np.datetime64("2010-05-02T00:00:00"), np.datetime64("2010-05-03T00:00:00")),
            (np.datetime64("2010-08-01T00:00:00"), np.datetime64("2010-08-02T00:00:00")),
            (np.datetime64("2010-08-09T00:00:00"), np.datetime64("2010-08-10T00:00:00")),
        ]

        l3a_files_and_parents = [
            "imap_glows_l3a_hist_20100101-repoint00001_v001.cdf",
            "imap_glows_l3a_hist_20100102-repoint00002_v001.cdf",
            "imap_glows_l3a_hist_20100501-repoint00050_v001.cdf",
            "imap_glows_l3a_hist_20100502-repoint00051_v002.cdf",
            "imap_glows_l3a_hist_20100801-repoint00100_v001.cdf",
            "imap_glows_l3a_hist_20100809-repoint00109_v001.cdf",
        ]

        expected = {
            2091: {
                "imap_glows_l3a_hist_20100101-repoint00001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint00002_v001.cdf",
            },
            2096: {
                "imap_glows_l3a_hist_20100501-repoint00050_v001.cdf",
                "imap_glows_l3a_hist_20100502-repoint00051_v002.cdf",
            },
            2099: {
                "imap_glows_l3a_hist_20100801-repoint00100_v001.cdf",
                "imap_glows_l3a_hist_20100809-repoint00109_v001.cdf",
            },
            2100: {
                "imap_glows_l3a_hist_20100809-repoint00109_v001.cdf",
            }
        }

        grouped_l3a_files = GlowsInitializer.group_l3a_by_cr(l3a_files_and_parents)

        self.assertEqual(expected, grouped_l3a_files)

    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.download")
    def test_read_cdf_parents(self, mock_download):

        with (TemporaryDirectory() as temp_dir):
            cdf_downloaded_path = Path(temp_dir) / "l3b.cdf"

            with CDF(str(cdf_downloaded_path), masterpath='') as cdf:
                cdf.attrs["Parents"] = ["l3a_1.cdf", "l3a_2.cdf"]

            mock_download.return_value = cdf_downloaded_path

            cdf_path = "l3b.cdf"
            parents = GlowsInitializer.read_cdf_parents(cdf_path)

            mock_download.assert_called_once_with(cdf_path)

            self.assertEqual({"l3a_1.cdf", "l3a_2.cdf"}, parents)

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializerAncillaryDependencies")
    def test_validate_and_initialize_returns_empty_list_when_missing_ancillary_dependencies(self,
                                                                                            mock_glows_initializer_ancillary_dependencies: Mock):
        test_cases = [
            ("Missing f107_path", None, Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock()),
            ("Missing lyman_alpha_path", Mock(), None, Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock()),
            ("Missing omni_data_path", Mock(), Mock(), None, Mock(), Mock(), Mock(), Mock(), Mock(), Mock()),
            ("Missing uv_anisotropy", Mock(), Mock(), Mock(), None, Mock(), Mock(), Mock(), Mock(), Mock()),
            ("Missing waw_helioion_mp", Mock(), Mock(), Mock(), Mock(), None, Mock(), Mock(), Mock(), Mock()),
            ("Missing pipeline_settings", Mock(), Mock(), Mock(), Mock(), Mock(), None, Mock(), Mock(), Mock()),
            ("Missing pipeline_settings buffer", Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), None, Mock(), Mock()),
            ("Missing bad_days", Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), None, Mock()),
            ("Missing repointing", Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), None),
            ("Missing all dependencies", None, None, None, None, None, None, None, None, None),
        ]

        for name, f107_path, lyman_alpha_path, omni_data_path, uv_anisotropy, waw_helioion_mp, pipeline_settings, pipeline_settings_buffer, bad_days, repointing in test_cases:
            with self.subTest(name):
                mock_glows_initializer_ancillary_dependencies.fetch_dependencies.return_value = GlowsInitializerAncillaryDependencies(
                    uv_anisotropy, waw_helioion_mp, pipeline_settings, bad_days, pipeline_settings_buffer, f107_path,
                    lyman_alpha_path,
                    omni_data_path, repointing)

                self.assertEqual([], GlowsInitializer.validate_and_initialize("", Mock()))

    @patch("imap_l3_processing.glows.glows_initializer.archive_dependencies")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializerAncillaryDependencies")
    @patch("imap_l3_processing.glows.glows_initializer.query")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.find_unprocessed_carrington_rotations")
    def test_validate_and_initialize_l3b(self, mock_find_unprocessed_carrington_rotations: Mock, mock_query: Mock,
                                         mock_glows_initializer_ancillary_dependencies: Mock,
                                         mock_archive_dependencies: Mock):
        version = "v003"
        mock_l3a = MagicMock()
        mock_l3b = MagicMock()

        mock_query.side_effect = [
            [mock_l3a],
            [mock_l3b]
        ]
        mock_archive_dependencies.side_effect = [
            sentinel.zip_path_1,
            sentinel.zip_path_2
        ]

        ancillary_dependencies = GlowsInitializerAncillaryDependencies(Mock(), Mock(), Mock(), Mock(), Mock(), Mock(),
                                                                       Mock(), Mock(), Mock())

        mock_glows_initializer_ancillary_dependencies.fetch_dependencies.return_value = ancillary_dependencies
        mock_cr_to_process_1 = Mock(cr_rotation_number=1)
        mock_cr_to_process_2 = Mock(cr_rotation_number=2)
        mock_find_unprocessed_carrington_rotations.return_value = [mock_cr_to_process_1, mock_cr_to_process_2]

        mock_dependencies = Mock()
        expected_l3a_version = "v123"
        expected_science_inputs = ScienceInput(f"imap_glows_l3a_hist_20100606_{expected_l3a_version}.cdf")
        mock_dependencies.get_science_inputs.return_value = [expected_science_inputs]

        actual_zip_paths = GlowsInitializer.validate_and_initialize(version, mock_dependencies)

        mock_dependencies.get_science_inputs.assert_called_once_with("glows")
        self.assertEqual(2, mock_query.call_count)
        mock_query.assert_has_calls(
            [call(instrument="glows", descriptor='hist', version=expected_l3a_version, data_level="l3a"),
             call(instrument="glows", descriptor='ion-rate-profile', version=version, data_level="l3b")])

        mock_find_unprocessed_carrington_rotations.assert_called_once_with([mock_l3a], [mock_l3b], ancillary_dependencies)

        mock_glows_initializer_ancillary_dependencies.fetch_dependencies.assert_called_once_with(mock_dependencies)

        mock_archive_dependencies.assert_has_calls([
            call(mock_cr_to_process_1, version, ancillary_dependencies),
            call(mock_cr_to_process_2, version, ancillary_dependencies),
        ])

        self.assertEqual(2, len(actual_zip_paths))
        self.assertEqual(sentinel.zip_path_1, actual_zip_paths[0])
        self.assertEqual(sentinel.zip_path_2, actual_zip_paths[1])

    @patch("imap_l3_processing.glows.glows_initializer.validate_dependencies")
    def test_find_unprocessed_carrington_rotations(self,
                                                   mock_validate_dependencies: Mock):
        set_global_repoint_table_paths([Path("not_set_yet")])
        l3a_files_january = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_200001{str(i + 1).zfill(2)}-repoint{str(i).zfill(5)}_v001.pkts',
                data_level='l3a', start_date=f'200001{str(i + 1).zfill(2)}', repointing=i) for i in range(4, 31)
        ]
        l3a_files_february = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_200002{str(i + 1).zfill(2)}-repoint{str(i + 31).zfill(5)}_v001.pkts',
                data_level='l3a', start_date=f'200002{str(i + 1).zfill(2)}', repointing=i + 31) for i in range(1, 29)
        ]
        l3a_files_march = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_200003{str(i + 1).zfill(2)}-repoint{str(i + 60).zfill(5)}_v001.pkts',
                data_level='l3a', start_date=f'200003{str(i + 1).zfill(2)}', repointing=i + 60) for i in range(1, 27)
        ]

        l3a_files_april = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000403-repoint00093_v001.pkts',
                data_level='l3a', start_date=f'20000403', repointing=93),
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000423-repoint00113_v001.pkts',
                data_level='l3a', start_date=f'20000423', repointing=113),
        ]
        l3a_files_july = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000720-repoint00201_v001.pkts',
                data_level='l3a', start_date=f'20000720', repointing=201),
        ]

        l3a_files = l3a_files_february + l3a_files_march + l3a_files_january + l3a_files_april + l3a_files_july

        l3b_files = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3bc/2000/01/imap_glows_l3b_hist_20000130_v001.pkts',
                data_level='l3b', start_date=f'20000130')
        ]

        mock_validate_dependencies.side_effect = [True, False, True, True]

        expected_l3a_1958 = [create_l3a_path_by_date(f'200001{str(i + 1).zfill(2)}', i) for i in range(4, 28)]
        expected_l3a_1959 = [create_l3a_path_by_date(f'200001{str(i + 1).zfill(2)}', i) for i in range(27, 31)]
        expected_l3a_1959 += [create_l3a_path_by_date(f'200002{str(i + 1).zfill(2)}', i + 31) for i in range(1, 24)]
        # expected_l3a_1960 = [create_l3a_path_by_date(f'200002{str(i + 1).zfill(2)}', i + 31) for i in range(24, 29)]
        # expected_l3a_1960 += [create_l3a_path_by_date(f'200003{str(i).zfill(2)}', i + 60) for i in range(1, 24)]
        expected_l3a_1961 = [create_l3a_path_by_date(f'200003{str(i + 1).zfill(2)}', i + 60) for i in range(21, 27)]
        expected_l3a_1961.append(create_l3a_path_by_date(f'20000403', 93))
        expected_l3a_1962 = [create_l3a_path_by_date(f'20000423', 113)]
        # expected_l3a_1965 = [create_l3a_path_by_date(f'20000711', 192)]

        initializer_dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                                         waw_helioion_mp_path="waw_helioion",
                                                                         bad_days_list="bad_days_list",
                                                                         pipeline_settings="pipeline_settings",
                                                                         lyman_alpha_path=Path("lyman_alpha"),
                                                                         omni2_data_path=Path("omni"),
                                                                         initializer_time_buffer=TimeDelta(52,
                                                                                                           format="jd"),
                                                                         f107_index_file_path=Path("f107"),
                                                                         repointing_file=get_test_data_path(
                                                                             "fake_1_day_repointing_file.csv"))

        actual_crs_to_process: [CRToProcess] = GlowsInitializer.find_unprocessed_carrington_rotations(l3a_files, l3b_files,
                                                                                     initializer_dependencies)

        self.assertEqual(3, len(actual_crs_to_process))
        cr_to_process_1958 = actual_crs_to_process[0]
        self.assertEqual(expected_l3a_1958, cr_to_process_1958.l3a_paths)
        self.assertEqual(Time('2000-01-01 14:11:11.040').value, cr_to_process_1958.cr_start_date.value)
        self.assertEqual(Time('2000-01-28 20:47:36.960').value, cr_to_process_1958.cr_end_date.value)
        self.assertEqual(1958, cr_to_process_1958.cr_rotation_number)

        cr_to_process_1961 = actual_crs_to_process[1]
        self.assertEqual(expected_l3a_1961, cr_to_process_1961.l3a_paths)
        self.assertEqual(Time('2000-03-23 10:00:28.800').value, cr_to_process_1961.cr_start_date.value)
        self.assertEqual(Time('2000-04-19 16:36:54.720').value, cr_to_process_1961.cr_end_date.value)
        self.assertEqual(1961, cr_to_process_1961.cr_rotation_number)

        cr_to_process_1962 = actual_crs_to_process[2]
        self.assertEqual(expected_l3a_1962, cr_to_process_1962.l3a_paths)
        self.assertEqual(Time('2000-04-19 16:36:54.720').value, cr_to_process_1962.cr_start_date.value)
        self.assertEqual(Time('2000-05-16 23:13:20.640').value, cr_to_process_1962.cr_end_date.value)
        self.assertEqual(1962, cr_to_process_1962.cr_rotation_number)

        self.assertEqual(Time('2000-01-28 20:47:36.960').value,
                         mock_validate_dependencies.call_args_list[0][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[0][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[0][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[0][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[0][0][4])

        self.assertEqual(Time('2000-03-23 10:00:28.800').value,
                         mock_validate_dependencies.call_args_list[1][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[1][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[1][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[1][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[1][0][4])

        self.assertEqual(Time('2000-04-19 16:36:54.720').value,
                         mock_validate_dependencies.call_args_list[2][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[2][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[2][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[2][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[2][0][4])

    @patch("imap_l3_processing.glows.glows_initializer.validate_dependencies")
    def test_find_unprocessed_carrington_rotations_handles_multi_day_repointing(self, mock_validate_dependencies: Mock):
        l3a_in_1958 = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000104-repoint00004_v001.cdf',
            data_level='l3a',
            start_date=f'20000104', repointing=4)
        l3a_in_1958_and_1959 = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000128-repoint00027_v001.cdf',
            data_level='l3a',
            start_date=f'20000128', repointing=27)
        l3a_in_1959 = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_20001031-repoint00030_v001.cdf',
            data_level='l3a',
            start_date=f'20000131', repointing=30)
        l3a_past_buffer_range = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_2000528-repoint00148_v001.cdf',
            data_level='l3a',
            start_date=f'20000528', repointing=148)
        l3a_files = [l3a_in_1958, l3a_in_1958_and_1959, l3a_in_1959, l3a_past_buffer_range]

        mock_validate_dependencies.return_value = True

        expected_l3a_1958 = [l3a_in_1958.get('file_path'), l3a_in_1958_and_1959.get('file_path')]
        expected_l3a_1959 = [l3a_in_1958_and_1959.get('file_path'), l3a_in_1959.get('file_path')]

        mock_dependencies = Mock(initializer_time_buffer=56,
                                 repointing_file=get_test_data_path("fake_1_day_repointing_file.csv"))
        actual_crs_to_process: [CRToProcess] = GlowsInitializer.find_unprocessed_carrington_rotations(l3a_files, [], mock_dependencies)

        self.assertEqual(2, len(actual_crs_to_process))
        cr_to_process_1958 = actual_crs_to_process[0]
        self.assertEqual(expected_l3a_1958, cr_to_process_1958.l3a_paths)
        self.assertEqual(Time('2000-01-01 14:11:11.040').value, cr_to_process_1958.cr_start_date.value)
        self.assertEqual(Time('2000-01-28 20:47:36.960').value, cr_to_process_1958.cr_end_date.value)
        self.assertEqual(1958, cr_to_process_1958.cr_rotation_number)

        cr_to_process_1959 = actual_crs_to_process[1]
        self.assertEqual(expected_l3a_1959, cr_to_process_1959.l3a_paths)
        self.assertEqual(Time('2000-01-28 20:47:36.960').value, cr_to_process_1959.cr_start_date.value)
        self.assertEqual(Time('2000-02-25 03:24:02.880').value, cr_to_process_1959.cr_end_date.value)
        self.assertEqual(1959, cr_to_process_1959.cr_rotation_number)



