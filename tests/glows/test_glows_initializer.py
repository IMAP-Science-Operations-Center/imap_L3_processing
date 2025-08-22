import dataclasses
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

    @patch("imap_l3_processing.glows.glows_initializer.get_date_range_of_cr")
    @patch("imap_l3_processing.glows.glows_initializer.get_best_ancillary")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.group_l3a_by_cr")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.should_process_cr_candidate")
    def test_gr_crs_to_process(self, mock_should_process_cr_candidate, mock_group_l3a_by_cr, mock_query, mock_get_best_ancillary, mock_get_date_range_of_cr):

        mock_query.side_effect = [
            [
                {"file_path": "some/server/path/glows_l3a_hist_20100101_v001.cdf"},
                {"file_path": "some/server/path/glows_l3a_hist_20100201_v001.cdf"},
            ],
            sentinel.uv_anisotropy_query_result,
            sentinel.waw_helio_ion_query_result,
            sentinel.bad_days_list_query_result,
            sentinel.pipeline_settings_query_result
        ]

        mock_group_l3a_by_cr.return_value = {
            2091: {"glows_l3a_hist_20100101_v001.cdf"},
            2092: {"glows_l3a_hist_20100201_v001.cdf"}
        }

        mock_get_date_range_of_cr.side_effect = [
            (datetime(2010, 1, 1), datetime(2010, 1, 31)),
            (datetime(2010, 2, 1), datetime(2010, 2, 28)),
        ]

        mock_get_best_ancillary.side_effect = [
            "uv_anisotropy_best_ancillary_1",
            "waw_helio_ion_best_ancillary_1",
            "bad_days_list_best_ancillary_1",
            "pipeline_settings_best_ancillary_1",
            "uv_anisotropy_best_ancillary_2",
            "waw_helio_ion_best_ancillary_2",
            "bad_days_list_best_ancillary_2",
            "pipeline_settings_best_ancillary_2",
        ]

        mock_should_process_cr_candidate.side_effect = [2, None]

        actual_crs_to_process = GlowsInitializer.get_crs_to_process()

        mock_query.assert_has_calls([
            call(instrument="glows", data_level="l3a", version="latest"),
            call(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR"),
            call(table="ancillary", instrument="glows", descriptor="WawHelioIonMP"),
            call(table="ancillary", instrument="glows", descriptor="bad-days-list"),
            call(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde")
        ])

        mock_group_l3a_by_cr.assert_called_once_with([
            "glows_l3a_hist_20100101_v001.cdf",
            "glows_l3a_hist_20100201_v001.cdf",
        ])

        mock_get_best_ancillary.assert_has_calls([
            call(datetime(2010, 1, 1), datetime(2010, 1, 31), sentinel.uv_anisotropy_query_result),
            call(datetime(2010, 1, 1), datetime(2010, 1, 31), sentinel.waw_helio_ion_query_result),
            call(datetime(2010, 1, 1), datetime(2010, 1, 31), sentinel.bad_days_list_query_result),
            call(datetime(2010, 1, 1), datetime(2010, 1, 31), sentinel.pipeline_settings_query_result),
            call(datetime(2010, 2, 1), datetime(2010, 2, 28), sentinel.uv_anisotropy_query_result),
            call(datetime(2010, 2, 1), datetime(2010, 2, 28), sentinel.waw_helio_ion_query_result),
            call(datetime(2010, 2, 1), datetime(2010, 2, 28), sentinel.bad_days_list_query_result),
            call(datetime(2010, 2, 1), datetime(2010, 2, 28), sentinel.pipeline_settings_query_result),
        ])

        expected_cr_candidate_1 = CRToProcess(l3a_file_names={"glows_l3a_hist_20100101_v001.cdf"},
                              uv_anisotropy_file_name="uv_anisotropy_best_ancillary_1",
                              waw_helio_ion_mp_file_name="waw_helio_ion_best_ancillary_1",
                              bad_days_list_file_name="bad_days_list_best_ancillary_1",
                              pipeline_settings_file_name="pipeline_settings_best_ancillary_1",
                              cr_start_date=datetime(2010, 1, 1), cr_end_date=datetime(2010, 1, 31),
                              cr_rotation_number=2091)

        expected_cr_candidate_2 = CRToProcess(l3a_file_names={"glows_l3a_hist_20100201_v001.cdf"},
                              uv_anisotropy_file_name="uv_anisotropy_best_ancillary_2",
                              waw_helio_ion_mp_file_name="waw_helio_ion_best_ancillary_2",
                              bad_days_list_file_name="bad_days_list_best_ancillary_2",
                              pipeline_settings_file_name="pipeline_settings_best_ancillary_2",
                              cr_start_date=datetime(2010, 2, 1), cr_end_date=datetime(2010, 2, 28),
                              cr_rotation_number=2092)

        mock_should_process_cr_candidate.assert_has_calls([
            call(expected_cr_candidate_1),
            call(expected_cr_candidate_2)
        ], any_order=False)

        self.assertEqual([(2, expected_cr_candidate_1)], actual_crs_to_process)

    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_should_process_cr_no_existing_l3b(self, mock_query):

        cr_candidate = CRToProcess(
            l3a_file_names=set(),
            uv_anisotropy_file_name="uv_anisotropy.dat",
            waw_helio_ion_mp_file_name="waw_helio_ion.dat",
            bad_days_list_file_name="bad_days_list.dat",
            pipeline_settings_file_name="pipeline_settings.json",
            cr_start_date=datetime(2010, 1, 1),
            cr_end_date=datetime(2010, 2, 1),
            cr_rotation_number=2091
        )

        mock_query.return_value = []

        actual_version = GlowsInitializer.should_process_cr_candidate(cr_candidate)

        mock_query.assert_called_once_with(
            instrument="glows",
            data_level="l3b",
            descriptor="ion-rate-profile",
            start_date="20100101",
            end_date="20100201",
            version="latest"
        )

        self.assertEqual(1, actual_version)

    @patch("imap_l3_processing.glows.glows_initializer.read_cdf_parents")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_should_process_cr_existing_cr_should_process(self, mock_query, mock_read_cdf_parents):

        new_l3a_version = {
            "l3a_file_names": {
                "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint000002_v002.cdf"
            }
        }

        new_l3a_file = {
            "l3a_file_names": {
                "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"
                "imap_glows_l3a_hist_20100103-repoint000002_v001.cdf"
            }
        }

        new_ancillary = {
            "uv_anisotropy_file_name": "new_uv_anisotropy.dat"
        }

        test_cases = [
            ("new l3a file version", new_l3a_version),
            ("new l3a files arrived", new_l3a_file),
            ("new ancillary file arrived", new_ancillary)
        ]

        for name, new_server_data in test_cases:
            with self.subTest(name):
                mock_query.reset_mock()
                mock_read_cdf_parents.reset_mock()

                existing_file_version = 2

                already_processed_cr = CRToProcess(
                    l3a_file_names={
                        "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                        "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"
                    },
                    uv_anisotropy_file_name="uv_anisotropy.dat",
                    waw_helio_ion_mp_file_name="waw_helio_ion.dat",
                    bad_days_list_file_name="bad_days_list.dat",
                    pipeline_settings_file_name="pipeline_settings.json",
                    cr_start_date=datetime(2010, 1, 1),
                    cr_end_date=datetime(2010, 2, 1),
                    cr_rotation_number=2091
                )

                cr_candidate = dataclasses.replace(already_processed_cr, **new_server_data)

                mock_query.return_value = [
                    {"file_path": f"some/server_path/imap_glows_l3b_ion-rate-profile_20100101-cr02091_v00{existing_file_version}.cdf"}]

                mock_read_cdf_parents.return_value = {
                    "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                    "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf",
                    "uv_anisotropy.dat",
                    "waw_helio_ion.dat",
                    "bad_days_list.dat",
                    "pipeline_settings.json",
                    "some_zip_file.zip"
                }

                actual_version = GlowsInitializer.should_process_cr_candidate(cr_candidate)

                mock_query.assert_called_once_with(
                    instrument="glows",
                    data_level="l3b",
                    descriptor="ion-rate-profile",
                    start_date="20100101",
                    end_date="20100201",
                    version="latest"
                )

                mock_read_cdf_parents.assert_called_once_with(f"imap_glows_l3b_ion-rate-profile_20100101-cr02091_v00{existing_file_version}.cdf")

                self.assertEqual(existing_file_version + 1, actual_version)

    @patch("imap_l3_processing.glows.glows_initializer.read_cdf_parents")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_should_process_cr_existing_cr_should_not_reprocess(self, mock_query, mock_read_cdf_parents):
        cr_candidate = CRToProcess(
            l3a_file_names={
                "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"
            },
            uv_anisotropy_file_name="uv_anisotropy.dat",
            waw_helio_ion_mp_file_name="waw_helio_ion.dat",
            bad_days_list_file_name="bad_days_list.dat",
            pipeline_settings_file_name="pipeline_settings.json",
            cr_start_date=datetime(2010, 1, 1),
            cr_end_date=datetime(2010, 2, 1),
            cr_rotation_number=2091
        )
        mock_query.return_value = [{"file_path": "some/server_path/imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf"}]

        mock_read_cdf_parents.return_value = {
            "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
            "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf",
            "uv_anisotropy.dat",
            "waw_helio_ion.dat",
            "bad_days_list.dat",
            "pipeline_settings.json",
            "some_zip_file.zip"
        }

        self.assertIsNone(GlowsInitializer.should_process_cr_candidate(cr_candidate))

        mock_query.assert_called_once_with(
            instrument="glows",
            data_level="l3b",
            descriptor="ion-rate-profile",
            start_date="20100101",
            end_date="20100201",
            version="latest"
        )

        mock_read_cdf_parents.assert_called_once_with("imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf")

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.determine_crs_to_process")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_get_crs_to_process_return_no_crs_when_missing_ancillaries(self, mock_query, mock_determine_crs_to_process):

        test_cases = [
            "uv-anisotropy-1CR",
            "WawHelioIonMP",
            "bad-days-list",
            "pipeline-settings-l3bcde"
        ]

        for missing_ancillary in test_cases:
            with self.subTest(missing_ancillary):
                def query_that_returns_empty_list_for_missing_ancillary(**kwargs):
                    if kwargs.get("table") == "ancillary" and kwargs["descriptor"] != missing_ancillary:
                        return [sentinel.ancillary_1, sentinel.ancillary_2]
                    else:
                        return []

                mock_query.side_effect = query_that_returns_empty_list_for_missing_ancillary

                actual_crs_to_process = GlowsInitializer.get_crs_to_process()

                mock_determine_crs_to_process.assert_not_called()
                self.assertEqual([], actual_crs_to_process)

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



