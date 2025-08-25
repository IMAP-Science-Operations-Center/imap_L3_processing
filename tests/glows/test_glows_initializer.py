import dataclasses
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from unittest.mock import Mock, patch, call, sentinel

import numpy as np

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.glows_initializer import GlowsInitializer, _comment_headers
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import \
    F107_FLUX_TABLE_URL, LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from imap_l3_processing.glows.l3bc.models import CRToProcess


class TestGlowsInitializer(unittest.TestCase):
    @patch("imap_l3_processing.glows.glows_initializer.get_date_range_of_cr")
    @patch("imap_l3_processing.glows.glows_initializer.get_best_ancillary")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.group_l3a_by_cr")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.should_process_cr_candidate")
    @patch("imap_l3_processing.glows.glows_initializer.download_external_dependency")
    @patch("imap_l3_processing.glows.glows_initializer._comment_headers")
    def test_gr_crs_to_process(self, mock_comment_header, mock_download_external_dependencies, mock_should_process_cr_candidate, mock_group_l3a_by_cr, mock_query, mock_get_best_ancillary, mock_get_date_range_of_cr):

        mock_query.side_effect = [
            [
                {"file_path": "some/server/path/glows_l3a_hist_20100101_v001.cdf"},
                {"file_path": "some/server/path/glows_l3a_hist_20100201_v001.cdf"},
            ],
            sentinel.uv_anisotropy_query_result,
            sentinel.waw_helio_ion_query_result,
            sentinel.bad_days_list_query_result,
            sentinel.pipeline_settings_query_result,
            [
                {"file_path": "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf"},
                {"file_path": "imap_glows_l3b_ion-rate-profile_20100201-cr02092_v001.cdf"},
            ]
        ]

        mock_download_external_dependencies.side_effect = [
            Path("external/file/f107_path"),
            Path("external/file/lya_path"),
            Path("external/file/omni_path"),
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
            call(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde"),
            call(instrument="glows", data_level="l3b", descriptor="ion-rate-profile", version="latest")
        ])

        mock_group_l3a_by_cr.assert_called_once_with([
            "glows_l3a_hist_20100101_v001.cdf",
            "glows_l3a_hist_20100201_v001.cdf",
        ])

        mock_download_external_dependencies.assert_has_calls([
            call(F107_FLUX_TABLE_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt'),
            call(LYMAN_ALPHA_COMPOSITE_INDEX_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt'),
            call(OMNI2_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')
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

        expected_cr_candidate_1 = CRToProcess(
            l3a_file_names={"glows_l3a_hist_20100101_v001.cdf"},
            uv_anisotropy_file_name="uv_anisotropy_best_ancillary_1",
            waw_helio_ion_mp_file_name="waw_helio_ion_best_ancillary_1",
            bad_days_list_file_name="bad_days_list_best_ancillary_1",
            pipeline_settings_file_name="pipeline_settings_best_ancillary_1",
            cr_start_date=datetime(2010, 1, 1), cr_end_date=datetime(2010, 1, 31),
            cr_rotation_number=2091,
            f107_index_file_path=Path("external/file/f107_path"),
            lyman_alpha_path=Path("external/file/lya_path"),
            omni2_data_path=Path("external/file/omni_path")
        )

        expected_cr_candidate_2 = CRToProcess(
            l3a_file_names={"glows_l3a_hist_20100201_v001.cdf"},
            uv_anisotropy_file_name="uv_anisotropy_best_ancillary_2",
            waw_helio_ion_mp_file_name="waw_helio_ion_best_ancillary_2",
            bad_days_list_file_name="bad_days_list_best_ancillary_2",
            pipeline_settings_file_name="pipeline_settings_best_ancillary_2",
            cr_start_date=datetime(2010, 2, 1), cr_end_date=datetime(2010, 2, 28),
            cr_rotation_number=2092,
            f107_index_file_path=Path("external/file/f107_path"),
            lyman_alpha_path=Path("external/file/lya_path"),
            omni2_data_path=Path("external/file/omni_path")
        )

        expected_l3bs_by_cr = {
            2091: "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf",
            2092: "imap_glows_l3b_ion-rate-profile_20100201-cr02092_v001.cdf"
        }

        mock_should_process_cr_candidate.assert_has_calls([
            call(expected_cr_candidate_1, expected_l3bs_by_cr),
            call(expected_cr_candidate_2, expected_l3bs_by_cr)
        ], any_order=False)
        mock_comment_header.assert_called_once_with(expected_cr_candidate_2.f107_index_file_path)

        self.assertEqual([(2, expected_cr_candidate_1)], actual_crs_to_process)

    @patch("imap_l3_processing.glows.glows_initializer._comment_headers")
    @patch("imap_l3_processing.glows.glows_initializer.download_external_dependency")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.group_l3a_by_cr")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    def test_get_crs_to_process_return_no_crs_when_missing_ancillaries(self,  mock_query, mock_group_l3a_by_cr, _, __):

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
                        return [{"start_date": "20100101", "end_date": None, "ingestion_date": "20100101 00:00:00", "file_path": f"some/ancillary/{missing_ancillary}"}]
                    else:
                        return []

                mock_group_l3a_by_cr.return_value = {
                    2091: {"glows_l3a_hist_20100101_v001.cdf"}
                }

                mock_query.side_effect = query_that_returns_empty_list_for_missing_ancillary

                actual_crs_to_process = GlowsInitializer.get_crs_to_process()

                mock_query.assert_has_calls([
                    call(instrument="glows", data_level="l3a", version="latest"),
                    call(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR"),
                    call(table="ancillary", instrument="glows", descriptor="WawHelioIonMP"),
                    call(table="ancillary", instrument="glows", descriptor="bad-days-list"),
                    call(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde")
                ])

                self.assertEqual([], actual_crs_to_process)

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.should_process_cr_candidate", return_value=True)
    @patch("imap_l3_processing.glows.glows_initializer.get_best_ancillary")
    @patch("imap_l3_processing.glows.glows_initializer.imap_data_access.query")
    @patch("imap_l3_processing.glows.glows_initializer.download_external_dependency")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializer.group_l3a_by_cr")
    def test_get_crs_to_process_returns_no_crs_when_missing_external_deps(self, mock_group_l3a_by_cr, mock_download_external_dep, __, ___, ____):
        test_cases = [
            F107_FLUX_TABLE_URL,
            LYMAN_ALPHA_COMPOSITE_INDEX_URL,
            OMNI2_URL,
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            fake_external_dep_path = temp_dir / 'fake_external_dep'
            fake_external_dep_path.write_text("some content\nsome other content")

            for missing_external_dependency in test_cases:
                with self.subTest(missing_external_dependency):
                    def download_that_returns_none_for_missing_dependency(url, *args):
                        return None if url == missing_external_dependency else fake_external_dep_path

                    mock_group_l3a_by_cr.return_value = {2091: {"glows_l3a_hist_20100101_v001.cdf"}}

                    mock_download_external_dep.side_effect = download_that_returns_none_for_missing_dependency

                    actual_crs_to_process = GlowsInitializer.get_crs_to_process()

                    self.assertEqual([], actual_crs_to_process)



    def test_should_process_cr_excludes_crs_within_buffer_period(self):
        cr_candidate = CRToProcess(
            l3a_file_names=set(),
            uv_anisotropy_file_name="uv_anisotropy.dat",
            waw_helio_ion_mp_file_name="waw_helio_ion.dat",
            bad_days_list_file_name="bad_days_list.dat",
            pipeline_settings_file_name="pipeline_settings.json",
            cr_start_date=datetime(2012, 2, 6, 3, 36, 31, 680000),
            cr_end_date=datetime(2012, 3, 4, 10, 12, 57, 600000),
            cr_rotation_number=2120,
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=False)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

        self.assertIsNone(GlowsInitializer.should_process_cr_candidate(cr_candidate, {}))

    def test_should_process_cr_no_existing_l3b(self):

        cr_candidate = CRToProcess(
            l3a_file_names=set(),
            uv_anisotropy_file_name="uv_anisotropy.dat",
            waw_helio_ion_mp_file_name="waw_helio_ion.dat",
            bad_days_list_file_name="bad_days_list.dat",
            pipeline_settings_file_name="pipeline_settings.json",
            cr_start_date=datetime(2010, 1, 1),
            cr_end_date=datetime(2010, 2, 1),
            cr_rotation_number=2091,
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

        actual_version = GlowsInitializer.should_process_cr_candidate(cr_candidate, {})

        self.assertEqual(1, actual_version)

    @patch("imap_l3_processing.glows.glows_initializer.read_cdf_parents")
    def test_should_process_cr_existing_cr_should_process(self, mock_read_cdf_parents):

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
                    f107_index_file_path=Path("f107_index_file_path"),
                    lyman_alpha_path=Path("lyman_alpha_path"),
                    omni2_data_path=Path("omni2_data_path"),
                    cr_start_date=datetime(2010, 1, 1),
                    cr_end_date=datetime(2010, 2, 1),
                    cr_rotation_number=2091,
                )

                cr_candidate = dataclasses.replace(already_processed_cr, **new_server_data)
                cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
                cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

                mock_read_cdf_parents.return_value = {
                    "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                    "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf",
                    "uv_anisotropy.dat",
                    "waw_helio_ion.dat",
                    "bad_days_list.dat",
                    "pipeline_settings.json",
                    "some_zip_file.zip"
                }

                l3bs = {2091: f"imap_glows_l3b_ion-rate-profile_20100101-cr02091_v00{existing_file_version}.cdf"}
                actual_version = GlowsInitializer.should_process_cr_candidate(cr_candidate, l3bs)

                mock_read_cdf_parents.assert_called_once_with(f"imap_glows_l3b_ion-rate-profile_20100101-cr02091_v00{existing_file_version}.cdf")

                self.assertEqual(existing_file_version + 1, actual_version)

    @patch("imap_l3_processing.glows.glows_initializer.read_cdf_parents")
    def test_should_process_cr_existing_cr_should_not_reprocess(self, mock_read_cdf_parents):
        cr_candidate = CRToProcess(
            l3a_file_names={
                "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
                "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"
            },
            uv_anisotropy_file_name="uv_anisotropy.dat",
            waw_helio_ion_mp_file_name="waw_helio_ion.dat",
            bad_days_list_file_name="bad_days_list.dat",
            pipeline_settings_file_name="pipeline_settings.json",
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
            cr_start_date=datetime(2010, 1, 1),
            cr_end_date=datetime(2010, 2, 1),
            cr_rotation_number=2091,
        )
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

        l3bs = {2091: "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf"}

        mock_read_cdf_parents.return_value = {
            "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
            "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf",
            "uv_anisotropy.dat",
            "waw_helio_ion.dat",
            "bad_days_list.dat",
            "pipeline_settings.json",
            "some_zip_file.zip"
        }

        self.assertIsNone(GlowsInitializer.should_process_cr_candidate(cr_candidate, l3bs))

        mock_read_cdf_parents.assert_called_once_with("imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf")

    def test_should_process_cr_with_invalid_dependencies(self):
        cr_candidate = CRToProcess(
            l3a_file_names=set(),
            uv_anisotropy_file_name="uv_anisotropy.dat",
            waw_helio_ion_mp_file_name="waw_helio_ion.dat",
            bad_days_list_file_name="bad_days_list.dat",
            pipeline_settings_file_name="pipeline_settings.json",
            cr_start_date=datetime(2010, 1, 1),
            cr_end_date=datetime(2010, 2, 1),
            cr_rotation_number=2091,
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=False)

        self.assertIsNone(GlowsInitializer.should_process_cr_candidate(cr_candidate, {}))

        cr_candidate.has_valid_external_dependencies.assert_called_once()

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

    def test_comment_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = Path(tmpdir) / "flux_table.txt"
            with open(file_name, "w") as fp:
                fp.writelines([
                    "fluxdate    fluxtime    fluxjulian    fluxcarrington  fluxobsflux  fluxadjflux  fluxursi\n",
                    "----------  ----------  ------------  --------------  -----------  -----------  ----------\n",
                    "20041028    170000      02453307.229  002022.605      000132.7     000130.9     000117.8\n",
                    "20041028    200000      02453307.354  002022.610      000135.8     000134.0     000120.6\n"])

            _comment_headers(file_name)
            with open(file_name, "r") as fp:
                lines = fp.readlines()
                self.assertEqual("#", lines[0][0])
                self.assertEqual("#", lines[1][0])



