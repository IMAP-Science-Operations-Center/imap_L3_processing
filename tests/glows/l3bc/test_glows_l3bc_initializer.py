import dataclasses
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, call, sentinel

import numpy as np
from imap_data_access import ProcessingInputCollection, RepointInput

from imap_l3_processing.glows.l3bc.glows_l3bc_initializer import GlowsL3BCInitializer, GlowsL3BCInitializerData
from imap_l3_processing.glows.l3bc.models import CRToProcess, ExternalDependencies
from tests.test_helpers import create_glows_mock_query_results


class TestGlowsL3BCInitializer(unittest.TestCase):
    def setUp(self):
        self.external_dependencies_fetch_patcher = patch(
            "imap_l3_processing.glows.l3bc.glows_l3bc_initializer.ExternalDependencies.fetch_dependencies")
        self.mock_external_dependencies_fetch = self.external_dependencies_fetch_patcher.start()
        
        self.download_patcher = patch(
            "imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.download")
        self.download_patcher.start()
        
        self.set_global_repoint_table_paths_patcher = patch(
            'imap_l3_processing.glows.l3bc.glows_l3bc_initializer.set_global_repoint_table_paths')
        self.set_global_repoint_table_paths_patcher.start()

    def tearDown(self):
        self.external_dependencies_fetch_patcher.stop()
        self.download_patcher.stop()
        self.set_global_repoint_table_paths_patcher.stop()

    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_initializer.set_global_repoint_table_paths')
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_initializer.GlowsL3BCDependencies.download_from_cr_to_process')
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.ExternalDependencies.fetch_dependencies")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.get_date_range_of_cr")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.get_best_ancillary")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.query")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.GlowsL3BCInitializer.group_l3a_by_cr")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.GlowsL3BCInitializer.should_process_cr_candidate")
    def test_get_crs_to_process(self, mock_should_process_cr_candidate, mock_group_l3a_by_cr, mock_query,
                                mock_get_best_ancillary, mock_get_date_range_of_cr, mock_fetch_external_deps,
                                mock_l3bc_deps_from_cr, mock_set_global_repoint_table_paths, mock_download):

        mock_query.side_effect = [
            create_glows_mock_query_results([
                "imap_glows_l3a_hist_20100101_v001.cdf",
                "imap_glows_l3a_hist_20100201_v001.cdf"
            ]),
            create_glows_mock_query_results([
                "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf",
                "imap_glows_l3b_ion-rate-profile_20100201-cr02092_v001.cdf"
            ]),
            create_glows_mock_query_results([
                "imap_glows_l3c_sw-profile_20100101-cr02091_v001.cdf",
                "imap_glows_l3c_sw-profile_20100201-cr02092_v001.cdf"
            ]),
            sentinel.uv_anisotropy_query_result,
            sentinel.waw_helio_ion_query_result,
            sentinel.bad_days_list_query_result,
            sentinel.pipeline_settings_query_result
        ]

        mock_group_l3a_by_cr.return_value = {
            2091: {"imap_glows_l3a_hist_20100101_v001.cdf"},
            2092: {"imap_glows_l3a_hist_20100201_v001.cdf"}
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

        expected_l3bs_by_cr = {
            2091: "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf",
            2092: "imap_glows_l3b_ion-rate-profile_20100201-cr02092_v001.cdf"
        }

        expected_l3cs_by_cr = {
            2091: "imap_glows_l3c_sw-profile_20100101-cr02091_v001.cdf",
            2092: "imap_glows_l3c_sw-profile_20100201-cr02092_v001.cdf"
        }

        l3bc_initializer_data = GlowsL3BCInitializer.get_crs_to_process(ProcessingInputCollection(RepointInput("imap_2026_269_05.repoint.csv")))

        mock_download.assert_called_once()
        mock_set_global_repoint_table_paths.assert_called_with([mock_download.return_value])

        mock_query.assert_has_calls([
            call(instrument="glows", data_level="l3a", descriptor="hist", version="latest"),
            call(instrument="glows", data_level="l3b", descriptor='ion-rate-profile', version="latest"),
            call(instrument="glows", data_level="l3c", descriptor='sw-profile', version="latest"),
            call(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR"),
            call(table="ancillary", instrument="glows", descriptor="WawHelioIonMP"),
            call(table="ancillary", instrument="glows", descriptor="bad-days-list"),
            call(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde"),
        ])

        mock_group_l3a_by_cr.assert_called_once_with([
            "imap_glows_l3a_hist_20100101_v001.cdf",
            "imap_glows_l3a_hist_20100201_v001.cdf",
        ])

        mock_fetch_external_deps.assert_called_once()

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
            l3a_file_names={"imap_glows_l3a_hist_20100101_v001.cdf"},
            uv_anisotropy_file_name="uv_anisotropy_best_ancillary_1",
            waw_helio_ion_mp_file_name="waw_helio_ion_best_ancillary_1",
            bad_days_list_file_name="bad_days_list_best_ancillary_1",
            pipeline_settings_file_name="pipeline_settings_best_ancillary_1",
            cr_start_date=datetime(2010, 1, 1), cr_end_date=datetime(2010, 1, 31),
            cr_rotation_number=2091,
        )

        expected_cr_candidate_2 = CRToProcess(
            l3a_file_names={"imap_glows_l3a_hist_20100201_v001.cdf"},
            uv_anisotropy_file_name="uv_anisotropy_best_ancillary_2",
            waw_helio_ion_mp_file_name="waw_helio_ion_best_ancillary_2",
            bad_days_list_file_name="bad_days_list_best_ancillary_2",
            pipeline_settings_file_name="pipeline_settings_best_ancillary_2",
            cr_start_date=datetime(2010, 2, 1), cr_end_date=datetime(2010, 2, 28),
            cr_rotation_number=2092,
        )

        mock_should_process_cr_candidate.assert_has_calls([
            call(expected_cr_candidate_1, expected_l3bs_by_cr, mock_fetch_external_deps.return_value),
            call(expected_cr_candidate_2, expected_l3bs_by_cr, mock_fetch_external_deps.return_value)
        ], any_order=False)

        mock_l3bc_deps_from_cr.assert_called_once_with(expected_cr_candidate_1, 2,
                                                       mock_fetch_external_deps.return_value,
                                                       mock_download.return_value)

        expected_initializer_data = GlowsL3BCInitializerData(
            external_dependencies=mock_fetch_external_deps.return_value,
            l3bc_dependencies=[mock_l3bc_deps_from_cr.return_value],
            l3bs_by_cr=expected_l3bs_by_cr,
            l3cs_by_cr=expected_l3cs_by_cr,
            repoint_file_path=mock_download.return_value,
        )

        self.assertEqual(expected_initializer_data, l3bc_initializer_data)

    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.download')
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.GlowsL3BCInitializer.group_l3a_by_cr")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.query")
    def test_get_crs_to_process_return_no_crs_when_missing_ancillaries(self, mock_query, mock_group_l3a_by_cr, mock_download):

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
                        return [{"start_date": "20100101", "end_date": None, "ingestion_date": "20100101 00:00:00",
                                 "file_path": f"some/ancillary/{missing_ancillary}"}]
                    else:
                        return []

                mock_group_l3a_by_cr.return_value = {
                    2091: {"glows_l3a_hist_20100101_v001.cdf"}
                }

                mock_query.side_effect = query_that_returns_empty_list_for_missing_ancillary

                actual_crs_to_process = GlowsL3BCInitializer.get_crs_to_process(ProcessingInputCollection(RepointInput("imap_2026_269_05.repoint.csv")))

                mock_query.assert_has_calls([
                    call(instrument="glows", data_level="l3a", descriptor="hist", version="latest"),
                    call(instrument="glows", data_level="l3b", descriptor="ion-rate-profile", version="latest"),
                    call(instrument="glows", data_level="l3c", descriptor="sw-profile", version="latest"),
                    call(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR"),
                    call(table="ancillary", instrument="glows", descriptor="WawHelioIonMP"),
                    call(table="ancillary", instrument="glows", descriptor="bad-days-list"),
                    call(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde")
                ])

                expected_initializer_result = GlowsL3BCInitializerData(
                    external_dependencies=self.mock_external_dependencies_fetch.return_value,
                    l3bc_dependencies=[],
                    l3bs_by_cr={},
                    l3cs_by_cr={},
                    repoint_file_path=mock_download.return_value
                )

                self.assertEqual(expected_initializer_result, actual_crs_to_process)

    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.GlowsL3BCInitializer.should_process_cr_candidate",
           return_value=True)
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.get_best_ancillary")
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.download')
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.imap_data_access.query")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.ExternalDependencies.fetch_dependencies")
    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.GlowsL3BCInitializer.group_l3a_by_cr")
    def test_get_crs_to_process_returns_no_crs_when_missing_external_deps(self, mock_group_l3a_by_cr,
                                                                          mock_fetch_ext_dependencies, mock_query, mock_download, ___,
                                                                          ____):
        test_cases = [
            ExternalDependencies(None, Path("some_lyman"), Path("some_omni")),
            ExternalDependencies(Path("some_f107"), None, Path("some_omni")),
            ExternalDependencies(Path("some_f107"), Path("some_lyman"), None),
        ]

        for external_deps in test_cases:
            with self.subTest(external_deps):
                mock_query.reset_mock()

                mock_fetch_ext_dependencies.return_value = external_deps
                mock_group_l3a_by_cr.return_value = {2091: {"glows_l3a_hist_20100101_v001.cdf"}}

                mock_query.side_effect = [
                    create_glows_mock_query_results(["imap_glows_l3a_hist_20100101_v001.cdf"]),
                    create_glows_mock_query_results(["imap_glows_l3b_ion-rate-profile_20100101-cr00001_v001.cdf"]),
                    create_glows_mock_query_results(["imap_glows_l3c_sw-profile_20100101-cr00001_v001.cdf"]),
                    create_glows_mock_query_results(["imap_glows_uv-anisotropy-1CR_20100101_v001.dat"]),
                    create_glows_mock_query_results(["imap_glows_WawHelioIonMP_20100101_v001.dat"]),
                    create_glows_mock_query_results(["imap_glows_bad-days-list_20100101_v001.dat"]),
                    create_glows_mock_query_results(["imap_glows_pipeline-settings-l3bcde_20100101_v001.json"])
                ]

                actual_crs_to_process = GlowsL3BCInitializer.get_crs_to_process(ProcessingInputCollection(RepointInput("imap_2026_269_05.repoint.csv")))

                mock_query.assert_has_calls([
                    call(instrument="glows", data_level="l3a", descriptor="hist", version="latest"),
                    call(instrument="glows", data_level="l3b", descriptor="ion-rate-profile", version="latest"),
                    call(instrument="glows", data_level="l3c", descriptor="sw-profile", version="latest"),
                    call(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR"),
                    call(table="ancillary", instrument="glows", descriptor="WawHelioIonMP"),
                    call(table="ancillary", instrument="glows", descriptor="bad-days-list"),
                    call(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde"),
                ])

                expected_initializer_result = GlowsL3BCInitializerData(
                    external_dependencies=external_deps,
                    l3bc_dependencies=[],
                    l3bs_by_cr={1: "imap_glows_l3b_ion-rate-profile_20100101-cr00001_v001.cdf"},
                    l3cs_by_cr={1: "imap_glows_l3c_sw-profile_20100101-cr00001_v001.cdf"},
                    repoint_file_path=mock_download.return_value
                )

                self.assertEqual(expected_initializer_result, actual_crs_to_process)

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

        )
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=False)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

        external_dependencies = ExternalDependencies(
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )

        self.assertIsNone(GlowsL3BCInitializer.should_process_cr_candidate(cr_candidate, {}, external_dependencies))

    def test_should_process_cr_no_existing_l3b(self):

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
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

        external_dependencies = ExternalDependencies(
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )

        actual_version = GlowsL3BCInitializer.should_process_cr_candidate(cr_candidate, {}, external_dependencies)

        self.assertEqual(1, actual_version)

    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.read_cdf_parents")
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
                    cr_start_date=datetime(2010, 1, 1),
                    cr_end_date=datetime(2010, 2, 1),
                    cr_rotation_number=2091,
                )

                external_dependencies = ExternalDependencies(
                    f107_index_file_path=Path("f107_index_file_path"),
                    lyman_alpha_path=Path("lyman_alpha_path"),
                    omni2_data_path=Path("omni2_data_path"),
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
                actual_version = GlowsL3BCInitializer.should_process_cr_candidate(cr_candidate, l3bs,
                                                                                  external_dependencies)

                mock_read_cdf_parents.assert_called_once_with(
                    f"imap_glows_l3b_ion-rate-profile_20100101-cr02091_v00{existing_file_version}.cdf")

                self.assertEqual(existing_file_version + 1, actual_version)

    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.read_cdf_parents")
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
            cr_start_date=datetime(2010, 1, 1),
            cr_end_date=datetime(2010, 2, 1),
            cr_rotation_number=2091,
        )
        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=True)

        l3bs = {2091: "imap_glows_l3b_ion-rate-profile_20100101-cr02091_v001.cdf"}

        external_dependencies = ExternalDependencies(
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )

        mock_read_cdf_parents.return_value = {
            "imap_glows_l3a_hist_20100101-repoint000001_v001.cdf",
            "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf",
            "uv_anisotropy.dat",
            "waw_helio_ion.dat",
            "bad_days_list.dat",
            "pipeline_settings.json",
            "some_zip_file.zip"
        }

        self.assertIsNone(GlowsL3BCInitializer.should_process_cr_candidate(cr_candidate, l3bs, external_dependencies))

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
        )

        cr_candidate.buffer_time_has_elapsed_since_cr = Mock(return_value=True)
        cr_candidate.has_valid_external_dependencies = Mock(return_value=False)

        external_dependencies = ExternalDependencies(
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )

        self.assertIsNone(GlowsL3BCInitializer.should_process_cr_candidate(cr_candidate, {}, external_dependencies))

        cr_candidate.has_valid_external_dependencies.assert_called_once_with(external_dependencies)

    @patch("imap_l3_processing.glows.l3bc.glows_l3bc_initializer.get_pointing_date_range")
    def test_group_l3a_by_cr(self, mock_get_pointing_date_range):
        mock_get_pointing_date_range.side_effect = [
            (datetime(2010, 1, 1), datetime(2010, 1, 2)),
            (datetime(2010, 1, 2), datetime(2010, 1, 3)),
            (datetime(2010, 5, 1), datetime(2010, 5, 2)),
            (datetime(2010, 5, 2), datetime(2010, 5, 3)),
            (datetime(2010, 8, 1), datetime(2010, 8, 2)),
            (datetime(2010, 8, 9), datetime(2010, 8, 10)),
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

        grouped_l3a_files = GlowsL3BCInitializer.group_l3a_by_cr(l3a_files_and_parents)

        self.assertEqual(expected, grouped_l3a_files)
