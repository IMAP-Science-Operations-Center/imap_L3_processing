import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call, sentinel, Mock, create_autospec

import numpy as np
from imap_data_access import RepointInput
from imap_data_access.file_validation import Version

from imap_l3_processing.glows.descriptors import GLOWS_L3BCDE_DESCRIPTORS, GLOWS_L3E_HI_45_DESCRIPTOR, \
    GLOWS_L3E_HI_90_DESCRIPTOR, GLOWS_L3E_LO_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, GLOWS_L3E_ULTRA_HF_DESCRIPTOR, \
    GLOWS_L3E_DESCRIPTORS
from imap_l3_processing.glows.l3d.models import GlowsL3DProcessorOutput
from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializer, GlowsL3EInitializerOutput, \
    find_first_updated_cr, identify_versions_for_l3e_output_files
from imap_l3_processing.glows.l3e.glows_l3e_utils import GlowsL3eVersionsForRepointings
from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo
from imap_l3_processing.models import VersionMap
from tests.test_helpers import create_mock_query_results, get_test_data_path

MODULE = 'imap_l3_processing.glows.l3e.glows_l3e_initializer'

class TestGlowsL3EInitializer(unittest.TestCase):
    @patch(f'{MODULE}.get_pointing_date_range')
    @patch(f'{MODULE}.GlowsL3EDependencies.fetch_dependencies')
    @patch(f'{MODULE}.identify_versions_for_l3e_output_files')
    @patch(f'{MODULE}.find_first_updated_cr')
    @patch(f'{MODULE}.get_most_recently_uploaded_ancillary')
    @patch(f'{MODULE}.imap_data_access.query')
    @patch(f'{MODULE}.GlowsL3EDependencies.collect_spice_dependencies')
    def test_get_repointings_to_process(self, mock_collect_spice_dependencies, mock_query, mock_get_most_recently_uploaded_ancillary,
                                        mock_find_first_updated_cr, mock_identify_versions_for_l3e_output_files,
                                        mock_fetch_dependencies, mock_get_pointing_date_range):
        mock_query.side_effect = create_mock_query_results([
            'imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf',
            'imap_glows_energy-grid-lo_20200101_v000.cdf',
            'imap_glows_tess-xyz-8_20200101_v000.cdf',
            'imap_glows_energy-grid-hi_20200101_v000.cdf',
            'imap_glows_energy-grid-ultra_20200101_v000.cdf',
            'imap_glows_tess-ang-16_20200101_v000.cdf',
        ])

        mock_get_most_recently_uploaded_ancillary.side_effect = [
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-lo_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-xyz-8_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-hi_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-ultra_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-ang-16_20200101_v000.cdf'])[0],
        ]

        mock_spice_with_predict, mock_spice_without_predict = Mock(), Mock()

        mock_collect_spice_dependencies.return_value = (mock_spice_with_predict, mock_spice_without_predict)

        updated_l3d = Path('path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf')
        updated_l3d_text_file_path = Path("imap_glows_e-dens_19470303_20100101_v000.dat")
        glows_l3d_processor_output = GlowsL3DProcessorOutput(updated_l3d, [updated_l3d_text_file_path], sentinel.last_processed_cr)
        previous_l3d = 'imap_glows_l3d_solar-hist_19470303-cr02090_v000.cdf'

        mock_find_first_updated_cr.return_value = 2091

        mock_l3e_dependencies = mock_fetch_dependencies.return_value
        mock_l3e_dependencies.pipeline_settings = {"start_cr": sentinel.start_of_mission_cr}
        mock_l3e_dependencies.repointing_file = Path('path/to/repointing_file')

        expected_hi_45 = {1234: Version(None, 1), 2468: Version(None, 1)}
        expected_hi_90 = {1234: Version(None, 2), 2468: Version(None, 2)}
        expected_lo = {1234: Version(None, 3), 2468: Version(None, 3)}
        expected_ultra = {1234: Version(None, 4), 2468: Version(None, 4)}

        expected_repointings = GlowsL3eVersionsForRepointings(
            repointing_numbers=[2468, 1234],
            hi_90_repointings=expected_hi_90,
            hi_45_repointings=expected_hi_45,
            lo_repointings=expected_lo,
            ultra_hf_repointings=expected_ultra,
            ultra_sf_repointings=expected_ultra,
        )

        expected_initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_l3e_dependencies,
            repointings=expected_repointings,
            l3d_cdf_path=updated_l3d,
            metakernel_without_predict_ephem=mock_spice_without_predict,
            metakernel_with_predict_ephem=mock_spice_with_predict,
        )

        mock_identify_versions_for_l3e_output_files.return_value = expected_repointings

        mock_get_pointing_date_range.side_effect = [
            (datetime(2010, 1, 1), datetime(2010, 1, 2)),
            (datetime(2011, 2, 1), datetime(2011, 2, 2)),
        ]

        input_major_version = VersionMap(
            {descriptor: Version(2, 1) for descriptor in GLOWS_L3BCDE_DESCRIPTORS}
        )

        repointing_file_path = Path("imap_2026_105_01.repoint.csv")
        actual_initializer_output = GlowsL3EInitializer.get_repointings_to_process(
            glows_l3d_processor_output,
            previous_l3d,
            repointing_file_path,
            input_major_version,
            sentinel.reprocess_info,
        )

        mock_find_first_updated_cr.assert_called_once_with(updated_l3d, previous_l3d)

        mock_identify_versions_for_l3e_output_files.assert_called_once_with(sentinel.start_of_mission_cr, sentinel.last_processed_cr, 2090, repointing_file_path,
                                                                            input_major_version, sentinel.reprocess_info)

        mock_query.assert_has_calls([
            call(table="ancillary", instrument='glows', descriptor='pipeline-settings-l3bcde'),
            call(table="ancillary", instrument='glows', descriptor='energy-grid-lo'),
            call(table="ancillary", instrument='glows', descriptor='tess-xyz-8'),
            call(table="ancillary", instrument='glows', descriptor='energy-grid-hi'),
            call(table="ancillary", instrument='glows', descriptor='energy-grid-ultra'),
            call(table="ancillary", instrument='glows', descriptor='tess-ang-16'),
        ])

        mock_l3e_dependencies.copy_dependencies.assert_called_once()

        [fetch_dependencies_call] = mock_fetch_dependencies.call_args_list

        [actual_l3e_inputs] = fetch_dependencies_call.args

        pipeline_l3e_input_paths = actual_l3e_inputs.get_file_paths(source="glows")
        pipeline_l3e_input_filenames = [p.name for p in pipeline_l3e_input_paths]

        self.assertEqual([
            updated_l3d.name,
            "imap_glows_e-dens_19470303_20100101_v000.dat",
            'imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf',
            'imap_glows_energy-grid-lo_20200101_v000.cdf',
            'imap_glows_tess-xyz-8_20200101_v000.cdf',
            'imap_glows_energy-grid-hi_20200101_v000.cdf',
            'imap_glows_energy-grid-ultra_20200101_v000.cdf',
            'imap_glows_tess-ang-16_20200101_v000.cdf',
        ], pipeline_l3e_input_filenames)

        [repoint_input] = actual_l3e_inputs.get_file_paths(data_type=RepointInput.data_type)
        self.assertEqual("imap_2026_105_01.repoint.csv", repoint_input.name)

        self.assertEqual(actual_initializer_output, expected_initializer_data)

        mock_get_pointing_date_range.assert_has_calls([
            call(1234),
            call(2468)
        ])

        mock_collect_spice_dependencies.assert_called_once_with(
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2011, 2, 2),
        )

    @patch(f'{MODULE}.get_most_recently_uploaded_ancillary')
    @patch(f'{MODULE}.imap_data_access')
    @patch(f'{MODULE}.GlowsL3EDependencies.fetch_dependencies')
    @patch(f'{MODULE}.identify_versions_for_l3e_output_files')
    @patch(f'{MODULE}.find_first_updated_cr')
    def test_get_repointings_to_process_identical_l3d_files(
        self,
        mock_find_first_updated_cr,
        mock_identify_versions_for_l3e_output_files,
        mock_fetch_dependencies,
        mock_imap_data_access,
        mock_get_most_recently_uploaded_ancillary,
    ):
        updated_l3d = Path(
            "path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf"
        )
        updated_l3d_text_file_path = Path(
            "imap_glows_e-dens_19470303_20100101_v000.dat"
        )
        expected_last_cr = 2091
        glows_l3d_processor_output = GlowsL3DProcessorOutput(
            updated_l3d, [updated_l3d_text_file_path], expected_last_cr
        )
        previous_l3d = "imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf"

        mock_l3e_dependencies = mock_fetch_dependencies.return_value
        mock_l3e_dependencies.pipeline_settings = {
            "start_cr": sentinel.start_of_mission_cr
        }
        mock_find_first_updated_cr.return_value = None
        mock_identify_versions_for_l3e_output_files.return_value = GlowsL3eVersionsForRepointings(
            repointing_numbers=[],
            hi_90_repointings={},
            hi_45_repointings={},
            lo_repointings={},
            ultra_sf_repointings={},
            ultra_hf_repointings={},
        )
        mock_get_most_recently_uploaded_ancillary.side_effect = [
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-lo_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-xyz-8_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-hi_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-ultra_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-ang-16_20200101_v000.cdf'])[0],
        ]

        repointing_file_path = Path("imap_2026_105_01.repoint.csv")
        actual_initializer_output = GlowsL3EInitializer.get_repointings_to_process(
            glows_l3d_processor_output,
            previous_l3d,
            repointing_file_path,
            sentinel.version_map,
            sentinel.reprocess_info,
        )
        mock_find_first_updated_cr.assert_called_once_with(glows_l3d_processor_output.l3d_cdf_file_path, previous_l3d)
        mock_identify_versions_for_l3e_output_files.assert_called_once_with(
            sentinel.start_of_mission_cr,
            expected_last_cr,
            None,
            repointing_file_path,
            sentinel.version_map,
            sentinel.reprocess_info,
        )
        self.assertIsNone(actual_initializer_output)

    @patch(f'{MODULE}.imap_data_access.query')
    @patch(f'{MODULE}.get_most_recently_uploaded_ancillary')
    @patch(f'{MODULE}.GlowsL3EDependencies.fetch_dependencies')
    @patch(f'{MODULE}.identify_versions_for_l3e_output_files')
    @patch(f'{MODULE}.find_first_updated_cr')
    def test_get_repointings_to_process_uses_mission_start_cr_when_no_previous_l3d(self, mock_find_first_updated_cr,
                                                                                   mock_identify_versions_for_l3e_output_files,
                                                                                   mock_fetch_dependencies,
                                                                                   mock_get_most_recently_uploaded_ancillary,
                                                                                   _, ):
        mock_get_most_recently_uploaded_ancillary.side_effect = [
            create_mock_query_results(['imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-lo_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-xyz-8_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-hi_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_energy-grid-ultra_20200101_v000.cdf'])[0],
            create_mock_query_results(['imap_glows_tess-ang-16_20200101_v000.cdf'])[0],
        ]
        mock_identify_versions_for_l3e_output_files.return_value = GlowsL3eVersionsForRepointings(
            repointing_numbers=[],
            ultra_sf_repointings={},
            ultra_hf_repointings={},
            lo_repointings={},
            hi_45_repointings={},
            hi_90_repointings={}
        )

        updated_l3d = Path('path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v012.0001.cdf')
        glows_l3d_processor_output = GlowsL3DProcessorOutput(updated_l3d, [], sentinel.last_processed_cr)
        previous_l3d = None

        mock_fetch_dependencies.return_value.pipeline_settings = {"start_cr": sentinel.start_of_mission_cr}

        repointing_file_path = Path("imap_2026_105_01.repoint.csv")
        _ = GlowsL3EInitializer.get_repointings_to_process(
            glows_l3d_processor_output, previous_l3d,
            repointing_file_path, sentinel.version_map,
            sentinel.reprocess_info,
        )

        mock_find_first_updated_cr.assert_not_called()
        mock_identify_versions_for_l3e_output_files.assert_called_once_with(
            sentinel.start_of_mission_cr,
            sentinel.last_processed_cr,
            sentinel.start_of_mission_cr,
            repointing_file_path,
            sentinel.version_map,
            sentinel.reprocess_info,
        )

    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.CDF')
    def test_find_first_updated_cr(self, mock_CDF, mock_download):
        num_crs = 10
        mock_download.return_value = sentinel.download_old_l3d

        old_l3d = {
            'cr_grid': np.arange(num_crs) + 0.5,
            'lyman_alpha': np.arange(num_crs),
            'phion': np.arange(num_crs),
            'plasma_speed': np.arange(0, 20).reshape((num_crs, 2)),
            'plasma_speed_flag': np.arange(num_crs),
            'proton_density': np.arange(0, 20).reshape((num_crs, 2)),
            'proton_density_flag': np.arange(num_crs),
            'uv_anisotropy': np.arange(0, 20).reshape((num_crs, 2)),
            'uv_anisotropy_flag': np.arange(num_crs),
            'glows_flags': np.arange(num_crs),
        }

        new_lyman_alpha = np.arange(num_crs)
        new_lyman_alpha[1] = 10

        new_phion = np.arange(num_crs)
        new_phion[2] = 10

        new_plasma_speed_flag = np.arange(num_crs)
        new_plasma_speed_flag[3] = 10

        new_proton_density_flag = np.arange(num_crs)
        new_proton_density_flag[4] = 10

        new_uv_anisotropy_flag = np.arange(num_crs)
        new_uv_anisotropy_flag[5] = 10

        new_plasma_speed = np.arange(0, 20).reshape((num_crs, 2))
        new_plasma_speed[6, :] = 10

        new_proton_density = np.arange(0, 20).reshape((num_crs, 2))
        new_proton_density[7, :] = 10

        new_uv_anisotropy = np.arange(0, 20).reshape((num_crs, 2))
        new_uv_anisotropy[8, :] = 10

        new_glows_flags = np.arange(num_crs)
        new_glows_flags[9] = 10

        cases = [
            ('cr_grid', np.append(old_l3d['cr_grid'], 10.5), 10),
            ('lyman_alpha', new_lyman_alpha, 1),
            ('phion', new_phion, 2),
            ('plasma_speed_flag', new_plasma_speed_flag, 3),
            ('proton_density_flag', new_proton_density_flag, 4),
            ('uv_anisotropy_flag', new_uv_anisotropy_flag, 5),

            ('plasma_speed', new_plasma_speed, 6),
            ('proton_density', new_proton_density, 7),
            ('uv_anisotropy', new_uv_anisotropy, 8),
            ('glows_flags', new_glows_flags, 9),
            ('no_change', None, None)
        ]

        for case, change, expected in cases:
            mock_download.reset_mock()
            mock_CDF.reset_mock()

            with self.subTest(case=case):
                new_l3d = {**old_l3d}
                if case != "no_change":
                    new_l3d[case] = change

                mock_CDF.side_effect = [old_l3d, new_l3d]

                actual_cr = find_first_updated_cr(
                    sentinel.new_l3d_path, sentinel.old_l3d_filename
                )

                mock_download.assert_called_once_with(sentinel.old_l3d_filename)
                mock_CDF.assert_has_calls(
                    [
                        call(str(sentinel.download_old_l3d)),
                        call(str(sentinel.new_l3d_path)),
                    ]
                )

                self.assertEqual(actual_cr, expected)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access')
    def test_identify_versions_for_l3e_output_files_gives_minor_version_1_for_non_existing_l3e(self,
                                                                                               mock_imap_data_access,
                                                                                               mock_get_repoint_numbers_within_cr_window):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2094
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
        ]

        all_repointing_numbers = set(range(3682, 3736))
        updated_repointing_numbers = set()
        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]
        reprocess_info = ReprocessInfo({})

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission, first_cr_updated_in_l3d,
                                                        repointing_path, version_map, reprocess_info)

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        expected_versions_for_hi45_repoint_number = {repoint_number: Version(3, 1) for repoint_number in
                                                     all_repointing_numbers}
        expected_versions_for_hi90_repoint_number = {repoint_number: Version(4, 1) for repoint_number in
                                                     all_repointing_numbers}
        expected_versions_for_lo_repoint_number = {repoint_number: Version(5, 1) for repoint_number in
                                                   all_repointing_numbers}
        expected_versions_for_ultra_sf_repoint_number = {repoint_number: Version(6, 1) for repoint_number in
                                                         all_repointing_numbers}
        expected_versions_for_ultra_hf_repoint_number = {repoint_number: Version(7, 1) for repoint_number in
                                                         all_repointing_numbers}

        self.assertCountEqual(all_repointing_numbers, result.repointing_numbers)
        self.assertEqual(expected_versions_for_hi90_repoint_number, result.hi_90_repointings)
        self.assertEqual(expected_versions_for_hi45_repoint_number, result.hi_45_repointings)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)
        self.assertEqual(
            expected_versions_for_ultra_sf_repoint_number, result.ultra_sf_repointings
        )
        self.assertEqual(expected_versions_for_ultra_hf_repoint_number, result.ultra_hf_repointings)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access')
    def test_identify_versions_for_l3e_output_files_increments_major_and_minor_when_given_higher_major_version(self,
                                                                                                               mock_imap_data_access,
                                                                                                               mock_get_repoint_numbers_within_cr_window):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2094
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})

        all_repointing_numbers = set(range(3682, 3736))
        updated_repointing_numbers = set()
        old_major_version = 2
        mock_get_repoint_numbers_within_cr_window.reset_mock()
        mock_imap_data_access.reset_mock()

        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03682_{Version(old_major_version, 1)}.cdf',
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03683_{Version(3, 1)}.cdf',
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03735_{Version(3, 1)}.cdf'
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03683_{Version(old_major_version, 2)}.cdf',
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03684_{Version(4, 2)}.cdf',
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03735_{Version(4, 2)}.cdf'
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint03684_{Version(old_major_version, 3)}.cdf',
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint03685_{Version(5, 3)}.cdf',
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint03735_{Version(5, 3)}.cdf'
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint03685_{Version(old_major_version, 4)}.cdf',
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint03686_{Version(6, 4)}.cdf',
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint03735_{Version(6, 4)}.cdf'
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint03686_{Version(old_major_version, 5)}.cdf',
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint03687_{Version(7, 5)}.cdf',
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint03735_{Version(7, 5)}.cdf'
            ])
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission,
                                                        first_cr_updated_in_l3d, repointing_path,
                                                        version_map, ReprocessInfo({}))

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        self.assertCountEqual(list(range(3682, 3735)), result.repointing_numbers)

        self.assertNotIn(3683, result.hi_45_repointings)
        self.assertNotIn(3684, result.hi_90_repointings)
        self.assertNotIn(3685, result.lo_repointings)
        self.assertNotIn(3686, result.ultra_sf_repointings)
        self.assertNotIn(3687, result.ultra_hf_repointings)

        self.assertEqual(Version(3, 2), result.hi_45_repointings[3682])
        self.assertEqual(Version(4, 3), result.hi_90_repointings[3683])
        self.assertEqual(Version(5, 4), result.lo_repointings[3684])
        self.assertEqual(Version(6, 5), result.ultra_sf_repointings[3685])
        self.assertEqual(Version(7, 6), result.ultra_hf_repointings[3686])

    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access')
    def test_identify_versions_for_l3e_output_files_increments_minor_when_same_major_and_updated_l3d_covers_pointing(
            self, mock_imap_data_access, mock_get_repoint_numbers_within_cr_window
    ):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2095
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})

        all_repointing_numbers = set(range(3682, 3763))
        updated_repointing_numbers = set(range(3709, 3763))

        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint{repoint:05d}_{Version(3, 1)}.cdf' for
                repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint{repoint:05d}_{Version(4, 2)}.cdf' for
                repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint{repoint:05d}_{Version(5, 3)}.cdf' for repoint
                in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint{repoint:05d}_{Version(6, 4)}.cdf' for
                repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint{repoint:05d}_{Version(7, 5)}.cdf' for
                repoint in all_repointing_numbers
            ])
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission,
                                                        first_cr_updated_in_l3d, repointing_path,
                                                        version_map, ReprocessInfo({}))

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        expected_versions_for_hi45_repoint_number = {repoint_number: Version(3, 2) for repoint_number in
                                                     updated_repointing_numbers}
        expected_versions_for_hi90_repoint_number = {repoint_number: Version(4, 3) for repoint_number in
                                                     updated_repointing_numbers}
        expected_versions_for_lo_repoint_number = {repoint_number: Version(5, 4) for repoint_number in
                                                   updated_repointing_numbers}
        expected_versions_for_ultra_sf_repoint_number = {repoint_number: Version(6, 5) for repoint_number in
                                                         updated_repointing_numbers}
        expected_versions_for_ultra_hf_repoint_number = {repoint_number: Version(7, 6) for repoint_number in
                                                         updated_repointing_numbers}

        self.assertCountEqual(updated_repointing_numbers, result.repointing_numbers)
        self.assertEqual(expected_versions_for_hi90_repoint_number, result.hi_90_repointings)
        self.assertEqual(expected_versions_for_hi45_repoint_number, result.hi_45_repointings)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)
        self.assertEqual(expected_versions_for_ultra_sf_repoint_number, result.ultra_sf_repointings)
        self.assertEqual(expected_versions_for_ultra_hf_repoint_number, result.ultra_hf_repointings)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_repoint_data')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access')
    def test_identify_versions_for_l3e_output_files_uses_reprocess_info(
            self, mock_imap_data_access, mock_get_repoint_numbers_within_cr_window, mock_get_repoint_data
    ):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2095
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})
        repoint_number = 3700
        mock_reprocess_info = create_autospec(ReprocessInfo, instance=True)
        mock_reprocess_info.get_repoints_for_descriptor.side_effect = [
            set(), set(), {repoint_number}, set(), set()
        ]

        all_repointing_numbers = set(range(3682, 3763))
        updated_repointing_numbers = set()

        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint{repoint:05d}_{Version(3, 1)}.cdf' for
                repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint{repoint:05d}_{Version(4, 2)}.cdf' for
                repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint{repoint:05d}_{Version(5, 3)}.cdf' for repoint
                in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint{repoint:05d}_{Version(6, 4)}.cdf' for
                repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint{repoint:05d}_{Version(7, 5)}.cdf' for
                repoint in all_repointing_numbers
            ])
        ]

        result = identify_versions_for_l3e_output_files(
            start_cr_of_mission,
            end_cr_of_mission,
            first_cr_updated_in_l3d,
            repointing_path,
            version_map,
            mock_reprocess_info,
        )

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])
        mock_reprocess_info.get_repoints_for_descriptor.assert_has_calls([
            call(GLOWS_L3E_HI_45_DESCRIPTOR, mock_get_repoint_data.return_value),
            call(GLOWS_L3E_HI_90_DESCRIPTOR, mock_get_repoint_data.return_value),
            call(GLOWS_L3E_LO_DESCRIPTOR, mock_get_repoint_data.return_value),
            call(GLOWS_L3E_ULTRA_SF_DESCRIPTOR, mock_get_repoint_data.return_value),
            call(GLOWS_L3E_ULTRA_HF_DESCRIPTOR, mock_get_repoint_data.return_value),
        ])

        expected_versions_for_lo_repoint_number = {repoint_number: Version(5, 4)}

        self.assertEqual([repoint_number], result.repointing_numbers)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)