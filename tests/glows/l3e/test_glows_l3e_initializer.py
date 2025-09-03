import unittest
from pathlib import Path
from unittest.mock import patch, call
from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializer, GlowsL3EInitializerOutput
from imap_l3_processing.glows.l3e.glows_l3e_utils import GlowsL3eRepointings
from tests.test_helpers import create_glows_mock_query_results


class TestGlowsL3EInitializer(unittest.TestCase):
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.GlowsL3EDependencies.fetch_dependencies')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.determine_l3e_files_to_produce')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.find_first_updated_cr')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_most_recently_uploaded_ancillary')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access.query')
    def test_get_repointings_to_process(self, mock_query, mock_get_most_recently_uploaded_ancillary,
                                            mock_find_first_updated_cr, mock_determine_l3e_files_to_produce,
                                            mock_fetch_dependencies):
        mock_query.side_effect = create_glows_mock_query_results([
            'imap_glows_ionization-files_20200101_v000.cdf',
            'imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf',
            'imap_glows_energy-grid-lo_20200101_v000.cdf',
            'imap_glows_tess-xyz-8_20200101_v000.cdf',
            'imap_glows_elongation-data_20200101_v000.cdf',
            'imap_glows_energy-grid-hi_20200101_v000.cdf',
            'imap_glows_energy-grid-ultra_20200101_v000.cdf',
            'imap_glows_tess-ang-16_20200101_v000.cdf',
        ])

        mock_get_most_recently_uploaded_ancillary.side_effect = [
            Path('imap_glows_ionization-files_20200101_v000.cdf'),
            Path('imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'),
            Path('imap_glows_energy-grid-lo_20200101_v000.cdf'),
            Path('imap_glows_tess-xyz-8_20200101_v000.cdf'),
            Path('imap_glows_elongation-data_20200101_v000.cdf'),
            Path('imap_glows_energy-grid-hi_20200101_v000.cdf'),
            Path('imap_glows_energy-grid-ultra_20200101_v000.cdf'),
            Path('imap_glows_tess-ang-16_20200101_v000.cdf'),
        ]

        updated_l3d = Path('path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf')
        previous_l3d = Path('path/to/previous_l3d')

        mock_find_first_updated_cr.return_value = 2090

        mock_fetch_dependencies.return_value.repointing_file = Path('path/to/repointing_file')

        expected_hi_45 = {1234: 1}
        expected_hi_90 = {1234: 2}
        expected_lo = {1234: 3}
        expected_ultra = {1234: 4}

        expected_repointings = GlowsL3eRepointings(
                repointing_numbers=[1234],
                hi_90_repointings=expected_hi_90,
                hi_45_repointings=expected_hi_45,
                lo_repointings=expected_lo,
                ultra_repointings=expected_ultra
        )

        expected_initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_fetch_dependencies.return_value,
            repointings=expected_repointings
        )

        mock_determine_l3e_files_to_produce.return_value = expected_repointings

        actual_initializer_output = GlowsL3EInitializer.get_repointings_to_process(updated_l3d, previous_l3d)

        mock_find_first_updated_cr.assert_called_once_with(updated_l3d, previous_l3d)

        mock_determine_l3e_files_to_produce.assert_called_once_with(2090, 2091, actual_initializer_output.dependencies.repointing_file)

        mock_query.assert_has_calls([
            call(instrument='glows', descriptor='ionization-files'),
            call(instrument='glows', descriptor='pipeline-settings-l3bcde'),
            call(instrument='glows', descriptor='energy-grid-lo'),
            call(instrument='glows', descriptor='tess-xyz-8'),
            call(instrument='lo', descriptor='elongation-data'),
            call(instrument='glows', descriptor='energy-grid-hi'),
            call(instrument='glows', descriptor='energy-grid-ultra'),
            call(instrument='glows', descriptor='tess-ang-16'),
        ])

        mock_get_most_recently_uploaded_ancillary.assert_has_calls([call(query_result) for query_result in mock_query.side_effect])

        mock_fetch_dependencies.return_value.rename_dependencies.assert_called_once()

        [fetch_dependencies_call] = mock_fetch_dependencies.call_args_list

        [actual_l3e_inputs] = fetch_dependencies_call.args

        pipeline_l3e_input_paths = actual_l3e_inputs.get_file_paths(source="glows")
        pipeline_l3e_input_filenames = [p.name for p in pipeline_l3e_input_paths]

        self.assertEqual([
            updated_l3d.name,
            'imap_glows_ionization-files_20200101_v000.cdf',
            'imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf',
            'imap_glows_energy-grid-lo_20200101_v000.cdf',
            'imap_glows_tess-xyz-8_20200101_v000.cdf',
            'imap_glows_elongation-data_20200101_v000.cdf',
            'imap_glows_energy-grid-hi_20200101_v000.cdf',
            'imap_glows_energy-grid-ultra_20200101_v000.cdf',
            'imap_glows_tess-ang-16_20200101_v000.cdf',
        ], pipeline_l3e_input_filenames)

        self.assertEqual(actual_initializer_output, expected_initializer_data)


    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access.query')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_most_recently_uploaded_ancillary')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.GlowsL3EDependencies.fetch_dependencies')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.determine_l3e_files_to_produce')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.find_first_updated_cr')
    def test_get_repointings_to_process_handles_no_previous_l3d(self, mock_find_first_updated_cr,
                                                                mock_determine_l3e_files_to_produce,
                                                                mock_fetch_dependencies,
                                                                mock_get_most_recently_uploaded_ancillary,
                                                                _,):
        mock_get_most_recently_uploaded_ancillary.side_effect = [
            Path('imap_glows_ionization-files_20200101_v000.cdf'),
            Path('imap_glows_pipeline-settings-l3bcde_20200101_v000.cdf'),
            Path('imap_glows_energy-grid-lo_20200101_v000.cdf'),
            Path('imap_glows_tess-xyz-8_20200101_v000.cdf'),
            Path('imap_glows_elongation-data_20200101_v000.cdf'),
            Path('imap_glows_energy-grid-hi_20200101_v000.cdf'),
            Path('imap_glows_energy-grid-ultra_20200101_v000.cdf'),
            Path('imap_glows_tess-ang-16_20200101_v000.cdf'),
        ]

        updated_l3d = Path('path/to/imap_glows_l3d_solar-hist_19470303-cr02091_v000.cdf')
        previous_l3d = None

        mock_fetch_dependencies.return_value.repointing_file = Path('path/to/repointing_file')
        mock_fetch_dependencies.return_value.pipeline_settings = {"start_cr": 2089}

        actual_initializer_output = GlowsL3EInitializer.get_repointings_to_process(updated_l3d, previous_l3d)

        mock_find_first_updated_cr.assert_not_called()
        mock_determine_l3e_files_to_produce.assert_has_calls([
            call(2089, 2091, actual_initializer_output.dependencies.repointing_file, "survival-probability-hi-45"),
            call(2089, 2091, actual_initializer_output.dependencies.repointing_file, "survival-probability-hi-90"),
            call(2089, 2091, actual_initializer_output.dependencies.repointing_file, "survival-probability-lo"),
            call(2089, 2091, actual_initializer_output.dependencies.repointing_file, "survival-probability-ultra"),
        ])
