import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock, sentinel, call

import numpy as np

from imap_l3_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.models import UpstreamDataDependency, InputMetadata
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path


class TestGlowsProcessor(unittest.TestCase):

    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.process_l3a')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.imap_data_access.upload')
    def test_processor_handles_l3a(self, mock_upload, mock_save_data, mock_process_l3a_method,
                                   mock_glows_dependencies_class, mock_glows_initializer):
        instrument = 'glows'
        incoming_data_level = 'l2'
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        version = 'v001'
        descriptor = GLOWS_L2_DESCRIPTOR + '00001'

        outgoing_data_level = "l3a"
        outgoing_version = 'v02'

        mock_fetched_dependencies = mock_glows_dependencies_class.fetch_dependencies.return_value
        mock_light_curve = mock_process_l3a_method.return_value
        mock_cdf_path = mock_save_data.return_value

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)
        dependency_start_date = datetime(2025, 2, 24)
        dependency_end_date = None
        dependencies = [
            UpstreamDataDependency(instrument, incoming_data_level, dependency_start_date, dependency_end_date,
                                   version, descriptor),
        ]
        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)

        processor.process()
        mock_glows_initializer.assert_not_called()
        mock_glows_dependencies_class.fetch_dependencies.assert_called_with(dependencies)
        mock_process_l3a_method.assert_called_with(mock_fetched_dependencies)
        mock_save_data.assert_called_with(mock_light_curve)
        mock_upload.assert_called_with(mock_cdf_path)

    @patch('imap_l3_processing.glows.glows_processor.create_glows_l3a_from_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    def test_process_l3a(self, l3a_data_constructor, create_glows_l3a_from_dictionary):
        descriptor = GLOWS_L2_DESCRIPTOR + '00001'

        input_metadata = InputMetadata('glows', "l3a", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        dependencies = [
            UpstreamDataDependency('glows', 'l2', datetime(2024, 10, 7, 10, 00, 00), datetime(2024, 10, 8, 10, 00, 00),
                                   'v001', descriptor),
        ]
        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)

        processor.add_spin_angle_delta = Mock()
        fetched_dependencies = Mock()
        result = processor.process_l3a(fetched_dependencies)

        self.assertIs(create_glows_l3a_from_dictionary.return_value, result)
        l3a_data_constructor.assert_called_once_with(fetched_dependencies.ancillary_files)
        l3a_data_constructor.return_value.process_l2_data_file.assert_called_once_with(fetched_dependencies.data)
        l3a_data_constructor.return_value.generate_l3a_data.assert_called_once_with(
            fetched_dependencies.ancillary_files)
        processor.add_spin_angle_delta.assert_called_with(l3a_data_constructor.return_value.data,
                                                          fetched_dependencies.ancillary_files)
        create_glows_l3a_from_dictionary.assert_called_once_with(processor.add_spin_angle_delta.return_value,
                                                                 input_metadata.to_upstream_data_dependency(
                                                                     dependencies[0].descriptor))

    @patch("imap_l3_processing.glows.glows_processor.imap_data_access")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    def test_does_not_process_l3b_if_no_zip_files(self, mock_glows_initializer_class,
                                                  mock_save_data,
                                                  mock_imap_data_access):

        mock_glows_initializer_class.validate_and_initialize.return_value = []

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')
        dependencies = [
            UpstreamDataDependency('glows', 'l3a', datetime(2024, 10, 7, 10, 00, 00), datetime(2024, 10, 8, 10, 00, 00),
                                   'v001', GLOWS_L2_DESCRIPTOR + '00001'),
        ]

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        processor.process()

        mock_glows_initializer_class.validate_and_initialize.assert_called_with(input_metadata.version)
        mock_save_data.assert_not_called()
        mock_imap_data_access.upload.assert_not_called()

    def test_add_spin_angle_delta(self):
        cases = [
            (60, 3),
            (90, 2)
        ]

        for bins, expected_delta in cases:
            with self.subTest(bin=bins, expected_delta=expected_delta):
                ancillary_files = {}
                with open(get_test_data_path("glows/imap_glows_l3a_20130908085214_orbX_modX_p_v00.json")) as f:
                    example_data = json.load(f)

                with tempfile.TemporaryDirectory() as tempdir:
                    temp_file_path = Path(tempdir) / "settings.json"
                    example_settings = get_test_instrument_team_data_path(
                        "glows/imap_glows_l3a_pipeline-settings-json-not-cdf_20250707_v002.cdf")

                    with open(example_settings) as file:
                        loaded_file = json.load(file)

                    loaded_file['l3a_nominal_number_of_bins'] = bins

                    with open(temp_file_path, 'w') as file:
                        json.dump(loaded_file, file)
                    ancillary_files['settings'] = temp_file_path

                    result = GlowsProcessor.add_spin_angle_delta(deepcopy(example_data), ancillary_files)
                for k, v in example_data.items():
                    if k != "daily_lightcurve":
                        self.assertEqual(v, result[k])

                for k2, v2 in example_data["daily_lightcurve"].items():
                    self.assertEqual(v2, result["daily_lightcurve"][k2])
                spin_angle_delta = result["daily_lightcurve"]["spin_angle_delta"]
                self.assertEqual(len(example_data["daily_lightcurve"]["spin_angle"]), len(spin_angle_delta))
                self.assertTrue(np.all(spin_angle_delta == expected_delta))

    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCDependencies")
    @patch("imap_l3_processing.glows.glows_processor.imap_data_access")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind')
    @patch('imap_l3_processing.glows.glows_processor.filter_out_bad_days')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    def test_process_l3bc(self, mock_generate_l3bc, mock_filter_bad_days, mock_l3c_model_class, mock_l3b_model_class,
                          mock_glows_initializer_class, mock_save_data, mock_imap_data_access, mock_l3bc_dependencies):
        mock_glows_initializer_class.validate_and_initialize.return_value = [
            sentinel.zip_file_path_1,
            sentinel.zip_file_path_2,
        ]

        first_dependency = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data_1,
                                                 external_files=sentinel.external_files_1,
                                                 ancillary_files={
                                                     'bad_days_list': sentinel.bad_days_list_1,
                                                 },
                                                 carrington_rotation_number=sentinel.cr_1,
                                                 start_date=datetime(2024, 1, 1),
                                                 end_date=datetime(2024, 1, 30))
        second_dependency = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data_2,
                                                  external_files=sentinel.external_files_2,
                                                  ancillary_files={
                                                      'bad_days_list': sentinel.bad_days_list_2,
                                                  },
                                                  carrington_rotation_number=sentinel.cr_2,
                                                  start_date=datetime(2024, 2, 1),
                                                  end_date=datetime(2024, 2, 28))

        mock_l3bc_dependencies.fetch_dependencies.side_effect = [first_dependency, second_dependency]

        mock_generate_l3bc.side_effect = [(sentinel.l3b_data_1, sentinel.l3c_data_1),
                                          (sentinel.l3b_data_2, sentinel.l3c_data_2)]
        mock_filter_bad_days.side_effect = [sentinel.filtered_days_1, sentinel.filtered_days_2]

        mock_l3b_model_class.from_instrument_team_dictionary.side_effect = [sentinel.l3b_1,
                                                                            sentinel.l3b_2]
        mock_l3c_model_class.from_instrument_team_dictionary.side_effect = [sentinel.l3c_1,
                                                                            sentinel.l3c_2]
        mock_save_data.side_effect = [sentinel.l3b_cdf_path_1,
                                      sentinel.l3c_cdf_path_1,
                                      sentinel.l3b_cdf_path_2,
                                      sentinel.l3c_cdf_path_2]

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)
        processor.process()

        mock_l3bc_dependencies.fetch_dependencies.assert_has_calls(
            [call(sentinel.zip_file_path_1), call(sentinel.zip_file_path_2)])

        dependencies_with_filtered_list_1 = replace(first_dependency, l3a_data=sentinel.filtered_days_1)
        dependencies_with_filtered_list_2 = replace(second_dependency, l3a_data=sentinel.filtered_days_2)

        mock_filter_bad_days.assert_has_calls(
            [call(sentinel.l3a_data_1, sentinel.bad_days_list_1),
             call(sentinel.l3a_data_2, sentinel.bad_days_list_2)])

        mock_generate_l3bc.assert_has_calls(
            [call(dependencies_with_filtered_list_1), call(dependencies_with_filtered_list_2)])

        expected_l3b_metadata_1 = UpstreamDataDependency("glows", "l3b", first_dependency.start_date,
                                                         first_dependency.end_date, 'v02', "ion-rate-profile")
        expected_l3b_metadata_2 = UpstreamDataDependency("glows", "l3b", second_dependency.start_date,
                                                         second_dependency.end_date, 'v02', "ion-rate-profile")
        expected_l3c_metadata_1 = UpstreamDataDependency("glows", "l3c", first_dependency.start_date,
                                                         first_dependency.end_date, 'v02', "sw-profile")
        expected_l3c_metadata_2 = UpstreamDataDependency("glows", "l3c", second_dependency.start_date,
                                                         second_dependency.end_date, 'v02', "sw-profile")
        mock_l3b_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3b_data_1, expected_l3b_metadata_1),
             call(sentinel.l3b_data_2, expected_l3b_metadata_2)])

        mock_l3c_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3c_data_1, expected_l3c_metadata_1),
             call(sentinel.l3c_data_2, expected_l3c_metadata_2)])

        mock_save_data.assert_has_calls(
            [call(sentinel.l3b_1), call(sentinel.l3c_1), call(sentinel.l3b_2), call(sentinel.l3c_2)])

        mock_imap_data_access.upload.assert_has_calls([
            call(sentinel.zip_file_path_1),
            call(sentinel.l3b_cdf_path_1),
            call(sentinel.l3c_cdf_path_1),
            call(sentinel.zip_file_path_2),
            call(sentinel.l3b_cdf_path_2),
            call(sentinel.l3c_cdf_path_2),
        ])


if __name__ == '__main__':
    unittest.main()
