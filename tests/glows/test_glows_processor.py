import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np

from imap_l3_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_l3_processing.glows.glows_processor import GlowsProcessor
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
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BDependencies")
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    def test_does_not_process_l3b_if_should_not_process(self, mock_glows_initializer_class,
                                                        mock_glows_l3b_dependencies_class,
                                                        mock_save_data,
                                                        mock_imap_data_access):
        mock_glows_initializer = mock_glows_initializer_class.return_value

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')
        dependencies = [
            UpstreamDataDependency('glows', 'l3a', datetime(2024, 10, 7, 10, 00, 00), datetime(2024, 10, 8, 10, 00, 00),
                                   'v001', GLOWS_L2_DESCRIPTOR + '00001'),
        ]

        mock_glows_initializer.should_process.return_value = False

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        processor.process()

        mock_glows_l3b_dependencies_class.fetch_dependencies.assert_called_with(dependencies)
        mock_glows_initializer.should_process.assert_called_with(
            mock_glows_l3b_dependencies_class.fetch_dependencies.return_value)
        mock_save_data.assert_not_called()
        mock_imap_data_access.upload.assert_not_called()

    @patch("imap_l3_processing.glows.glows_processor.imap_data_access")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BDependencies")
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    def test_processes_l3b_if_should_process(self, mock_glows_initializer_class,
                                             mock_glows_l3b_dependencies_class,
                                             mock_save_data,
                                             mock_imap_data_access):
        mock_glows_initializer = mock_glows_initializer_class.return_value

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')
        dependencies = [
            UpstreamDataDependency('glows', 'l3a', datetime(2024, 10, 7, 10, 00, 00), datetime(2024, 10, 8, 10, 00, 00),
                                   'v001', GLOWS_L2_DESCRIPTOR + '00001'),
        ]

        mock_glows_initializer.should_process.return_value = True

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        processor.process()

        mock_glows_l3b_dependencies_class.fetch_dependencies.assert_called_with(dependencies)
        mock_glows_initializer.should_process.assert_called_with(
            mock_glows_l3b_dependencies_class.fetch_dependencies.return_value)
        mock_save_data.assert_called_once()
        mock_imap_data_access.upload.assert_called_once()

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


if __name__ == '__main__':
    unittest.main()
