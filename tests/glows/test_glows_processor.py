import unittest
from datetime import datetime
from unittest.mock import patch, Mock

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_processing.glows.glows_processor import GlowsProcessor
from imap_processing.models import UpstreamDataDependency, InputMetadata


class TestGlowsProcessor(unittest.TestCase):

    @patch('imap_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_processing.glows.glows_processor.GlowsProcessor.process_l3a')
    @patch('imap_processing.glows.glows_processor.save_data')
    @patch('imap_processing.glows.glows_processor.imap_data_access.upload')
    def test_processor_handles_l3a(self, mock_upload, mock_save_data, mock_process_l3a_method,
                                   mock_glows_dependencies_class):
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

        dependencies = [
            UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                   version, descriptor),
        ]
        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)

        processor.process()

        mock_glows_dependencies_class.fetch_dependencies.assert_called_with(dependencies)
        mock_process_l3a_method.assert_called_with(mock_fetched_dependencies)
        mock_save_data.assert_called_with(mock_light_curve)
        mock_upload.assert_called_with(mock_cdf_path)

    @patch('imap_processing.glows.glows_processor.create_glows_l3a_from_dictionary')
    @patch('imap_processing.glows.glows_processor.L3aData')
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

        fetched_dependencies = Mock()
        result = processor.process_l3a(fetched_dependencies)

        self.assertIs(create_glows_l3a_from_dictionary.return_value, result)
        l3a_data_constructor.assert_called_once_with(fetched_dependencies.ancillary_files)
        l3a_data_constructor.return_value.process_l2_data_file.assert_called_once_with(fetched_dependencies.data)
        l3a_data_constructor.return_value.generate_l3a_data.assert_called_once_with(
            fetched_dependencies.ancillary_files)
        create_glows_l3a_from_dictionary.assert_called_once_with(l3a_data_constructor.return_value.data,
                                                                 input_metadata.to_upstream_data_dependency(
                                                                     dependencies[0].descriptor))


if __name__ == '__main__':
    unittest.main()
