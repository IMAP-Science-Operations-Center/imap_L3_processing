import unittest
from datetime import datetime
from unittest.mock import patch, Mock

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_processing.glows.glows_processor import GlowsProcessor
from imap_processing.glows.l3a.models import GlowsL3LightCurve
from imap_processing.models import UpstreamDataDependency, InputMetadata


class TestGlowsProcessor(unittest.TestCase):

    @patch('imap_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_processing.glows.glows_processor.read_l2_glows_data')
    @patch('imap_processing.glows.glows_processor.GlowsProcessor.process_l3a')
    @patch('imap_processing.glows.glows_processor.save_data')
    @patch('imap_processing.glows.glows_processor.imap_data_access.upload')
    def test_processor_handles_l3a(self, mock_upload, mock_save_data, mock_process_l3a_method, mock_read_glows_data, mock_glows_dependencies_class):
        instrument = 'glows'
        incoming_data_level = 'l2'
        start_date = datetime(2024,10,7,10,00,00)
        end_date = datetime(2024,10,8,10,00,00)
        version = 'v001'
        descriptor = GLOWS_L2_DESCRIPTOR + '00001'

        outgoing_data_level = "l3a"
        outgoing_version = 'v02'

        mock_fetched_dependencies = mock_glows_dependencies_class.fetch_dependencies.return_value
        mock_glows_l2_data = mock_read_glows_data.return_value
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
        mock_read_glows_data.assert_called_with(mock_fetched_dependencies.data)
        mock_process_l3a_method.assert_called_with(mock_glows_l2_data, mock_fetched_dependencies)
        mock_save_data.assert_called_with(mock_light_curve)
        mock_upload.assert_called_with(mock_cdf_path)

    @patch('imap_processing.glows.glows_processor.rebin_lightcurve')
    def test_process_l3a(self, mock_rebin_lightcurve):
        rebinned_flux = Mock()
        rebinned_exposure = Mock()
        mock_rebin_lightcurve.return_value = (rebinned_flux, rebinned_exposure)
        instrument = 'glows'
        incoming_data_level = 'l2'
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        version = 'v001'
        descriptor = GLOWS_L2_DESCRIPTOR + '00001'

        outgoing_data_level = "l3a"
        outgoing_version = 'v02'

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)

        dependencies = [
            UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                   version, descriptor),
        ]
        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)

        fetched_dependencies = Mock()
        data = Mock()
        result = processor.process_l3a(data, fetched_dependencies)
        self.assertIsInstance(result, GlowsL3LightCurve)

        mock_rebin_lightcurve.assert_called_with(data.photon_flux, data.histogram_flag_array,
                                                 data.exposure_times, fetched_dependencies.number_of_bins)
        self.assertEqual(rebinned_flux, result.photon_flux)
        self.assertEqual(rebinned_exposure, result.exposure_times)
        self.assertEqual(dependencies[0], result.input_metadata)





if __name__ == '__main__':
    unittest.main()