import unittest
from datetime import datetime
from unittest.mock import patch, Mock

import numpy as np

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_processing.glows.glows_processor import GlowsProcessor
from imap_processing.glows.l3a.models import GlowsL3LightCurve
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

    @patch('imap_processing.glows.glows_processor.rebin_lightcurve')
    @patch('imap_processing.glows.glows_processor.calculate_spin_angles')
    @patch('imap_processing.glows.glows_processor.unumpy.uarray')
    def test_process_l3a(self, mock_uarray, mock_calculate_spin_angles, mock_rebin_lightcurve):
        rebinned_flux = np.array([1, 2, 3, 4])
        rebinned_exposure = np.array([5, 6, 7, 8])
        mock_rebin_lightcurve.return_value = (rebinned_flux, rebinned_exposure)
        mock_calculate_spin_angles.return_value = np.array([90, 90, 90, 90])
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

        data = Mock()
        data.start_time = datetime(2024, 10, 7, 2)
        data.end_time = datetime(2024, 10, 7, 22)
        data.epoch = datetime(2024, 10, 7, 2)

        fetched_dependencies = Mock()
        fetched_dependencies.data = data

        expected_epoch_delta = 10 * 3600 * 1_000_000_000
        expected_epoch = np.array([datetime(2024, 10, 7, 12)])
        result = processor.process_l3a(fetched_dependencies)
        self.assertIsInstance(result, GlowsL3LightCurve)
        mock_uarray.assert_called_with(data.photon_flux, data.flux_uncertainties)
        mock_rebin_lightcurve.assert_called_with(fetched_dependencies.time_independent_background_table,
                                                 fetched_dependencies.bad_angle_flag_configuration,
                                                 mock_uarray.return_value,
                                                 data.ecliptic_lat,
                                                 data.ecliptic_lon,
                                                 data.histogram_flag_array,
                                                 data.exposure_times, fetched_dependencies.number_of_bins,
                                                 fetched_dependencies.time_dependent_background)
        mock_calculate_spin_angles.assert_called_with(fetched_dependencies.number_of_bins, data.spin_angle)
        np.testing.assert_equal(result.photon_flux, np.array([[1, 2, 3, 4]]), strict=True)
        np.testing.assert_equal(result.exposure_times, np.array([[5, 6, 7, 8]]), strict=True)
        self.assertEqual(input_metadata.to_upstream_data_dependency(descriptor), result.input_metadata)
        np.testing.assert_equal(result.epoch, expected_epoch, strict=True)
        self.assertEqual(result.epoch_delta, [expected_epoch_delta])
        np.testing.assert_equal(result.spin_angle, np.array([[90, 90, 90, 90]]), strict=True)


if __name__ == '__main__':
    unittest.main()
