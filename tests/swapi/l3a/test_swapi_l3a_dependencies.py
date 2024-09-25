import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, sentinel, call

import imap_processing
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.processor import SWAPI_L2_DESCRIPTOR
from imap_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies


class TestSwapiL3ADependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_imap_patcher = patch('imap_processing.utils.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()
        self.mock_imap_api.query.side_effect = [
            [{'file_path': sentinel.data_file_path}],
            [{'file_path': sentinel.proton_lookup_table_file_path}],
            [{'file_path': sentinel.alpha_lookup_table_file_path}],
            [{'file_path': sentinel.clock_and_deflection_table_file_path}]
        ]

    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.CDF')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.ClockAngleCalibrationTable')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.AlphaTemperatureDensityCalibrationTable')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.ProtonTemperatureAndDensityCalibrationTable')
    def test_fetch_dependencies(self, mock_proton_temperature_and_density_calibrator_class,
                                mock_alpha_temperature_and_density_calibrator_class,
                                mock_clock_angle_calibration_table_constructor,
                                mock_cdf_constructor):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = SWAPI_L2_DESCRIPTOR
        end_date = datetime.now()
        version = 'f'
        start_date = datetime.now() - timedelta(days=1)

        dependencies = UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                              version, descriptor)

        data_file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.download.side_effect = [
            data_file_path,
            sentinel.proton_density_temp_local_lookup_table_path,
            sentinel.alpha_density_temp_local_lookup_table_path,
            sentinel.clock_deflection_angle_local_lookup_table_path
        ]

        fetched_dependencies = SwapiL3ADependencies.fetch_dependencies([dependencies])

        start_date_as_str = start_date.strftime("%Y%m%d")
        end_date_as_str = end_date.strftime("%Y%m%d")
        self.mock_imap_api.query.assert_has_calls([call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=descriptor, start_date=start_date_as_str,
                                                        end_date=end_date_as_str, version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor="density-temperature-lut-text-not-cdf",
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor="alpha-density-temperature-lut-text-not-cdf",
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor="clock-angle-and-flow-deflection-lut-text-not-cdf",
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   ])
        self.mock_imap_api.download.assert_has_calls(
            [call(sentinel.data_file_path), call(sentinel.proton_lookup_table_file_path),
             call(sentinel.alpha_lookup_table_file_path),
             call(sentinel.clock_and_deflection_table_file_path)])

        mock_cdf_constructor.assert_called_with(str(data_file_path))
        mock_proton_temperature_and_density_calibrator_class.from_file.assert_called_with(
            sentinel.proton_density_temp_local_lookup_table_path)
        mock_alpha_temperature_and_density_calibrator_class.from_file.assert_called_with(
            sentinel.alpha_density_temp_local_lookup_table_path)
        mock_clock_angle_calibration_table_constructor.from_file.assert_called_with(
            sentinel.clock_deflection_angle_local_lookup_table_path)

        self.assertIs(mock_cdf_constructor.return_value,
                      fetched_dependencies.data)
        self.assertIs(mock_proton_temperature_and_density_calibrator_class.from_file.return_value,
                      fetched_dependencies.proton_temperature_density_calibration_table)
        self.assertIs(mock_alpha_temperature_and_density_calibrator_class.from_file.return_value,
                      fetched_dependencies.alpha_temperature_density_calibration_table)
        self.assertIs(mock_clock_angle_calibration_table_constructor.from_file.return_value,
                      fetched_dependencies.clock_angle_and_flow_deflection_calibration_table)