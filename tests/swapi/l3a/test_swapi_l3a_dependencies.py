import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, sentinel, call

import imap_processing
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR, \
    INSTRUMENT_RESPONSE_LOOKUP_TABLE, DENSITY_OF_NEUTRAL_HELIUM
from imap_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies


class TestSwapiL3ADependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_imap_patcher = patch('imap_processing.utils.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()
        self.mock_imap_api.query.side_effect = [
            [{'file_path': sentinel.data_file_path}],
            [{'file_path': sentinel.proton_lookup_table_file_path}],
            [{'file_path': sentinel.alpha_lookup_table_file_path}],
            [{'file_path': sentinel.clock_and_deflection_table_file_path}],
            [{'file_path': sentinel.geometric_factor_calibration_table_file_path}],
            [{'file_path': sentinel.instrument_response_calibration_table_file_path}],
            [{'file_path': sentinel.density_of_neutral_helium_calibration_table_file_path}]
        ]

    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.CDF')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.ClockAngleCalibrationTable')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.AlphaTemperatureDensityCalibrationTable')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.ProtonTemperatureAndDensityCalibrationTable')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.GeometricFactorCalibrationTable')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.InstrumentResponseLookupTableCollection')
    @patch('imap_processing.swapi.l3a.swapi_l3a_dependencies.DensityOfNeutralHeliumLookupTable')
    def test_fetch_dependencies(self, mock_density_of_neutral_hydrogen_lookup_table_class
                                , mock_instrument_response_lookup_table_class
                                , mock_geometric_factor_calibration_class
                                , mock_proton_temperature_and_density_calibrator_class,
                                mock_alpha_temperature_and_density_calibrator_class,
                                mock_clock_angle_calibration_table_constructor,
                                mock_cdf_constructor):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = SWAPI_L2_DESCRIPTOR
        end_date = datetime.now()
        version = 'v002'
        start_date = datetime.now() - timedelta(days=1)

        dependencies = UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                              version, descriptor)

        data_file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.download.side_effect = [
            data_file_path,
            sentinel.proton_density_temp_local_lookup_table_path,
            sentinel.alpha_density_temp_local_lookup_table_path,
            sentinel.clock_deflection_angle_local_lookup_table_path,
            sentinel.geometric_factor_calibration_table_file_path,
            sentinel.instrument_response_calibration_table_file_path,
            sentinel.density_of_neutral_helium_calibration_table_file_path
        ]

        fetched_dependencies = SwapiL3ADependencies.fetch_dependencies([dependencies])

        start_date_as_str = start_date.strftime("%Y%m%d")
        end_date_as_str = end_date.strftime("%Y%m%d")
        self.mock_imap_api.query.assert_has_calls([call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=descriptor, start_date=start_date_as_str,
                                                        end_date=end_date_as_str, version='v002'),
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
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR,
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=INSTRUMENT_RESPONSE_LOOKUP_TABLE,
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=DENSITY_OF_NEUTRAL_HELIUM,
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   ],
                                                  )
        self.mock_imap_api.download.assert_has_calls(
            [call(sentinel.data_file_path), call(sentinel.proton_lookup_table_file_path),
             call(sentinel.alpha_lookup_table_file_path),
             call(sentinel.clock_and_deflection_table_file_path),
             call(sentinel.geometric_factor_calibration_table_file_path),
             call(sentinel.instrument_response_calibration_table_file_path),
             call(sentinel.density_of_neutral_helium_calibration_table_file_path),
             ])

        mock_cdf_constructor.assert_called_with(str(data_file_path))
        mock_proton_temperature_and_density_calibrator_class.from_file.assert_called_with(
            sentinel.proton_density_temp_local_lookup_table_path)
        mock_alpha_temperature_and_density_calibrator_class.from_file.assert_called_with(
            sentinel.alpha_density_temp_local_lookup_table_path)
        mock_clock_angle_calibration_table_constructor.from_file.assert_called_with(
            sentinel.clock_deflection_angle_local_lookup_table_path)
        mock_geometric_factor_calibration_class.from_file.assert_called_with(
            sentinel.geometric_factor_calibration_table_file_path)
        mock_instrument_response_lookup_table_class.from_file.assert_called_with(
            sentinel.instrument_response_calibration_table_file_path)
        mock_density_of_neutral_hydrogen_lookup_table_class.from_file.assert_called_with(
            sentinel.density_of_neutral_helium_calibration_table_file_path)

        self.assertIs(mock_cdf_constructor.return_value,
                      fetched_dependencies.data)
        self.assertIs(mock_proton_temperature_and_density_calibrator_class.from_file.return_value,
                      fetched_dependencies.proton_temperature_density_calibration_table)
        self.assertIs(mock_alpha_temperature_and_density_calibrator_class.from_file.return_value,
                      fetched_dependencies.alpha_temperature_density_calibration_table)
        self.assertIs(mock_clock_angle_calibration_table_constructor.from_file.return_value,
                      fetched_dependencies.clock_angle_and_flow_deflection_calibration_table)
        self.assertIs(mock_geometric_factor_calibration_class.from_file.return_value,
                      fetched_dependencies.geometric_factor_calibration_table)
        self.assertIs(mock_instrument_response_lookup_table_class.from_file.return_value,
                      fetched_dependencies.instrument_response_calibration_table)
        self.assertIs(mock_density_of_neutral_hydrogen_lookup_table_class.from_file.return_value,
                      fetched_dependencies.density_of_neutral_helium_calibration_table)

    def test_throws_exception_when_more_than_one_file_is_downloaded(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.download.return_value = file_path

        self.mock_imap_api.query.side_effect = [
            [{'file_path': '1 thing'}],
            [{'file_path': '2 thing'}, {'file_path': '3 thing'}]
        ]

        dependencies = [UpstreamDataDependency('swapi', 'l2', datetime.now() - timedelta(days=1),
                                               datetime.now(), 'f', SWAPI_L2_DESCRIPTOR)]

        with self.assertRaises(ValueError) as cm:
            SwapiL3ADependencies.fetch_dependencies(dependencies)
        exception = cm.exception
        self.assertEqual(f"Unexpected files found for SWAPI L3:"
                         f"{['2 thing', '3 thing']}. Expected one file to download, found 2.",
                         str(exception))

    def test_throws_exception_when_missing_swapi_data(self):
        dependencies = [
            UpstreamDataDependency('swapi', 'l2', datetime.now() - timedelta(days=1), datetime.now(), 'f', 'data')]

        with self.assertRaises(ValueError) as cm:
            SwapiL3ADependencies.fetch_dependencies(dependencies)
        exception = cm.exception
        self.assertEqual(f"Missing {SWAPI_L2_DESCRIPTOR} dependency.",
                         str(exception))
