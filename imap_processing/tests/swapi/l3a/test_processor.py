import os
import shutil
from datetime import datetime, timedelta, date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, sentinel, call

import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

import imap_processing
from imap_processing.constants import TEMP_CDF_FOLDER_PATH, THIRTY_SECONDS_IN_NANOSECONDS
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.swapi.l3a.processor import SwapiL3AProcessor, TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR


class TestProcessor(TestCase):
    def setUp(self) -> None:
        self.temp_directory = f"{TEMP_CDF_FOLDER_PATH}"
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

        self.mock_imap_patcher = patch('imap_processing.processor.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()
        self.mock_imap_api.query.side_effect = [
            [{'file_path': sentinel.data_file_path}],
            [{'file_path': sentinel.lookup_table_file_path}]
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)
        self.mock_imap_patcher.stop()

    @patch('imap_processing.processor.ImapAttributeManager')
    @patch('imap_processing.swapi.l3a.processor.SwapiL3AlphaSolarWindData')
    @patch('imap_processing.swapi.l3a.processor.SwapiL3ProtonSolarWindData')
    @patch('imap_processing.processor.write_cdf')
    @patch('imap_processing.processor.uuid')
    @patch('imap_processing.swapi.l3a.processor.chunk_l2_data')
    @patch('imap_processing.swapi.l3a.processor.read_l2_swapi_data')
    @patch('imap_processing.swapi.l3a.processor.calculate_proton_solar_wind_speed')
    @patch('imap_processing.swapi.l3a.processor.calculate_alpha_solar_wind_speed')
    @patch('imap_processing.swapi.l3a.processor.TemperatureAndDensityCalibrationTable')
    @patch('imap_processing.swapi.l3a.processor.calculate_proton_solar_wind_temperature_and_density')
    def test_processor(self, mock_calculate_temperature_and_density, mock_temperature_and_density_calibrator_class,
                       mock_calculate_alpha_solar_wind_speed,
                       mock_calculate_proton_solar_wind_speed,
                       mock_read_l2_swapi_data, mock_chunk_l2_data, mock_uuid, mock_write_cdf,
                       mock_proton_solar_wind_data_constructor, mock_alpha_solar_wind_data_constructor,
                       mock_imap_attribute_manager):
        data_file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'
        mock_uuid_value = 123
        mock_uuid.uuid4.return_value = mock_uuid_value

        self.mock_imap_api.download.side_effect = [
            data_file_path,
            sentinel.local_lookup_table_path,
        ]
        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = 'c'
        end_date = datetime.now()
        version = 'f'
        outgoing_data_level = "l3a"
        start_date = datetime.now() - timedelta(days=1)
        outgoing_version = "12345"

        returned_proton_sw_speed = ufloat(400000, 2)
        mock_calculate_proton_solar_wind_speed.return_value = (
            returned_proton_sw_speed, sentinel.a, sentinel.phi, sentinel.b)

        mock_calculate_alpha_solar_wind_speed.return_value = ufloat(450000, 1000)

        returned_proton_sw_temp = ufloat(99000, 1000)
        returned_proton_sw_density = ufloat(4.97, 0.25)
        mock_calculate_temperature_and_density.return_value = (
            returned_proton_sw_temp, returned_proton_sw_density)

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        spin_angles = np.array([[24, 25, 26, 27, 28], [29, 30, 31, 32, 33], [34, 35, 36, 37, 38], [39, 40, 41, 42, 43]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_chunk_l2_data.return_value = [
            SwapiL2Data(epoch, energy, coincidence_count_rate, spin_angles, coincidence_count_rate_uncertainty)]

        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value
        alpha_solar_wind_data = mock_alpha_solar_wind_data_constructor.return_value
        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiL3AProcessor(
            [
                UpstreamDataDependency(instrument, incoming_data_level, descriptor, start_date, end_date,
                                       version),
                UpstreamDataDependency(instrument, incoming_data_level, "density-temperature-lut-text-not-cdf",
                                       None, None,
                                       version),
            ], instrument, outgoing_data_level, start_date, end_date,
            outgoing_version)
        swapi_processor.process()

        start_date_as_str = start_date.strftime("%Y%m%d")
        end_date_as_str = end_date.strftime("%Y%m%d")
        self.mock_imap_api.query.assert_has_calls([call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=descriptor, start_date=start_date_as_str,
                                                        end_date=end_date_as_str, version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor="density-temperature-lut-text-not-cdf",
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest')])
        self.mock_imap_api.download.assert_has_calls(
            [call(sentinel.data_file_path), call(sentinel.lookup_table_file_path)])
        mock_temperature_and_density_calibrator_class.from_file.assert_called_with(sentinel.local_lookup_table_path)
        mock_chunk_l2_data.assert_called_with(mock_read_l2_swapi_data.return_value, 5)

        expected_count_rate_with_uncertainties = uarray(coincidence_count_rate, coincidence_count_rate_uncertainty)
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(spin_angles, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[1])
        np.testing.assert_array_equal(energy, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[2])
        np.testing.assert_array_equal(epoch, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[3])

        self.assertEqual(mock_temperature_and_density_calibrator_class.from_file.return_value,
                         mock_calculate_temperature_and_density.call_args_list[0].args[0])
        self.assert_ufloat_equal(returned_proton_sw_speed,
                                 mock_calculate_temperature_and_density.call_args_list[0].args[1])
        self.assert_ufloat_equal(ufloat(0.01, 1.0), mock_calculate_temperature_and_density.call_args_list[0].args[2])
        self.assertEqual(sentinel.phi, mock_calculate_temperature_and_density.call_args_list[0].args[3])
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(mock_calculate_temperature_and_density.call_args_list[0].args[4]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_temperature_and_density.call_args_list[0].args[4]))
        np.testing.assert_array_equal(energy, mock_calculate_temperature_and_density.call_args_list[0].args[5])

        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(energy, mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[1])

        actual_proton_epoch, actual_proton_sw_speed, actual_proton_sw_temperature, actual_proton_sw_density = mock_proton_solar_wind_data_constructor.call_args.args
        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]), actual_proton_epoch,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_speed]), actual_proton_sw_speed, strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_temp]), actual_proton_sw_temperature, strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_density]), actual_proton_sw_density, strict=True)

        proton_cdf_path = f"{self.temp_directory}/imap_swapi_l3a_proton-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345.cdf"
        alpha_cdf_path = f"{self.temp_directory}/imap_swapi_l3a_alpha-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345.cdf"
        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date", date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 "imap_swapi_l3a_proton-sw"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_proton-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345"),
                                                            call("Data_version", outgoing_version),
                                                            call("Generation_date", date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 "imap_swapi_l3a_alpha-sw"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_alpha-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345"),
                                                            ])

        actual_alpha_epoch, actual_alpha_sw_speed = mock_alpha_solar_wind_data_constructor.call_args.args

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]), actual_alpha_epoch)
        np.testing.assert_array_equal(np.array([mock_calculate_alpha_solar_wind_speed.return_value]),
                                      actual_alpha_sw_speed)
        mock_manager.add_instrument_attrs.assert_has_calls([call("swapi", "l3a"), call("swapi", "l3a")])
        mock_write_cdf.assert_has_calls([
            call(proton_cdf_path, proton_solar_wind_data, mock_manager),
            call(alpha_cdf_path, alpha_solar_wind_data, mock_manager)
        ])
        self.mock_imap_api.upload.assert_has_calls([call(proton_cdf_path), call(alpha_cdf_path)])

    def assert_ufloat_equal(self, expected_ufloat, actual_ufloat):
        self.assertEqual(expected_ufloat.n, actual_ufloat.n)
        self.assertEqual(expected_ufloat.s, actual_ufloat.s)

    def test_processor_throws_exception_when_more_than_one_file_is_downloaded(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.download.return_value = file_path

        self.mock_imap_api.query.side_effect = [
            [{'file_path': '1 thing'}],
            [{'file_path': '2 thing'}, {'file_path': '3 thing'}]
        ]

        dependencies = [UpstreamDataDependency('swapi', 'l2', 'c', datetime.now() - timedelta(days=1), datetime.now(),
                                               'f'),
                        UpstreamDataDependency('swapi', 'l2', TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR,
                                               datetime.now() - timedelta(days=1), datetime.now(),
                                               'f')
                        ]
        swapi_processor = SwapiL3AProcessor(
            dependencies, 'swapi', "l3a", datetime.now() - timedelta(days=1),
            datetime.now(),
            "12345")

        with self.assertRaises(ValueError) as cm:
            swapi_processor.process()
        exception = cm.exception
        self.assertEqual(f"Unexpected files found for SWAPI L3:"
                         f"{['2 thing', '3 thing']}. Expected only one file to download.",
                         str(exception))

    def test_processor_throws_exception_when_wrong_number_of_dependencies_are_passed_in(self):
        test_cases = [
            ("too many dependencies",
             [
                 UpstreamDataDependency('swapi', 'l2', 'data', datetime.now() - timedelta(days=1), datetime.now(),
                                        'f'),
                 UpstreamDataDependency('swapi', 'l2', 'sample-lut', datetime.now() - timedelta(days=1), datetime.now(),
                                        'f'),
                 UpstreamDataDependency('swapi', 'l2', 'something else', datetime.now() - timedelta(days=1),
                                        datetime.now(),
                                        'f')
             ]),
            ("too few dependencies",
             [
                 UpstreamDataDependency('swapi', 'l2', 'data', datetime.now() - timedelta(days=1), datetime.now(),
                                        'f')
             ])
        ]
        for name, dependencies in test_cases:
            with self.subTest(name):
                swapi_processor = SwapiL3AProcessor(dependencies, 'swapi', "l3a", datetime.now() - timedelta(days=1),
                                                    datetime.now(),
                                                    "12345")
                with self.assertRaises(ValueError) as cm:
                    swapi_processor.process()
                exception = cm.exception
                self.assertEqual(f"Incorrect dependencies provided for SWAPI L3:"
                                 f"{dependencies}. Expected exactly two dependencies.", str(exception))

    def test_processor_throws_exception_when_missing_temp_density_lookup_table(self):
        dependencies = [
            UpstreamDataDependency('swapi', 'l2', 'data', datetime.now() - timedelta(days=1), datetime.now(),
                                   'f'),
            UpstreamDataDependency('swapi', 'l2', 'not-the-lut', datetime.now() - timedelta(days=1), datetime.now(),
                                   'f')]
        swapi_processor = SwapiL3AProcessor(dependencies, 'swapi', "l3a", datetime.now() - timedelta(days=1),
                                            datetime.now(),
                                            "12345")
        with self.assertRaises(ValueError) as cm:
            swapi_processor.process()
        exception = cm.exception
        self.assertEqual(f"Missing {TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR} dependency.",
                         str(exception))
