import json
import unittest
from datetime import datetime

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3a.models import GlowsL2Data, GlowsL2Header
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_from_dictionary, \
    create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    def test_reading_l2_glows_data_into_models(self):
        cdf = CDF(str(get_test_data_path('glows/imap_glows_l2_hist_20130908-repoint01000_v001.cdf')))

        glows_l2_data: GlowsL2Data = read_l2_glows_data(cdf)

        expected_start_time = datetime(2013, 9, 8, 8, 52, 14)
        self.assertEqual(str(expected_start_time), glows_l2_data["start_time"])

        expected_end_time = datetime(2013, 9, 9, 4, 58, 14)
        self.assertEqual(str(expected_end_time), glows_l2_data["end_time"])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["photon_flux"].shape)
        self.assertEqual(681.6, glows_l2_data["daily_lightcurve"]["photon_flux"][0])
        self.assertEqual(678.4, glows_l2_data["daily_lightcurve"]["photon_flux"][-1])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["flux_uncertainties"].shape)
        self.assertEqual(5.828, glows_l2_data["daily_lightcurve"]["flux_uncertainties"][0])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["exposure_times"].shape)
        self.assertEqual(2.007e+01, glows_l2_data["daily_lightcurve"]["exposure_times"][0])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["spin_angle"].shape)
        self.assertEqual(0.072, glows_l2_data["daily_lightcurve"]["spin_angle"][0])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["histogram_flag_array"].shape)
        self.assertEqual(np.float64, glows_l2_data["daily_lightcurve"]["histogram_flag_array"].dtype)
        self.assertEqual(np.uint8(4), glows_l2_data["daily_lightcurve"]["histogram_flag_array"][84])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["ecliptic_lat"].shape)
        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["ecliptic_lon"].shape)
        self.assertEqual(75.000, glows_l2_data["daily_lightcurve"]["ecliptic_lat"][0])
        self.assertEqual(161.823, glows_l2_data["daily_lightcurve"]["ecliptic_lon"][0])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["raw_histogram"].shape)
        self.assertEqual(13677, glows_l2_data["daily_lightcurve"]["raw_histogram"][0])

        self.assertEqual({
            "lon": 162.092,
            "lat": 0.000
        }, glows_l2_data["spin_axis_orientation_average"])

        self.assertEqual(cdf['identifier'][0], glows_l2_data['identifier'])
        self.assertEqual(cdf['filter_temperature_average'][0], glows_l2_data['filter_temperature_average'])
        self.assertEqual(cdf['filter_temperature_std_dev'][0], glows_l2_data['filter_temperature_std_dev'])
        self.assertEqual(cdf['hv_voltage_average'][0], glows_l2_data['hv_voltage_average'])
        self.assertEqual(cdf['hv_voltage_std_dev'][0], glows_l2_data['hv_voltage_std_dev'])
        self.assertEqual(cdf['spin_period_average'][0], glows_l2_data['spin_period_average'])
        self.assertEqual(cdf['spin_period_std_dev'][0], glows_l2_data['spin_period_std_dev'])
        self.assertEqual(cdf['spin_period_ground_average'][0], glows_l2_data['spin_period_ground_average'])
        self.assertEqual(cdf['spin_period_ground_std_dev'][0], glows_l2_data['spin_period_ground_std_dev'])
        self.assertEqual(cdf['pulse_length_average'][0], glows_l2_data['pulse_length_average'])
        self.assertEqual(cdf['pulse_length_std_dev'][0], glows_l2_data['pulse_length_std_dev'])
        self.assertEqual(cdf['position_angle_offset_average'][0], glows_l2_data['position_angle_offset_average'])
        self.assertEqual(cdf['position_angle_offset_std_dev'][0], glows_l2_data['position_angle_offset_std_dev'])
        self.assertEqual("imap_glows_l2_hist_20130908-repoint01000_v001.cdf", glows_l2_data['l2_file_name'])

        self.assert_equal_xyz(cdf, "spacecraft_location_average", glows_l2_data)
        self.assert_equal_xyz(cdf, "spacecraft_location_std_dev", glows_l2_data)
        self.assert_equal_xyz(cdf, "spacecraft_velocity_average", glows_l2_data)
        self.assert_equal_xyz(cdf, "spacecraft_velocity_std_dev", glows_l2_data)

        expected_header = GlowsL2Header(
            flight_software_version=-1,
            pkts_file_name="",
            ancillary_data_files=[],
        )
        self.assertEqual(expected_header, glows_l2_data["header"])

    def assert_equal_xyz(self, cdf, variable_name, actual):
        expected_spacecraft_location_average = {
            'x': cdf[variable_name][0, 0],
            'y': cdf[variable_name][0, 1],
            'z': cdf[variable_name][0, 2],
        }
        self.assertEqual(expected_spacecraft_location_average, actual[variable_name])

    def test_fails_if_cdf_has_more_than_one_histogram(self):
        cdf = CDF(str(get_test_data_path('glows/glows_l2_with_too_many_histograms.cdf')))

        with self.assertRaises(AssertionError) as cm:
            glows_l2_data: GlowsL2Data = read_l2_glows_data(cdf)
        self.assertEqual(("Level 2 file should have only one histogram",), cm.exception.args)

    def test_create_glows_l3a_from_dictionary(self):
        with open(get_test_data_path("glows/imap_glows_l3a_20130908085214_orbX_modX_p_v00.json")) as f:
            data = json.load(f)
            expected_spin_delta = 3
            data["daily_lightcurve"]['spin_angle_delta'] = np.full_like(data['daily_lightcurve']['spin_angle'],
                                                                        expected_spin_delta)
            input_metadata = InputMetadata(instrument='glows', data_level='l3a',
                                           descriptor='hist', start_date=datetime(2013, 9, 8),
                                           end_date=datetime(2013, 9, 9), version="v003")

            result = create_glows_l3a_from_dictionary(data, input_metadata)

            self.assertEqual(1, len(result.epoch))
            self.assertEqual(datetime(2013, 9, 8, 18, 55, 14), result.epoch[0])

            self.assertEqual((65,), result.spin_angle.shape)
            self.assertEqual(2.000, result.spin_angle[0])

            self.assertEqual((65,), result.spin_angle_delta.shape)
            self.assertEqual(expected_spin_delta, result.spin_angle_delta[0])

            self.assertEqual((1, 65), result.photon_flux.shape)
            self.assertEqual(620.9, result.photon_flux[0, 0])

            self.assertEqual((1, 65), result.raw_histogram.shape)
            self.assertEqual(498484, result.raw_histogram[0, 0])

            self.assertEqual((1, 65), result.exposure_times.shape)
            self.assertEqual(8.028e+02, result.exposure_times[0, 0])

            self.assertEqual(1, len(result.epoch_delta))
            self.assertEqual(72360 / 2 * 1e9, result.epoch_delta[0])

            self.assertEqual((1, 65), result.photon_flux_uncertainty.shape)
            self.assertEqual(8.795e-01, result.photon_flux_uncertainty[0, 0])

            self.assertEqual((1, 65), result.longitude.shape)
            self.assertEqual(154.671, result.longitude[0, 0])

            self.assertEqual((1, 65), result.latitude.shape)
            self.assertEqual(74.870, result.latitude[0, 0])

            self.assertEqual((1, 65), result.extra_heliospheric_background.shape)
            self.assertEqual(1.23, result.extra_heliospheric_background[0, 10])

            self.assertEqual((1, 65), result.time_dependent_background.shape)
            self.assertEqual(1.66, result.time_dependent_background[0, 15])

            self.assertEqual(input_metadata, result.input_metadata)

            self.assertEqual((1,), result.filter_temperature_average.shape)
            self.assertEqual((1,), result.filter_temperature_std_dev.shape)
            self.assertEqual((1,), result.hv_voltage_average.shape)
            self.assertEqual((1,), result.hv_voltage_std_dev.shape)
            self.assertEqual((1,), result.spin_period_average.shape)
            self.assertEqual((1,), result.spin_period_std_dev.shape)
            self.assertEqual((1,), result.spin_period_ground_average.shape)
            self.assertEqual((1,), result.spin_period_ground_std_dev.shape)
            self.assertEqual((1,), result.pulse_length_average.shape)
            self.assertEqual((1,), result.pulse_length_std_dev.shape)
            self.assertEqual((1,), result.position_angle_offset_average.shape)
            self.assertEqual((1,), result.position_angle_offset_std_dev.shape)
            self.assertEqual((1, 2), result.spin_axis_orientation_average.shape)
            self.assertEqual((1, 2), result.spin_axis_orientation_std_dev.shape)
            self.assertEqual((1, 3), result.spacecraft_location_average.shape)
            self.assertEqual((1, 3), result.spacecraft_location_std_dev.shape)
            self.assertEqual((1, 3), result.spacecraft_velocity_average.shape)
            self.assertEqual((1, 3), result.spacecraft_velocity_std_dev.shape)
            self.assertEqual(result.epoch.shape, result.number_of_bins.shape)

            self.assertEqual(-27.84, result.filter_temperature_average[0])
            self.assertEqual(0.000e+00, result.filter_temperature_std_dev[0])
            self.assertEqual(1527.1, result.hv_voltage_average[0])
            self.assertEqual(8.795e+01, result.hv_voltage_std_dev[0])
            self.assertEqual(15.0000000, result.spin_period_average[0])
            self.assertEqual(0.000e+00, result.spin_period_std_dev[0])
            self.assertEqual(15.2366823, result.spin_period_ground_average[0])
            self.assertEqual(1.498e-03, result.spin_period_ground_std_dev[0])
            self.assertEqual(2.990e-01, result.pulse_length_average[0])
            self.assertEqual(1.726e-02, result.pulse_length_std_dev[0])
            self.assertEqual(91.578, result.position_angle_offset_average[0])
            self.assertEqual(9.991e-03, result.position_angle_offset_std_dev[0])
            np.testing.assert_array_equal(65, result.number_of_bins[0])

            np.testing.assert_array_equal([162.092, 0.000], result.spin_axis_orientation_average[0])
            np.testing.assert_array_equal([2.345e-01, 0.000e+00], result.spin_axis_orientation_std_dev[0])
            np.testing.assert_array_equal([146231106.1, -36212054.4, 1049.3], result.spacecraft_location_average[0])
            np.testing.assert_array_equal([1.421e+05, 6.001e+05, 4.082e+01], result.spacecraft_location_std_dev[0])
            np.testing.assert_array_equal([6.669, 28.805, -0.002], result.spacecraft_velocity_average[0])
            np.testing.assert_array_equal([1.188e-01, 2.961e-02, 4.337e-19], result.spacecraft_velocity_std_dev[0])

    def test_create_glows_l3a_dictionary_from_cdf(self):
        cdf_path = get_test_data_path("glows") / "imap_glows_l3a_hist_20100101_v001.cdf"
        actual_dictionary = create_glows_l3a_dictionary_from_cdf(cdf_path)

        self.assertEqual("imap_glows_l3a_hist_20100101_v001.cdf", actual_dictionary["filename"])
        self.assertEqual(datetime(2013, 9, 8, 8, 52, 14).strftime("%Y-%m-%d %H:%M:%S"),
                         actual_dictionary['start_time'])
        self.assertEqual(datetime(2013, 9, 9, 4, 58, 14).strftime("%Y-%m-%d %H:%M:%S"), actual_dictionary['end_time'])

        self.assertEqual(74.87028793594668, actual_dictionary['daily_lightcurve']['ecliptic_lat'][0])
        self.assertEqual(74.87028793594668, actual_dictionary['daily_lightcurve']['ecliptic_lat'][-1])
        self.assertEqual(154.67118387992855, actual_dictionary['daily_lightcurve']['ecliptic_lon'][0])
        self.assertEqual(169.51280659858702, actual_dictionary['daily_lightcurve']['ecliptic_lon'][-1])
        self.assertEqual(802.8, actual_dictionary['daily_lightcurve']['exposure_times'][0])
        self.assertEqual(80.28, actual_dictionary['daily_lightcurve']['exposure_times'][2])
        self.assertAlmostEqual(620.93173891, actual_dictionary['daily_lightcurve']['photon_flux'][0])
        self.assertAlmostEqual(635.46586946, actual_dictionary['daily_lightcurve']['photon_flux'][-1])
        self.assertAlmostEqual(0.87946437, actual_dictionary['daily_lightcurve']['flux_uncertainties'][0])
        self.assertAlmostEqual(0.88969763, actual_dictionary['daily_lightcurve']['flux_uncertainties'][-1])
        self.assertAlmostEqual(7.20014416e-04, actual_dictionary['daily_lightcurve']['extra_heliospheric_bckgrd'][0])
        self.assertAlmostEqual(3.66514655e-04, actual_dictionary['daily_lightcurve']['extra_heliospheric_bckgrd'][-1])
        self.assertEqual(0, actual_dictionary['daily_lightcurve']['time_dependent_bckgrd'][0])
        self.assertEqual(1.0000e-02, actual_dictionary['daily_lightcurve']['time_dependent_bckgrd'][10])
        self.assertEqual(2, actual_dictionary['daily_lightcurve']['spin_angle'][0])
        self.assertEqual(358, actual_dictionary['daily_lightcurve']['spin_angle'][-1])
        self.assertEqual(498484, actual_dictionary['daily_lightcurve']['raw_histogram'][0])
        self.assertEqual(510152, actual_dictionary['daily_lightcurve']['raw_histogram'][-1])
        self.assertEqual(65, actual_dictionary['daily_lightcurve']['number_of_bins'])

        self.assertEqual(-27.84000015258789, actual_dictionary['filter_temperature_average'])
        self.assertEqual(0.0, actual_dictionary['filter_temperature_std_dev'])
        self.assertEqual(1527.0999755859375, actual_dictionary['hv_voltage_average'])
        self.assertEqual(87.94999694824219, actual_dictionary['hv_voltage_std_dev'])
        self.assertEqual(15.0, actual_dictionary['spin_period_average'])
        self.assertEqual(0.0, actual_dictionary['spin_period_std_dev'])
        self.assertEqual(15.236681938171387, actual_dictionary['spin_period_ground_average'])
        self.assertEqual(0.0014979999978095293, actual_dictionary['spin_period_ground_std_dev'])
        self.assertEqual(0.29899999499320984, actual_dictionary['pulse_length_average'])
        self.assertEqual(0.017260000109672546, actual_dictionary['pulse_length_std_dev'])

        self.assertEqual(91.5780029296875, actual_dictionary['position_angle_offset_average'])
        self.assertEqual(0.009991000406444073, actual_dictionary['position_angle_offset_std_dev'])

        self.assertAlmostEqual(162.09199524, actual_dictionary['spin_axis_orientation_average']['lon'])
        self.assertAlmostEqual(0.0, actual_dictionary['spin_axis_orientation_average']['lat'])
        self.assertAlmostEqual(0.23450001, actual_dictionary['spin_axis_orientation_std_dev']['lon'])
        self.assertAlmostEqual(0.0, actual_dictionary['spin_axis_orientation_std_dev']['lat'])

        self.assertAlmostEqual(1.46231104e+08, actual_dictionary['spacecraft_location_average']['x'])
        self.assertAlmostEqual(-3.62120560e+07, actual_dictionary['spacecraft_location_average']['y'])
        self.assertAlmostEqual(1049.300048828125, actual_dictionary['spacecraft_location_average']['z'])

        self.assertAlmostEqual(1.42100000e+05, actual_dictionary['spacecraft_location_std_dev']['x'])
        self.assertAlmostEqual(6.00100000e+05, actual_dictionary['spacecraft_location_std_dev']['y'])
        self.assertAlmostEqual(4.08199997e+01, actual_dictionary['spacecraft_location_std_dev']['z'])

        self.assertAlmostEqual(6.66900015e+00, actual_dictionary['spacecraft_velocity_average']['x'])
        self.assertAlmostEqual(2.88050003e+01, actual_dictionary['spacecraft_velocity_average']['y'])
        self.assertAlmostEqual(-2.00000009e-03, actual_dictionary['spacecraft_velocity_average']['z'])

        self.assertAlmostEqual(1.18799999e-01, actual_dictionary['spacecraft_velocity_std_dev']['x'])
        self.assertAlmostEqual(2.96100006e-02, actual_dictionary['spacecraft_velocity_std_dev']['y'])
        self.assertAlmostEqual(4.33699998e-19, actual_dictionary['spacecraft_velocity_std_dev']['z'])


if __name__ == '__main__':
    unittest.main()
