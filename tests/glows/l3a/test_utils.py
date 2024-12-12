import json
import unittest
from datetime import datetime

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.glows.l3a.models import GlowsL2Data, GlowsL2Header
from imap_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_from_dictionary
from imap_processing.models import UpstreamDataDependency
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    def test_reading_l2_glows_data_into_models(self):
        cdf = CDF(str(get_test_data_path('glows/imap_glows_l2_hist_20130908_v001.cdf')))

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
        self.assertEqual(np.uint8, glows_l2_data["daily_lightcurve"]["histogram_flag_array"].dtype)
        self.assertEqual(np.uint8(4), glows_l2_data["daily_lightcurve"]["histogram_flag_array"][84])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["ecliptic_lat"].shape)
        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["ecliptic_lon"].shape)
        self.assertEqual(75.000, glows_l2_data["daily_lightcurve"]["ecliptic_lat"][0])
        self.assertEqual(161.823, glows_l2_data["daily_lightcurve"]["ecliptic_lon"][0])

        self.assertEqual((3600,), glows_l2_data["daily_lightcurve"]["raw_histogram"].shape)
        self.assertEqual(13677, glows_l2_data["daily_lightcurve"]["raw_histogram"][0])

        self.assertEqual({
            "lon": 162.0919952392578,
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
        self.assertEqual("imap_glows_l2_hist_20130908_v001.cdf", glows_l2_data['l2_file_name'])

        self.assert_equal_xyz(cdf, "spacecraft_location_average", glows_l2_data)
        self.assert_equal_xyz(cdf, "spacecraft_location_std_dev", glows_l2_data)
        self.assert_equal_xyz(cdf, "spacecraft_velocity_average", glows_l2_data)
        self.assert_equal_xyz(cdf, "spacecraft_velocity_std_dev", glows_l2_data)

        expected_header = GlowsL2Header(
            flight_software_version=131848,
            pkts_file_name="data_l0/imap_l0_sci_glows_20241018_mockByGlowsTeam051_v00.pkts",
            ancillary_data_files=[
                "data_ancillary/imap_glows_conversion_table_for_anc_data_v002.json",
                "data_ancillary/imap_glows_calibration_data_v002.dat",
                "data_ancillary/imap_glows_pipeline_settings_v002.json",
                "data_ancillary/imap_glows_map_of_uv_sources_v002.dat",
                "data_ancillary/imap_glows_map_of_excluded_regions_v002.dat",
                "data_ancillary/imap_glows_exclusions_by_instr_team_v002.dat",
                "data_ancillary/imap_glows_suspected_transients_v002.dat",
                "data_ancillary/imap_glows_map_of_extra_helio_bckgrd_v001.dat",
                "data_ancillary/imap_glows_time_dep_bckgrd_v001.dat",
                "data_ancillary/imap_l1_anc_sc_Merged_2010-2030_mockByGlowsTeam001.csv"],
        )
        self.assertEqual(expected_header, glows_l2_data["header"])

    def assert_equal_xyz(self, cdf, variable_name, actual):
        expected_spacecraft_location_average = {
            'x': cdf[f'{variable_name}_x'][0],
            'y': cdf[f'{variable_name}_y'][0],
            'z': cdf[f'{variable_name}_z'][0],
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
            upstream_data_dependency = UpstreamDataDependency(instrument='glows', data_level='l3a',
                                                              descriptor='histogram', start_date=datetime(2013, 9, 8),
                                                              end_date=datetime(2013, 9, 9), version="v003")
            result = create_glows_l3a_from_dictionary(data, upstream_data_dependency)

            self.assertEqual(1, len(result.epoch))
            self.assertEqual(datetime(2013, 9, 8, 18, 55, 14), result.epoch[0])

            self.assertEqual((1, 65), result.spin_angle.shape)
            self.assertEqual(2.000, result.spin_angle[0, 0])

            self.assertEqual((1, 65), result.photon_flux.shape)
            self.assertEqual(620.9, result.photon_flux[0, 0])

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

            self.assertEqual(upstream_data_dependency, result.input_metadata)


if __name__ == '__main__':
    unittest.main()
