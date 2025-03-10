import unittest
from unittest.mock import Mock

import numpy as np
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.glows.l3a.science.bad_angle_flag_configuration import BadAngleFlagConfiguration
from imap_l3_processing.glows.l3a.science.calculate_daily_lightcurve import rebin_lightcurve, calculate_spin_angles
from imap_l3_processing.glows.l3a.science.time_independent_background_lookup_table import \
    TimeIndependentBackgroundLookupTable


class TestCalculateDailyLightcurve(unittest.TestCase):
    def setUp(self):
        self.all_zero_time_independent_background = TimeIndependentBackgroundLookupTable(
            latitudes=np.array([-90, 90]),
            longitudes=np.array([0]),
            background_values=np.zeros(
                shape=(2, 1)))
        self.enable_all_flags_configuration = BadAngleFlagConfiguration(True, True, True, True)

    def test_rebin_with_no_flags_and_equal_exposure_times(self):
        photon_flux = np.full(3600, 1.0)
        photon_flux[1800:] = 5.0
        photon_flux[5] = 41.0
        latitudes = np.linspace(start=0, stop=0, num=3600)
        longitudes = np.linspace(start=0, stop=360, num=3600)

        flags = np.full((4, 3600), 0)
        exposure_times = np.full(3600, 24.0)
        output_size = 90
        background = np.zeros(90)
        result, rebinned_exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                                     self.enable_all_flags_configuration,
                                                     photon_flux, latitudes, longitudes, flags, exposure_times,
                                                     output_size, background)

        expected_output = np.full(90, 1.0)
        expected_output[45:] = 5.0
        expected_output[0] = 2.0
        np.testing.assert_equal(result, expected_output, strict=True)

        expected_exposure = np.full(90, 24.0 * 40)
        np.testing.assert_equal(rebinned_exposure, expected_exposure, strict=True)

    def test_rebin_with_different_output_size(self):
        photon_flux = np.full(2400, 1.0)
        flags = np.full((4, 2400), 0)
        exposure_times = np.full(2400, 10.0)
        background = np.zeros(80)
        latitudes = np.linspace(start=0, stop=0, num=2400)
        longitudes = np.linspace(start=0, stop=360, num=2400)

        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, 80, background)
        expected_output = np.full(80, 1.0)
        expected_exposure = np.full(80, 300.0)
        np.testing.assert_equal(result, expected_output, strict=True)
        np.testing.assert_equal(exposure, expected_exposure, strict=True)

    def test_rebin_masking_flagged_bins(self):
        photon_flux = np.full(3600, 1.0)
        photon_flux[1800:] = 5.0
        photon_flux[80:120] = 100.0
        photon_flux[0] = 41
        flags = np.full((4, 3600), False)
        flags[0, 80:100] = True
        flags[1, 100:120] = True
        flags[2, 20:30] = True
        flags[3, 30:40] = True

        exposure_times = np.full(3600, 24.0)
        output_size = 90
        background = np.zeros(90)
        latitudes = np.linspace(start=0, stop=0, num=3600)
        longitudes = np.linspace(start=0, stop=360, num=3600)

        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, output_size, background)

        expected_output = np.full(90, 1.0)
        expected_output[45:] = 5.0
        expected_output[2] = np.nan
        expected_output[0] = 3.0
        np.testing.assert_equal(result, expected_output, strict=True)
        self.assertFalse(np.ma.isMaskedArray(result))

        expected_exposure = np.full(90, 24.0 * 40)
        expected_exposure[0] = 24.0 * 20
        expected_exposure[2] = 0
        np.testing.assert_equal(expected_exposure, exposure, strict=True)
        self.assertFalse(np.ma.isMaskedArray(exposure))

    def test_uses_configuration_to_flag_bins(self):
        photon_flux = np.full(3600, 1.0)
        photon_flux[1800:] = 5.0
        photon_flux[80:120] = 100.0
        photon_flux[0] = 41
        flags = np.full((4, 3600), False)
        flags[0, 80:100] = True
        flags[1, 100:120] = True
        flags[2, 800:900] = True
        flags[3, 20:30] = True
        flags[3, 30:40] = True

        exposure_times = np.full(3600, 24.0)
        output_size = 90
        background = np.zeros(90)
        latitudes = np.linspace(start=0, stop=0, num=3600)
        longitudes = np.linspace(start=0, stop=360, num=3600)

        flag_configuration = BadAngleFlagConfiguration(mask_close_to_uv_source=True,
                                                       mask_inside_excluded_region=True,
                                                       mask_excluded_by_instr_team=False,
                                                       mask_suspected_transient=True, )
        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background, flag_configuration, photon_flux,
                                            latitudes,
                                            longitudes, flags,
                                            exposure_times, output_size, background)

        expected_output = np.full(90, 1.0)
        expected_output[45:] = 5.0
        expected_output[2] = np.nan
        expected_output[0] = 3.0
        np.testing.assert_equal(result, expected_output, strict=True)
        self.assertFalse(np.ma.isMaskedArray(result))

        expected_exposure = np.full(90, 24.0 * 40)
        expected_exposure[0] = 24.0 * 20
        expected_exposure[2] = 0
        np.testing.assert_equal(expected_exposure, exposure, strict=True)
        self.assertFalse(np.ma.isMaskedArray(exposure))

    def test_rebin_uses_exposure_times(self):
        photon_flux = np.array([10., 20., 30., 40.])
        flags = np.zeros((4, 4))
        exposure_times = np.array([2., 1., 0., 1.])
        background = np.zeros(1)
        latitudes = np.linspace(start=0, stop=0, num=4)
        longitudes = np.linspace(start=0, stop=360, num=4)

        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, 1, background)
        np.testing.assert_equal(result, [20.0], strict=True)
        np.testing.assert_equal(exposure, [4.0], strict=True)

    def test_rebin_uses_exposure_times_all_zero_bin(self):
        photon_flux = np.array([10., 20., 30., 40.])
        flags = np.zeros((4, 4))
        exposure_times = np.array([0., 0., 0., 0.])
        background = np.zeros(1)
        latitudes = np.linspace(start=0, stop=0, num=4)
        longitudes = np.linspace(start=0, stop=360, num=4)

        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, 1, background)
        np.testing.assert_equal(result, [np.nan], strict=True)
        np.testing.assert_equal(exposure, [0.0], strict=True)

    def test_rebin_uses_exposure_times_and_background(self):
        photon_flux = np.array([5., 20., 30., 40.])
        flags = np.zeros((4, 4))
        exposure_times = np.array([2., 1., 0., 1.])
        background = np.array([5, 4])
        latitudes = np.linspace(start=0, stop=0, num=4)
        longitudes = np.linspace(start=0, stop=360, num=4)

        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, 2, background)
        np.testing.assert_equal(result, [5.0, 36.0], strict=True)
        np.testing.assert_equal(exposure, [3.0, 1.0], strict=True)

    def test_rebin_uses_uncertain_photon_flux_and_background(self):
        photon_flux = np.array([5, 20, 30, 40])
        photon_flux_uncertainty = np.array([0.1, 2.0, 3.0, .5])
        photon_flux = uarray(photon_flux, photon_flux_uncertainty)
        latitudes = np.linspace(start=0, stop=0, num=4)
        longitudes = np.linspace(start=0, stop=360, num=4)

        flags = np.zeros((4, 4))
        exposure_times = np.array([2., 1., 0., 1.])
        background = np.array([5, 4])
        background_uncertainty = np.array([0.02, 0.08])
        background = uarray(background, background_uncertainty)

        result, exposure = rebin_lightcurve(self.all_zero_time_independent_background,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, 2, background)
        np.testing.assert_equal(nominal_values(result), [5.0, 36.0], strict=True)
        np.testing.assert_equal(std_devs(result), [0.6702901527613909, 0.5063595560468865], strict=True)
        np.testing.assert_equal(exposure, [3.0, 1.0], strict=True)

    def test_rebin_subtracts_time_independent_background(self):
        time_independent_background = np.array([20, 2, 1, 0.2])

        photon_flux = np.array([5, 20, 30, 40]) + time_independent_background
        photon_flux_uncertainty = np.array([0.1, 2.0, 3.0, .5])
        photon_flux = uarray(photon_flux, photon_flux_uncertainty)
        latitudes = np.linspace(start=0, stop=0, num=4)
        longitudes = np.linspace(start=0, stop=360, num=4)

        flags = np.zeros((4, 4))
        exposure_times = np.array([2., 1., 0., 1.])
        background = np.array([5, 4])
        background_uncertainty = np.array([0.02, 0.08])
        background = uarray(background, background_uncertainty)

        mock_time_independent_background_lookup_table = Mock()
        mock_time_independent_background_lookup_table.lookup.return_value = time_independent_background

        result, exposure = rebin_lightcurve(mock_time_independent_background_lookup_table,
                                            self.enable_all_flags_configuration, photon_flux, latitudes,
                                            longitudes, flags,
                                            exposure_times, 2, background)

        mock_time_independent_background_lookup_table.lookup.assert_called_once_with(lat=latitudes,
                                                                                     lon=longitudes)

        np.testing.assert_equal(nominal_values(result), [5.0, 36.0], strict=True)
        np.testing.assert_equal(std_devs(result), [0.6702901527613909, 0.5063595560468865], strict=True)
        np.testing.assert_equal(exposure, [3.0, 1.0], strict=True)

    def test_calculate_spin_angles(self):
        spin_angles = np.array([70, 130, 190, 250, 310, 10, 70, 130, 190, 350, 10, 30])

        np.testing.assert_array_equal([130, 310, 130, 10], calculate_spin_angles(4, spin_angles))

        realistic_angles = np.linspace(0, 360, 3600, endpoint=False) + 0.05
        expected_4_degree_bins = np.linspace(0, 360, 90, endpoint=False) + 2
        np.testing.assert_allclose(expected_4_degree_bins, calculate_spin_angles(90, realistic_angles))


if __name__ == '__main__':
    unittest.main()
