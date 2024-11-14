import unittest

import numpy as np

from imap_processing.glows.l3a.science.calculate_daily_lightcurve import rebin_lightcurve


class TestCalculateDailyLightcurve(unittest.TestCase):
    def test_rebin_with_no_flags_and_equal_exposure_times(self):
        photon_flux = np.full(3600, 1.0)
        photon_flux[1800:] = 5.0
        photon_flux[5]= 41.0
        flags = np.full((4, 3600), 0)
        exposure_times= np.full(3600, 24.0)
        output_size = 90
        result, rebinned_exposure = rebin_lightcurve(photon_flux, flags, exposure_times, output_size)

        expected_output = np.full(90, 1.0)
        expected_output[45:] = 5.0
        expected_output[0] = 2.0
        np.testing.assert_equal(result, expected_output, strict=True)

        expected_exposure = np.full(90, 24.0*40)
        np.testing.assert_equal(rebinned_exposure, expected_exposure, strict=True)

    def test_rebin_with_different_output_size(self):
        photon_flux = np.full(2400, 1.0)
        flags = np.full((4,2400),0)
        exposure_times = np.full(2400, 10.0)
        result, exposure = rebin_lightcurve(photon_flux, flags, exposure_times, 80)
        expected_output = np.full(80, 1.0)
        expected_exposure = np.full(80, 300.0)
        np.testing.assert_equal(result, expected_output, strict=True)
        np.testing.assert_equal(exposure, expected_exposure, strict=True)


    def test_rebin_masking_flagged_bins(self):
        photon_flux = np.full(3600, 1.0)
        photon_flux[1800:] = 5.0
        photon_flux[80:120] = 100.0
        photon_flux[0] = 41
        flags = np.full((4, 3600), 0)
        flags[0, 80:100] = 1
        flags[1, 100:120] = 1
        flags[2, 20:30] = 1
        flags[3, 30:40] = 1
        exposure_times= np.full(3600, 24.0)
        output_size = 90
        result, exposure = rebin_lightcurve(photon_flux, flags, exposure_times, output_size)

        expected_output = np.full(90, 1.0)
        expected_output[45:] = 5.0
        expected_output[2] = 0.0
        expected_output[0] = 3.0
        np.testing.assert_equal(result, expected_output, strict=True)
        self.assertFalse(np.ma.isMaskedArray(result))

        expected_exposure = np.full(90, 24.0*40)
        expected_exposure[0] = 24.0*20
        expected_exposure[2] = 0
        np.testing.assert_equal(expected_exposure, exposure, strict=True)
        self.assertFalse(np.ma.isMaskedArray(exposure))

    def test_rebin_uses_exposure_times(self):
        photon_flux = np.array([10, 20, 30, 40])
        flags = np.zeros((4,4))
        exposure_times = np.array([2, 1, 0, 1])
        result, exposure = rebin_lightcurve(photon_flux, flags, exposure_times, 1)
        np.testing.assert_equal(result, [20.0], strict=True)
        np.testing.assert_equal(exposure, [4.0], strict=True)

    def test_rebin_uses_exposure_times_all_zero_bin(self):
        photon_flux = np.array([10, 20, 30, 40])
        flags = np.zeros((4,4))
        exposure_times = np.array([0, 0, 0, 0])
        result, exposure = rebin_lightcurve(photon_flux, flags, exposure_times, 1)
        np.testing.assert_equal(result, [0.0], strict=True)
        np.testing.assert_equal(exposure, [0.0], strict=True)

if __name__ == '__main__':
    unittest.main()
