import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.hi.l3.utils import read_hi_l2_data, read_hi_l1c_data, read_glows_l3e_data
from tests.test_helpers import get_test_data_folder


class TestUtils(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_reads_hi_l2_data(self):
        rng = np.random.default_rng()
        pathname = "test_cdf"
        with CDF(pathname, '') as cdf:
            cdf.col_major(True)

            epoch = np.array([datetime(2000, 1, 1)])
            bins = np.arange(23)
            bin_boundaries = np.repeat(np.arange(23), 2)
            counts = rng.random((2, 180, 90, 23))
            counts_uncertainty = rng.random((2, 180, 90, 23))
            epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
            exposure = rng.random((2, 180, 90))
            flux = rng.random((2, 180, 90, 23))
            lat = np.arange(90)
            lon = np.arange(180)
            sensitivity = rng.random((2, 180, 90, 23))
            variance = rng.random((2, 180, 90, 23))

            cdf["Epoch"] = epoch
            cdf.new("bin", bins, recVary=False)
            cdf.new("bin_boundaries", bin_boundaries, recVary=False)
            cdf["counts"] = counts
            cdf["counts_uncertainty"] = counts_uncertainty
            cdf["epoch_delta"] = epoch_delta
            cdf["exposure"] = exposure
            cdf["flux"] = flux
            cdf.new("lat", lat, recVary=False)
            cdf.new("lon", lon, recVary=False)
            cdf["sensitivity"] = sensitivity
            cdf["variance"] = variance

            for var in cdf:
                cdf[var].attrs['FILLVAL'] = 1000000

        for path in [pathname, Path(pathname)]:
            with self.subTest(path=path):
                result = read_hi_l2_data(path)

                np.testing.assert_array_equal(epoch, result.epoch)
                np.testing.assert_array_equal(bins, result.energy)
                np.testing.assert_array_equal(bin_boundaries, result.energy_deltas)
                np.testing.assert_array_equal(counts, result.counts)
                np.testing.assert_array_equal(counts_uncertainty, result.counts_uncertainty)
                np.testing.assert_array_equal(epoch_delta, result.epoch_delta)
                np.testing.assert_array_equal(exposure, result.exposure)
                np.testing.assert_array_equal(flux, result.flux)
                np.testing.assert_array_equal(lat, result.lat)
                np.testing.assert_array_equal(lon, result.lon)
                np.testing.assert_array_equal(sensitivity, result.sensitivity)
                np.testing.assert_array_equal(variance, result.variance)

    def test_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'l2_map_with_fill_values.cdf'
        result = read_hi_l2_data(path)

        with CDF(str(path)) as cdf:
            np.testing.assert_array_equal(result.epoch, cdf['Epoch'])
            np.testing.assert_array_equal(result.energy, cdf['bin'])
            np.testing.assert_array_equal(result.counts, np.full_like(cdf['counts'], np.nan))
            np.testing.assert_array_equal(result.counts_uncertainty, np.full_like(cdf['counts_uncertainty'], np.nan))
            np.testing.assert_array_equal(result.epoch_delta, cdf['epoch_delta'])
            np.testing.assert_array_equal(result.exposure, np.full_like(cdf['exposure'], np.nan))
            np.testing.assert_array_equal(result.flux, np.full_like(cdf['flux'], np.nan))
            np.testing.assert_array_equal(result.lat, cdf['lat'])
            np.testing.assert_array_equal(result.lon, cdf['lon'])
            np.testing.assert_array_equal(result.sensitivity, np.full_like(cdf['sensitivity'], np.nan))
            np.testing.assert_array_equal(result.variance, np.full_like(cdf['variance'], np.nan))

    def test_read_l1c_data(self):
        rng = np.random.default_rng()
        pathname = "test_cdf"

        with CDF(pathname, '') as cdf:
            epoch = np.array([datetime(2000, 1, 2)])
            exposure_times = rng.random((1, 9, 3600))
            energy_step = rng.random((9))

            cdf.new("epoch", epoch, type=pycdf.const.CDF_TIME_TT2000)
            cdf["exposure_times"] = exposure_times
            cdf["esa_energy_step"] = energy_step
            for var in cdf:
                cdf[var].attrs['FILLVAL'] = 1000000

        for path in [pathname, Path(pathname)]:
            with self.subTest(path=path):
                result = read_hi_l1c_data(path)
                self.assertEqual(epoch[0], result.epoch)
                self.assertEqual([43264184000000], result.epoch_j2000)
                np.testing.assert_array_equal(exposure_times, result.exposure_times)
                np.testing.assert_array_equal(energy_step, result.esa_energy_step)

    def test_read_hi_l1c_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'l1c_pointing_set_with_fill_values.cdf'
        result = read_hi_l1c_data(path)

        with CDF(str(path)) as cdf:
            self.assertEqual(result.epoch, cdf['epoch'][0])
            np.testing.assert_array_equal(result.esa_energy_step, cdf['esa_energy_step'])
            np.testing.assert_array_equal(result.exposure_times, np.full_like(cdf['exposure_times'], np.nan))

    def test_read_glows_l3e_data(self):
        rng = np.random.default_rng()
        pathname = "test_cdf"

        with CDF(pathname, '') as cdf:
            epoch = np.array([datetime(2000, 1, 1)])
            energy = rng.random((16,))
            spin_angle = rng.random(360)
            probability_of_survival = rng.random((1, 16, 360,))

            cdf["epoch"] = epoch
            cdf["energy"] = energy
            cdf["spin_angle"] = spin_angle
            cdf["probability_of_survival"] = probability_of_survival
            for var in cdf:
                cdf[var].attrs['FILLVAL'] = 1000000

        for path in [pathname, Path(pathname)]:
            with self.subTest(path=path):
                result = read_glows_l3e_data(path)

                self.assertEqual(epoch[0], result.epoch)
                np.testing.assert_array_equal(energy, result.energy)
                np.testing.assert_array_equal(spin_angle, result.spin_angle)
                np.testing.assert_array_equal(probability_of_survival, result.probability_of_survival)

    def test_read_glows_l3e_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'l3e_survival_probabilities_with_fill_values.cdf'
        result = read_glows_l3e_data(path)

        with CDF(str(path)) as cdf:
            self.assertEqual(result.epoch, cdf['epoch'][0])
            np.testing.assert_array_equal(result.energy, np.full_like(cdf['energy'], np.nan))
            np.testing.assert_array_equal(result.spin_angle, np.full_like(cdf['spin_angle'], np.nan))
            np.testing.assert_array_equal(result.probability_of_survival,
                                          np.full_like(cdf['probability_of_survival'], np.nan))
