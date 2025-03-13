import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.hi.l3.utils import read_hi_l3_data


class TestUtils(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_reads_hi_l3_data(self):
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

        for path in [pathname, Path(pathname)]:
            with self.subTest(path=path):
                result = read_hi_l3_data(path)

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
