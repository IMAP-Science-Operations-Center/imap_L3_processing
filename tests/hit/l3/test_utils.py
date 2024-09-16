import os
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.hit.l3.utils import read_l2_hit_data
from imap_processing.utils import read_l2_mag_data


class TestUtils(TestCase):
    def setUp(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_read_l2_hit_data(self):
        hit_cdf = CDF('test_cdf', '')
        hit_cdf["Epoch"] = np.array([datetime(2010, 1, 1, 0, 0, 46)])
        hit_cdf["Epoch_DELTA"] = [1800000000000]
        hit_cdf["R26A_H_SECT_Flux"] = np.array([1, 2, 3, 4])
        hit_cdf["R26A_H_SECT_Rate"] = np.array([5, 6, 7, 8])
        hit_cdf["R26A_H_SECT_Uncertainty"] = np.array([2, 2, 2, 2, 2, 2, 2, 2])
        results = read_l2_hit_data(hit_cdf)

        epoch_as_tt2000 = 315576112184000000
        np.testing.assert_array_equal([epoch_as_tt2000], results.epoch)
        np.testing.assert_array_equal(hit_cdf["Epoch_DELTA"], results.epoch_delta)
        np.testing.assert_array_equal(hit_cdf["R26A_H_SECT_Flux"], results.flux)
        np.testing.assert_array_equal(hit_cdf["R26A_H_SECT_Rate"], results.count_rates)
        np.testing.assert_array_equal(hit_cdf["R26A_H_SECT_Uncertainty"], results.uncertainty)

