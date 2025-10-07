import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3e.glows_l3e_hi_model import PROBABILITY_OF_SURVIVAL_VAR_NAME, SPIN_ANGLE_VAR_NAME, \
    ENERGY_VAR_NAME, EPOCH_CDF_VAR_NAME
from imap_l3_processing.hi.l3.utils import read_l1c_rectangular_pointing_set_data, read_glows_l3e_data
from tests.test_helpers import get_test_data_folder


class TestUtils(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_read_l1c_data(self):
        rng = np.random.default_rng()
        pathname = "test_cdf"

        cases = ['exposure_times', 'exposure_time']

        for exposure_time_var_name in cases:
            with CDF(pathname, '') as cdf:
                epoch = np.array([datetime(2000, 1, 2)])
                exposure_times = rng.random((1, 9, 3600))
                energy_step = rng.random((9))

                cdf.new("epoch", epoch, type=pycdf.const.CDF_TIME_TT2000)
                cdf[exposure_time_var_name] = exposure_times
                cdf["esa_energy_step"] = energy_step
                for var in cdf:
                    cdf[var].attrs['FILLVAL'] = 1000000

            for path in [pathname, Path(pathname)]:
                with self.subTest(path=path, exposure_time_var_name=exposure_time_var_name):
                    result = read_l1c_rectangular_pointing_set_data(path)
                    self.assertEqual(epoch[0], result.epoch)
                    self.assertEqual([43264184000000], result.epoch_j2000)
                    np.testing.assert_array_equal(exposure_times, result.exposure_times)
                    np.testing.assert_array_equal(energy_step, result.esa_energy_step)

            self.tearDown()


    def test_read_hi_l1c_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'l1c_pointing_set_with_fill_values.cdf'
        result = read_l1c_rectangular_pointing_set_data(path)

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

            cdf[EPOCH_CDF_VAR_NAME] = epoch
            cdf[ENERGY_VAR_NAME] = energy
            cdf[SPIN_ANGLE_VAR_NAME] = spin_angle
            cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME] = probability_of_survival
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
            self.assertEqual(result.epoch, cdf[EPOCH_CDF_VAR_NAME][0])
            np.testing.assert_array_equal(result.energy, np.full_like(cdf[ENERGY_VAR_NAME], np.nan))
            np.testing.assert_array_equal(result.spin_angle, np.full_like(cdf[SPIN_ANGLE_VAR_NAME], np.nan))
            np.testing.assert_array_equal(result.probability_of_survival,
                                          np.full_like(cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME], np.nan))
