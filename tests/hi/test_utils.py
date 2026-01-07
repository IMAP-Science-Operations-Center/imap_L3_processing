import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3e.glows_l3e_hi_model import PROBABILITY_OF_SURVIVAL_VAR_NAME, SPIN_ANGLE_VAR_NAME, \
    ENERGY_VAR_NAME, EPOCH_CDF_VAR_NAME
from imap_l3_processing.hi.utils import read_l1c_rectangular_pointing_set_data, read_glows_l3e_data
from tests.test_helpers import get_test_data_folder, with_tempdir


class TestUtils(unittest.TestCase):
    def setUp(self):
        if os.path.exists('imap_hi_l1c_pointing-set_20250101-repoint01111_v000.cdf'):
            os.remove('imap_hi_l1c_pointing-set_20250101-repoint01111_v000.cdf')

    def tearDown(self):
        if os.path.exists('imap_hi_l1c_pointing-set_20250101-repoint01111_v000.cdf'):
            os.remove('imap_hi_l1c_pointing-set_20250101-repoint01111_v000.cdf')

    def test_read_l1c_data(self):
        rng = np.random.default_rng()
        pathname = "imap_hi_l1c_pointing-set_20250101-repoint01111_v000.cdf"

        cases = ['exposure_times', 'exposure_time']

        for exposure_time_var_name in cases:
            with CDF(pathname, '') as cdf:
                epoch = np.array([datetime(2000, 1, 2)])
                epoch_delta = np.array([1_000_000_000])
                repointing = 1111
                exposure_times = rng.random((1, 9, 3600))
                energy_step = rng.random((9))
                hae_longitude = np.ones(3600).reshape((1, 3600))
                hae_latitude = np.ones(3600).reshape((1, 3600))

                cdf.new("epoch", epoch, type=pycdf.const.CDF_TIME_TT2000)
                cdf["epoch_delta"] = epoch_delta
                cdf[exposure_time_var_name] = exposure_times
                cdf["esa_energy_step"] = energy_step
                cdf['hae_longitude'] = hae_longitude
                cdf['hae_latitude'] = hae_latitude

                for var in cdf:
                    cdf[var].attrs['FILLVAL'] = 1000000

            for path in [pathname, Path(pathname)]:
                with self.subTest(path=path, exposure_time_var_name=exposure_time_var_name):
                    result = read_l1c_rectangular_pointing_set_data(path)
                    self.assertEqual(epoch[0], result.epoch)
                    self.assertEqual(epoch_delta, result.epoch_delta)
                    self.assertEqual(repointing, result.repointing)
                    self.assertEqual([43264184000000], result.epoch_j2000)
                    np.testing.assert_array_equal(exposure_times, result.exposure_times)
                    np.testing.assert_array_equal(energy_step, result.esa_energy_step)
                    np.testing.assert_array_equal(hae_longitude, result.hae_longitude)
                    np.testing.assert_array_equal(hae_latitude, result.hae_latitude)

            self.tearDown()

    def test_read_hi_l1c_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'imap_hi_l1c_pset-with-fill-values_20250101-repoint00001_v000.cdf'
        result = read_l1c_rectangular_pointing_set_data(path)

        with CDF(str(path)) as cdf:
            self.assertEqual(cdf['epoch'][0], result.epoch)
            self.assertEqual(cdf['epoch_delta'][0], result.epoch_delta)
            self.assertEqual(1, result.repointing)
            np.testing.assert_array_equal(result.esa_energy_step, cdf['esa_energy_step'])
            np.testing.assert_array_equal(result.exposure_times, np.full_like(cdf['exposure_times'], np.nan))

    def test_read_lo_l1c(self):
        path = get_test_data_folder() / 'lo' / 'imap_lo_l1c_pset_20260101-repoint01261_v001.cdf'
        result = read_l1c_rectangular_pointing_set_data(path)

        with CDF(str(path)) as cdf:
            self.assertEqual(cdf['epoch'][0], result.epoch)
            self.assertEqual(None, result.epoch_delta)
            self.assertEqual(1261, result.repointing)
            np.testing.assert_array_equal(result.esa_energy_step, cdf['esa_energy_step'])
            np.testing.assert_array_equal(result.exposure_times, cdf['exposure_time'][...])
            self.assertEqual(cdf['pointing_start_met'][0], result.pointing_start_met)
            self.assertEqual(cdf['pointing_end_met'][0], result.pointing_end_met)

    @with_tempdir
    def test_read_glows_l3e_data(self, temp_dir: Path):
        rng = np.random.default_rng()

        pathname = str(temp_dir / "imap_glows_l3e_survival-probability-hi_20250415-repoint01000_v001.cdf")

        with CDF(pathname, '') as cdf:
            epoch = np.array([datetime(2000, 1, 2)])
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
                self.assertEqual([43264184000000], result.epoch_j2000)

                self.assertEqual(1000, result.repointing)
                np.testing.assert_array_equal(energy, result.energy)
                np.testing.assert_array_equal(spin_angle, result.spin_angle)
                np.testing.assert_array_equal(probability_of_survival, result.probability_of_survival)

    def test_read_glows_l3e_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'imap_glows_l3e_survival-probabilities-with-fill-values_20250101-repoint01111_v000.cdf'
        result = read_glows_l3e_data(path)

        with CDF(str(path)) as cdf:
            self.assertEqual(cdf[EPOCH_CDF_VAR_NAME][0], result.epoch)
            self.assertEqual(cdf.raw_var(EPOCH_CDF_VAR_NAME)[...], result.epoch_j2000)

            self.assertEqual(1111, result.repointing)
            np.testing.assert_array_equal(result.energy, np.full_like(cdf[ENERGY_VAR_NAME], np.nan))
            np.testing.assert_array_equal(result.spin_angle, np.full_like(cdf[SPIN_ANGLE_VAR_NAME], np.nan))
            np.testing.assert_array_equal(result.probability_of_survival,
                                          np.full_like(cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME], np.nan))
