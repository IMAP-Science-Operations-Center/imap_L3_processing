import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_processing.hit.l3.utils import read_l2_hit_data, calculate_pitch_angle, calculate_unit_vector


class TestUtils(TestCase):
    def setUp(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_read_l2_hit_data(self):
        rng = np.random.default_rng()
        pathname = 'test_cdf'
        with CDF(pathname, '') as cdf:
            cdf.col_major(True)

            epoch_count = 1
            start_time = datetime(2010, 1, 1, 0, 5)
            epoch_data = np.array([start_time])
            epoch_delta = np.full(epoch_count, FIVE_MINUTES_IN_NANOSECONDS)

            hydrogen_data = rng.random((1, 3, 8, 15))
            helium_data = rng.random((1, 2, 8, 15))
            cno_data = rng.random((1, 2, 8, 15))
            nemgsi_data = rng.random((1, 2, 8, 15))
            iron_data = rng.random((1, 1, 8, 15))
            cdf["hydrogen"] = hydrogen_data
            cdf["helium4"] = helium_data
            cdf["CNO"] = cno_data
            cdf["NeMgSi"] = nemgsi_data
            cdf["iron"] = iron_data

            cdf["epoch"] = epoch_data
            cdf["epoch_delta"] = epoch_delta

            cdf.new("h_energy_idx", np.arange(3), recVary=False)
            cdf.new("he4_energy_idx", np.arange(2), recVary=False)
            cdf.new("cno_energy_idx", np.arange(2), recVary=False)
            cdf.new("nemgsi_energy_idx", np.arange(2), recVary=False)
            cdf.new("fe_energy_idx", np.arange(1), recVary=False)

            hydrogen_delta = hydrogen_data * 0.1
            helium_delta = helium_data * 0.1
            cno_delta = cno_data * 0.1
            nemgsi_delta = nemgsi_data * 0.1
            iron_delta = iron_data * 0.1
            cdf["DELTA_PLUS_HYDROGEN"] = hydrogen_delta
            cdf["DELTA_MINUS_HYDROGEN"] = hydrogen_delta
            cdf["DELTA_PLUS_HELIUM4"] = helium_delta
            cdf["DELTA_MINUS_HELIUM4"] = helium_delta
            cdf["DELTA_PLUS_CNO"] = cno_delta
            cdf["DELTA_MINUS_CNO"] = cno_delta
            cdf["DELTA_PLUS_NEMGSI"] = nemgsi_delta
            cdf["DELTA_MINUS_NEMGSI"] = nemgsi_delta
            cdf["DELTA_PLUS_IRON"] = iron_delta
            cdf["DELTA_MINUS_IRON"] = iron_delta

            cdf.new("h_energy_high", [4, 6, 10], recVary=False)
            cdf.new("h_energy_low", np.array([1.8, 4, 6]), recVary=False)
            cdf.new("he4_energy_high", [6, 12], recVary=False)
            cdf.new("he4_energy_low", [4, 6], recVary=False)
            cdf.new("cno_energy_high", [6, 12], recVary=False)
            cdf.new("cno_energy_low", [4, 6], recVary=False)
            cdf.new("nemgsi_energy_high", [6, 12], recVary=False)
            cdf.new("nemgsi_energy_low", [4, 6], recVary=False)
            cdf.new("fe_energy_high", [12], recVary=False)
            cdf.new("fe_energy_low", [4], recVary=False)

        for path in [pathname, Path(pathname)]:
            with self.subTest(path):
                result = read_l2_hit_data(path)

                np.testing.assert_array_equal(hydrogen_data, result.hydrogen)
                np.testing.assert_array_equal(helium_data, result.helium4)
                np.testing.assert_array_equal(cno_data, result.CNO)
                np.testing.assert_array_equal(nemgsi_data, result.NeMgSi)
                np.testing.assert_array_equal(iron_data, result.iron)

                np.testing.assert_array_equal(epoch_data, result.epoch)
                np.testing.assert_array_equal([timedelta(minutes=5)], result.epoch_delta)

                np.testing.assert_array_equal([0, 1, 2], result.h_energy_idx)
                np.testing.assert_array_equal([0, 1], result.he4_energy_idx)
                np.testing.assert_array_equal([0, 1], result.cno_energy_idx)
                np.testing.assert_array_equal([0, 1], result.nemgsi_energy_idx)
                np.testing.assert_array_equal([0], result.fe_energy_idx)

                np.testing.assert_array_equal(hydrogen_delta, result.DELTA_PLUS_HYDROGEN)
                np.testing.assert_array_equal(hydrogen_delta, result.DELTA_MINUS_HYDROGEN)
                np.testing.assert_array_equal(helium_delta, result.DELTA_PLUS_HELIUM4)
                np.testing.assert_array_equal(helium_delta, result.DELTA_MINUS_HELIUM4)
                np.testing.assert_array_equal(cno_delta, result.DELTA_PLUS_CNO)
                np.testing.assert_array_equal(cno_delta, result.DELTA_MINUS_CNO)
                np.testing.assert_array_equal(nemgsi_delta, result.DELTA_PLUS_NEMGSI)
                np.testing.assert_array_equal(nemgsi_delta, result.DELTA_MINUS_NEMGSI)
                np.testing.assert_array_equal(iron_delta, result.DELTA_PLUS_IRON)
                np.testing.assert_array_equal(iron_delta, result.DELTA_MINUS_IRON)

                np.testing.assert_array_equal([4, 6, 10], result.h_energy_high)
                np.testing.assert_array_equal([1.8, 4, 6], result.h_energy_low)
                np.testing.assert_array_equal([6, 12], result.he4_energy_high)
                np.testing.assert_array_equal([4, 6], result.he4_energy_low)
                np.testing.assert_array_equal([6, 12], result.cno_energy_high)
                np.testing.assert_array_equal([4, 6], result.cno_energy_low)
                np.testing.assert_array_equal([6, 12], result.nemgsi_energy_high)
                np.testing.assert_array_equal([4, 6], result.nemgsi_energy_low)
                np.testing.assert_array_equal([12], result.fe_energy_high)
                np.testing.assert_array_equal([4], result.fe_energy_low)

    def test_calculate_pitch_angle(self):
        hit_unit_vector = np.array([-0.09362045, 0.8466484, 0.5238528])
        mag_unit_vector = np.array([-0.42566603, 0.7890057, 0.44303328])

        actual_pitch_angle = calculate_pitch_angle(hit_unit_vector, mag_unit_vector)
        self.assertAlmostEqual(19.957563418693166, actual_pitch_angle)

    def test_calculate_pitch_angle_throws_exception_if_inputs_of_unequal_length(self):
        vector_a = np.array([1, 2, 3, 4, 5])
        vector_b = np.array([6, 7])

        with self.assertRaises(Exception) as cm:
            calculate_pitch_angle(vector_a, vector_b)
        self.assertEqual(str(cm.exception), "Input vectors are of unequal length 5 and 2")

    def test_calculate_unit_vector(self):
        vector = np.array([27, 34, 56])

        unit_vector = calculate_unit_vector(vector)

        np.testing.assert_array_almost_equal(np.array([0.38103832, 0.47982603, 0.7903017]), unit_vector)
