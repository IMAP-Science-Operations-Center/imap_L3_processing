import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.hit.l3.utils import read_l2_hit_data
from tests.test_helpers import get_test_data_folder


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
            cdf["h"] = hydrogen_data
            cdf["he4"] = helium_data
            cdf["cno"] = cno_data
            cdf["nemgsi"] = nemgsi_data
            cdf["fe"] = iron_data

            cdf["epoch"] = epoch_data
            cdf["epoch_delta"] = epoch_delta

            cdf.new("h_energy_mean", np.arange(3), recVary=False)
            cdf.new("he4_energy_mean", np.arange(2), recVary=False)
            cdf.new("cno_energy_mean", np.arange(2), recVary=False)
            cdf.new("nemgsi_energy_mean", np.arange(2), recVary=False)
            cdf.new("fe_energy_mean", np.arange(1), recVary=False)

            hydrogen_delta = hydrogen_data * 0.1
            helium_delta = helium_data * 0.1
            cno_delta = cno_data * 0.1
            nemgsi_delta = nemgsi_data * 0.1
            iron_delta = iron_data * 0.1
            cdf["delta_plus_h"] = hydrogen_delta
            cdf["delta_minus_h"] = hydrogen_delta
            cdf["delta_plus_he4"] = helium_delta
            cdf["delta_minus_he4"] = helium_delta
            cdf["delta_plus_cno"] = cno_delta
            cdf["delta_minus_cno"] = cno_delta
            cdf["delta_plus_nemgsi"] = nemgsi_delta
            cdf["delta_minus_nemgsi"] = nemgsi_delta
            cdf["delta_plus_fe"] = iron_delta
            cdf["delta_minus_fe"] = iron_delta

            cdf.new("h_energy_delta_plus", [4, 6, 10], recVary=False)
            cdf.new("h_energy_delta_minus", np.array([1.8, 4, 6]), recVary=False)
            cdf.new("he4_energy_delta_plus", [6, 12], recVary=False)
            cdf.new("he4_energy_delta_minus", [4, 6], recVary=False)
            cdf.new("cno_energy_delta_plus", [6, 12], recVary=False)
            cdf.new("cno_energy_delta_minus", [4, 6], recVary=False)
            cdf.new("nemgsi_energy_delta_plus", [6, 12], recVary=False)
            cdf.new("nemgsi_energy_delta_minus", [4, 6], recVary=False)
            cdf.new("fe_energy_delta_plus", [12], recVary=False)
            cdf.new("fe_energy_delta_minus", [4], recVary=False)
            
            for var in cdf:
                cdf[var].attrs["FILLVAL"] = -1e31

        for path in [pathname, Path(pathname)]:
            with self.subTest(path):
                result = read_l2_hit_data(path)

                np.testing.assert_array_equal(hydrogen_data, result.h)
                np.testing.assert_array_equal(helium_data, result.he4)
                np.testing.assert_array_equal(cno_data, result.cno)
                np.testing.assert_array_equal(nemgsi_data, result.nemgsi)
                np.testing.assert_array_equal(iron_data, result.fe)

                np.testing.assert_array_equal(epoch_data, result.epoch)
                np.testing.assert_array_equal([timedelta(minutes=5)], result.epoch_delta)

                np.testing.assert_array_equal([0, 1, 2], result.h_energy)
                np.testing.assert_array_equal([0, 1], result.he4_energy)
                np.testing.assert_array_equal([0, 1], result.cno_energy)
                np.testing.assert_array_equal([0, 1], result.nemgsi_energy)
                np.testing.assert_array_equal([0], result.fe_energy)

                np.testing.assert_array_equal(hydrogen_delta, result.delta_plus_h)
                np.testing.assert_array_equal(hydrogen_delta, result.delta_minus_h)
                np.testing.assert_array_equal(helium_delta, result.delta_plus_he4)
                np.testing.assert_array_equal(helium_delta, result.delta_minus_he4)
                np.testing.assert_array_equal(cno_delta, result.delta_plus_cno)
                np.testing.assert_array_equal(cno_delta, result.delta_minus_cno)
                np.testing.assert_array_equal(nemgsi_delta, result.delta_plus_nemgsi)
                np.testing.assert_array_equal(nemgsi_delta, result.delta_minus_nemgsi)
                np.testing.assert_array_equal(iron_delta, result.delta_plus_fe)
                np.testing.assert_array_equal(iron_delta, result.delta_minus_fe)

                np.testing.assert_array_equal([4, 6, 10], result.h_energy_delta_plus)
                np.testing.assert_array_equal([1.8, 4, 6], result.h_energy_delta_minus)
                np.testing.assert_array_equal([6, 12], result.he4_energy_delta_plus)
                np.testing.assert_array_equal([4, 6], result.he4_energy_delta_minus)
                np.testing.assert_array_equal([6, 12], result.cno_energy_delta_plus)
                np.testing.assert_array_equal([4, 6], result.cno_energy_delta_minus)
                np.testing.assert_array_equal([6, 12], result.nemgsi_energy_delta_plus)
                np.testing.assert_array_equal([4, 6], result.nemgsi_energy_delta_minus)
                np.testing.assert_array_equal([12], result.fe_energy_delta_plus)
                np.testing.assert_array_equal([4], result.fe_energy_delta_minus)

    def test_read_l2_hit_data_handles_fill_values(self):
        path = get_test_data_folder() / 'hit' / 'l2_hit_data_with_fill_values.cdf'
        result = read_l2_hit_data(path)

        with CDF(str(path)) as cdf:
            np.testing.assert_array_equal(result.h, np.full_like(cdf["h"], np.nan))
            np.testing.assert_array_equal(result.he4, np.full_like(cdf["he4"], np.nan))
            np.testing.assert_array_equal(result.cno, np.full_like(cdf["cno"], np.nan))
            np.testing.assert_array_equal(result.nemgsi, np.full_like(cdf["nemgsi"], np.nan))
            np.testing.assert_array_equal(result.fe, np.full_like(cdf["fe"], np.nan))

            np.testing.assert_array_equal(result.delta_plus_h, np.full_like(cdf["delta_plus_h"], np.nan))
            np.testing.assert_array_equal(result.delta_minus_h, np.full_like(cdf["delta_minus_h"], np.nan))
            np.testing.assert_array_equal(result.delta_plus_he4, np.full_like(cdf["delta_plus_he4"], np.nan))
            np.testing.assert_array_equal(result.delta_minus_he4, np.full_like(cdf["delta_minus_he4"], np.nan))
            np.testing.assert_array_equal(result.delta_plus_cno, np.full_like(cdf["delta_plus_cno"], np.nan))
            np.testing.assert_array_equal(result.delta_minus_cno, np.full_like(cdf["delta_minus_cno"], np.nan))
            np.testing.assert_array_equal(result.delta_plus_nemgsi, np.full_like(cdf["delta_plus_nemgsi"], np.nan))
            np.testing.assert_array_equal(result.delta_minus_nemgsi, np.full_like(cdf["delta_minus_nemgsi"], np.nan))
            np.testing.assert_array_equal(result.delta_plus_fe, np.full_like(cdf["delta_plus_fe"], np.nan))
            np.testing.assert_array_equal(result.delta_minus_fe, np.full_like(cdf["delta_minus_fe"], np.nan))
