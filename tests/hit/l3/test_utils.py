import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.hit.l3.utils import read_l2_hit_data


class TestUtils(TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_read_l2_hit_data(self):
        rng = np.random.default_rng()
        pathname = os.path.join(self.temp_dir.name, 'test_cdf.cdf')
        expected_azimuth_data = [12., 36., 60., 84., 108., 132., 156., 180., 204., 228., 252.,
                                 276., 300., 324., 348.]
        expected_zenith_data = [11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75]
        with CDF(pathname, '') as cdf:
            cdf.col_major(True)

            start_time = datetime(2010, 1, 1, 0, 5)

            epoch_data = np.array([start_time, start_time + timedelta(minutes=5)])

            hydrogen_data = rng.random((1, 3, 15, 8))
            helium_data = rng.random((1, 2, 15, 8))
            cno_data = rng.random((1, 2, 15, 8))
            nemgsi_data = rng.random((1, 2, 15, 8))
            iron_data = rng.random((1, 1, 15, 8))
            cdf["h_macropixel_intensity"] = hydrogen_data
            cdf["he4_macropixel_intensity"] = helium_data
            cdf["cno_macropixel_intensity"] = cno_data
            cdf["nemgsi_macropixel_intensity"] = nemgsi_data
            cdf["fe_macropixel_intensity"] = iron_data

            cdf["epoch"] = epoch_data

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
            cdf["h_stat_uncert_plus"] = hydrogen_delta
            cdf["h_stat_uncert_minus"] = hydrogen_delta
            cdf["he4_stat_uncert_plus"] = helium_delta
            cdf["he4_stat_uncert_minus"] = helium_delta
            cdf["cno_stat_uncert_plus"] = cno_delta
            cdf["cno_stat_uncert_minus"] = cno_delta
            cdf["nemgsi_stat_uncert_plus"] = nemgsi_delta
            cdf["nemgsi_stat_uncert_minus"] = nemgsi_delta
            cdf["fe_stat_uncert_plus"] = iron_delta
            cdf["fe_stat_uncert_minus"] = iron_delta

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

            cdf.new("azimuth", expected_azimuth_data, recVary=False)
            cdf.new("zenith", expected_zenith_data, recVary=False)

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
                np.testing.assert_array_equal([timedelta(minutes=2.5), timedelta(minutes=2.5)], result.epoch_delta)

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

                np.testing.assert_array_equal(expected_azimuth_data, result.azimuth)
                np.testing.assert_array_equal(expected_zenith_data, result.zenith)

    def test_read_l2_hit_data_handles_fill_values(self):
        pathname = os.path.join(self.temp_dir.name, 'test_cdf_with_fill_values.cdf')

        fill_val = -1e31

        with CDF(pathname, '') as cdf:
            cdf.col_major(True)

            start_time = datetime(2010, 1, 1, 0, 5)

            epoch_data = np.array([start_time, start_time + timedelta(minutes=5)])

            hydrogen_data = np.full((1, 3, 15, 8), fill_value=fill_val)
            helium_data = np.full((1, 2, 15, 8), fill_value=fill_val)
            cno_data = np.full((1, 2, 15, 8), fill_value=fill_val)
            nemgsi_data = np.full((1, 2, 15, 8), fill_value=fill_val)
            iron_data = np.full((1, 1, 15, 8), fill_value=fill_val)
            cdf["h_macropixel_intensity"] = hydrogen_data
            cdf["he4_macropixel_intensity"] = helium_data
            cdf["cno_macropixel_intensity"] = cno_data
            cdf["nemgsi_macropixel_intensity"] = nemgsi_data
            cdf["fe_macropixel_intensity"] = iron_data

            cdf["epoch"] = epoch_data

            cdf.new("h_energy_mean", np.arange(3), recVary=False)
            cdf.new("he4_energy_mean", np.arange(2), recVary=False)
            cdf.new("cno_energy_mean", np.arange(2), recVary=False)
            cdf.new("nemgsi_energy_mean", np.arange(2), recVary=False)
            cdf.new("fe_energy_mean", np.arange(1), recVary=False)

            cdf["h_stat_uncert_plus"] = hydrogen_data
            cdf["h_stat_uncert_minus"] = hydrogen_data
            cdf["he4_stat_uncert_plus"] = helium_data
            cdf["he4_stat_uncert_minus"] = helium_data
            cdf["cno_stat_uncert_plus"] = cno_data
            cdf["cno_stat_uncert_minus"] = cno_data
            cdf["nemgsi_stat_uncert_plus"] = nemgsi_data
            cdf["nemgsi_stat_uncert_minus"] = nemgsi_data
            cdf["fe_stat_uncert_plus"] = iron_data
            cdf["fe_stat_uncert_minus"] = iron_data

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

            cdf.new("azimuth", [12., 36., 60., 84., 108., 132., 156., 180., 204., 228., 252.,
                                276., 300., 324., 348.], recVary=False)
            cdf.new("zenith", [11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75], recVary=False)

            for var in cdf:
                cdf[var].attrs["FILLVAL"] = fill_val

        for path in [pathname, Path(pathname)]:
            with self.subTest(path):
                result = read_l2_hit_data(path)

                np.testing.assert_array_equal(result.h, np.full_like(hydrogen_data, np.nan))
                np.testing.assert_array_equal(result.he4, np.full_like(helium_data, np.nan))
                np.testing.assert_array_equal(result.cno, np.full_like(cno_data, np.nan))
                np.testing.assert_array_equal(result.nemgsi, np.full_like(nemgsi_data, np.nan))
                np.testing.assert_array_equal(result.fe, np.full_like(iron_data, np.nan))

                np.testing.assert_array_equal(result.delta_plus_h, np.full_like(hydrogen_data, np.nan))
                np.testing.assert_array_equal(result.delta_minus_h, np.full_like(hydrogen_data, np.nan))
                np.testing.assert_array_equal(result.delta_plus_he4, np.full_like(helium_data, np.nan))
                np.testing.assert_array_equal(result.delta_minus_he4, np.full_like(helium_data, np.nan))
                np.testing.assert_array_equal(result.delta_plus_cno, np.full_like(cno_data, np.nan))
                np.testing.assert_array_equal(result.delta_minus_cno, np.full_like(cno_data, np.nan))
                np.testing.assert_array_equal(result.delta_plus_nemgsi, np.full_like(nemgsi_data, np.nan))
                np.testing.assert_array_equal(result.delta_minus_nemgsi, np.full_like(nemgsi_data, np.nan))
                np.testing.assert_array_equal(result.delta_plus_fe, np.full_like(iron_data, np.nan))
                np.testing.assert_array_equal(result.delta_minus_fe, np.full_like(iron_data, np.nan))
