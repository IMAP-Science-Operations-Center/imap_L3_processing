import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from imap_processing.ena_maps.utils.spatial_utils import build_solid_angle_map
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS, ONE_SECOND_IN_NANOSECONDS, SECONDS_PER_DAY
from imap_l3_processing.hi.l3.utils import read_hi_l2_data, read_hi_l1c_data, read_glows_l3e_data, CGCorrection, \
    SurvivalCorrection, PixelSize, Sensor, MapDescriptorParts, parse_map_descriptor, SpinPhase, Duration, MapQuantity
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

            ena_intensity = rng.random((1, 9, 90, 45))
            energy = rng.random(9)
            energy_delta_plus = rng.random(9)
            energy_delta_minus = rng.random(9)
            energy_label = energy.astype(str)
            ena_intensity_stat_unc = rng.random(ena_intensity.shape)
            ena_intensity_sys_err = rng.random(ena_intensity.shape)

            epoch = np.array([datetime.now()])
            epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
            exposure = np.full(ena_intensity.shape[:-1], 1.0)
            lat = np.arange(-88.0, 92.0, 4.0)
            lat_delta = np.full(lat.shape, 2.0)
            lat_label = [f"{x} deg" for x in lat]
            lon = np.arange(0.0, 360.0, 4.0)
            lon_delta = np.full(lon.shape, 2.0)
            lon_label = [f"{x} deg" for x in lon]

            obs_date = np.full(ena_intensity.shape, datetime.now())
            obs_date_range = np.full(ena_intensity.shape, ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY * 2)
            solid_angle = build_solid_angle_map(4)
            solid_angle = solid_angle[np.newaxis, ...]

            cdf.new("epoch", epoch)
            cdf.new("energy", energy, recVary=False)
            cdf.new("latitude", lat, recVary=False)
            cdf.new("latitude_delta", lat_delta, recVary=False)
            cdf.new("latitude_label", lat_label, recVary=False)
            cdf.new("longitude", lon, recVary=False)
            cdf.new("longitude_delta", lon_delta, recVary=False)
            cdf.new("longitude_label", lon_label, recVary=False)
            cdf.new("ena_intensity", ena_intensity, recVary=True)
            cdf.new("ena_intensity_stat_unc", ena_intensity_stat_unc, recVary=True)
            cdf.new("ena_intensity_sys_err", ena_intensity_sys_err, recVary=True)
            cdf.new("exposure_factor", exposure, recVary=True)
            cdf.new("obs_date", obs_date, recVary=True)
            cdf.new("obs_date_range", obs_date_range, recVary=True)
            cdf.new("solid_angle", solid_angle, recVary=True)
            cdf.new("epoch_delta", epoch_delta, recVary=True)
            cdf.new("energy_delta_plus", energy_delta_plus, recVary=False)
            cdf.new("energy_delta_minus", energy_delta_minus, recVary=False)
            cdf.new("energy_label", energy_label, recVary=False)

            for var in cdf:
                cdf[var].attrs['FILLVAL'] = 1000000

        for path in [pathname, Path(pathname)]:
            with self.subTest(path=path):
                result = read_hi_l2_data(path)

                np.testing.assert_array_equal(epoch, result.epoch)
                np.testing.assert_array_equal(epoch_delta, result.epoch_delta)
                np.testing.assert_array_equal(energy, result.energy)
                np.testing.assert_array_equal(energy_delta_plus, result.energy_delta_plus)
                np.testing.assert_array_equal(energy_delta_minus, result.energy_delta_minus)
                np.testing.assert_array_equal(energy_label, result.energy_label)
                np.testing.assert_array_equal(lat, result.latitude)
                np.testing.assert_array_equal(lat_delta, result.latitude_delta)
                np.testing.assert_array_equal(lat_label, result.latitude_label)
                np.testing.assert_array_equal(lon, result.longitude)
                np.testing.assert_array_equal(lon_delta, result.longitude_delta)
                np.testing.assert_array_equal(lon_label, result.longitude_label)
                np.testing.assert_array_equal(ena_intensity, result.ena_intensity)
                np.testing.assert_array_equal(ena_intensity_stat_unc, result.ena_intensity_stat_unc)
                np.testing.assert_array_equal(ena_intensity_sys_err, result.ena_intensity_sys_err)
                np.testing.assert_array_equal(exposure, result.exposure_factor)
                np.testing.assert_array_equal(obs_date, result.obs_date)
                np.testing.assert_array_equal(obs_date_range, result.obs_date_range)
                np.testing.assert_array_equal(solid_angle, result.solid_angle)

    def test_fill_values_create_nan_data(self):
        path = get_test_data_folder() / 'hi' / 'fake_l2_maps' / 'l2_map_with_fill_values.cdf'
        result = read_hi_l2_data(path)

        with CDF(str(path)) as cdf:
            np.testing.assert_array_equal(result.epoch, cdf["epoch"], )

            self.assertTrue(np.all(result.epoch_delta.mask))
            self.assertTrue(np.all(result.obs_date.mask))
            self.assertTrue(np.all(result.obs_date_range.mask))

            np.testing.assert_array_equal(result.energy, np.full_like(cdf["energy"], np.nan))
            np.testing.assert_array_equal(result.energy_delta_plus, np.full_like(cdf["energy_delta_plus"], np.nan))
            np.testing.assert_array_equal(result.energy_delta_minus, np.full_like(cdf["energy_delta_minus"], np.nan))
            np.testing.assert_array_equal(result.latitude, np.full_like(cdf["latitude"], np.nan))
            np.testing.assert_array_equal(result.latitude_delta, np.full_like(cdf["latitude_delta"], np.nan))
            np.testing.assert_array_equal(result.longitude, np.full_like(cdf["longitude"], np.nan))
            np.testing.assert_array_equal(result.longitude_delta, np.full_like(cdf["longitude_delta"], np.nan))
            np.testing.assert_array_equal(result.ena_intensity, np.full_like(cdf["ena_intensity"], np.nan))
            np.testing.assert_array_equal(result.ena_intensity_stat_unc,
                                          np.full_like(cdf["ena_intensity_stat_unc"], np.nan))
            np.testing.assert_array_equal(result.ena_intensity_sys_err,
                                          np.full_like(cdf["ena_intensity_sys_err"], np.nan))
            np.testing.assert_array_equal(result.exposure_factor, np.full_like(cdf["exposure_factor"], np.nan))
            np.testing.assert_array_equal(result.solid_angle, np.full_like(cdf["solid_angle"], np.nan))

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

    def test_parse_map_descriptor(self):
        cg = CGCorrection.CGCorrected
        no_cg = CGCorrection.NotCGCorrected
        sp = SurvivalCorrection.SurvivalCorrected
        no_sp = SurvivalCorrection.NotSurvivalCorrected
        test_cases = [
            ("h45-sf-sp-anti-hae-4deg-3mo", MapDescriptorParts(Sensor.Hi45, no_cg, sp, SpinPhase.AntiRamOnly,
                                                               PixelSize.FourDegrees, Duration.ThreeMonths,
                                                               MapQuantity.Intensity)),
            ("h90-hf-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.RamOnly,
                                                           PixelSize.SixDegrees, Duration.OneYear,
                                                           MapQuantity.Intensity)),
            ("h90-hf-hae-6deg-6mo", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.FullSpin,
                                                       PixelSize.SixDegrees, Duration.SixMonths,
                                                       MapQuantity.Intensity)),
            ("h45-hf-sp-hae-4deg-6mo-spectral", MapDescriptorParts(Sensor.Hi45, cg, sp, SpinPhase.FullSpin,
                                                                   PixelSize.FourDegrees, Duration.SixMonths,
                                                                   MapQuantity.SpectralIndex)),
            ("h-hf-hae-6deg-6mo", MapDescriptorParts(Sensor.Combined, cg, no_sp, SpinPhase.FullSpin,
                                                     PixelSize.SixDegrees, Duration.SixMonths,
                                                     MapQuantity.Intensity)),
            ("not-valid-at-all", None),
        ]

        for descriptor, expected in test_cases:
            with self.subTest(descriptor):
                descriptor_parts = parse_map_descriptor(descriptor)
                self.assertEqual(expected, descriptor_parts)
