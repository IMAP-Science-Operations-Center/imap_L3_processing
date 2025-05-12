import os
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.hi.l3.utils import read_hi_l1c_data, read_glows_l3e_data, ReferenceFrame, \
    SurvivalCorrection, PixelSize, Sensor, MapDescriptorParts, parse_map_descriptor, SpinPhase, Duration, MapQuantity
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
        cg = ReferenceFrame.Heliospheric
        no_cg = ReferenceFrame.Spacecraft
        sp = SurvivalCorrection.SurvivalCorrected
        no_sp = SurvivalCorrection.NotSurvivalCorrected

        test_cases = [
            ("h45-ena-h-hf-sp-ram-hae-4deg-3mo", MapDescriptorParts(sensor=Sensor.Hi45, quantity=MapQuantity.Intensity,
                                                                    survival_correction=sp, reference_frame=cg,
                                                                    spin_phase=SpinPhase.RamOnly,
                                                                    duration=Duration.ThreeMonths,
                                                                    grid=PixelSize.FourDegrees)),
            ("h45-ena-h-sf-sp-anti-hae-4deg-3mo", MapDescriptorParts(Sensor.Hi45, no_cg, sp, SpinPhase.AntiRamOnly,
                                                                     PixelSize.FourDegrees, Duration.ThreeMonths,
                                                                     MapQuantity.Intensity)),
            ("h90-ena-h-hf-nsp-ram-hae-6deg-1yr", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.RamOnly,
                                                                     PixelSize.SixDegrees, Duration.OneYear,
                                                                     MapQuantity.Intensity)),
            ("h90-ena-h-hf-nsp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.Hi90, cg, no_sp, SpinPhase.FullSpin,
                                                                      PixelSize.SixDegrees, Duration.SixMonths,
                                                                      MapQuantity.Intensity)),
            ("h45-spx-h-hf-sp-full-hae-4deg-6mo", MapDescriptorParts(Sensor.Hi45, cg, sp, SpinPhase.FullSpin,
                                                                     PixelSize.FourDegrees, Duration.SixMonths,
                                                                     MapQuantity.SpectralIndex)),
            ("hic-ena-h-hf-nsp-full-hae-6deg-6mo", MapDescriptorParts(Sensor.Combined, cg, no_sp, SpinPhase.FullSpin,
                                                                      PixelSize.SixDegrees, Duration.SixMonths,
                                                                      MapQuantity.Intensity)),
            ("not-valid-at-all", None),
            ("invalid_prefix-hic-ena-h-hf-nsp-full-hae-6deg-6mo", None),
            ("hic-ena-h-hf-nsp-full-hae-6deg-6mo-invalid-suffix", None),
        ]

        for descriptor, expected in test_cases:
            with self.subTest(descriptor):
                descriptor_parts = parse_map_descriptor(descriptor)
                self.assertEqual(expected, descriptor_parts)
