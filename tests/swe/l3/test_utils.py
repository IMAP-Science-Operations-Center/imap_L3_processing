import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import spiceypy
from spacepy.pycdf import CDF

from imap_l3_processing.swe.l3.models import SweL2Data, SwapiL3aProtonData, SweL1bData
from imap_l3_processing.swe.l3.utils import read_swe_config, read_l2_swe_data, read_l3a_swapi_proton_data, \
    read_l1b_swe_data, compute_epoch_delta_in_ns
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    def test_read_swe_config(self):
        result = read_swe_config(get_test_data_path('swe/example_swe_config.json'))
        self.assertEqual([
            0.0697327,
            0.138312,
            0.175125,
            0.181759,
            0.204686,
            0.151448,
            0.0781351
        ], result["geometric_fractions"])
        self.assertEqual(20, len(result["pitch_angle_bins"]))
        self.assertEqual(13.5, result["pitch_angle_bins"][1])
        self.assertEqual(20, len(result["pitch_angle_delta"]))
        self.assertEqual(4.5, result["pitch_angle_delta"][1])
        self.assertEqual(24, len(result["energy_bins"]))
        self.assertEqual(2.66, result["energy_bins"][0])
        self.assertEqual(24, len(result["energy_delta_minus"]))
        self.assertEqual(4.06132668e-01, result["energy_delta_minus"][0])
        self.assertEqual(24, len(result["energy_delta_plus"]))
        self.assertEqual(4.79315212e-01, result["energy_delta_plus"][0])
        self.assertEqual(0.6, result["energy_bin_low_multiplier"])
        self.assertEqual(1.4, result["energy_bin_high_multiplier"])

    def test_read_l1b_swe_data(self):
        result: SweL1bData = read_l1b_swe_data(
            get_test_data_path('swe/imap_swe_l1b_sci_20240510_v002.cdf'))

        self.assertEqual((6,), result.epoch.shape)
        self.assertEqual(datetime(2010, 1, 1, 0, 0), result.epoch[0])
        self.assertEqual(datetime(2010, 1, 1, 0, 1), result.epoch[1])

        self.assertEqual((6, 24, 30, 7), result.count_rates.shape)
        self.assertEqual(0, result.count_rates[0, 0, 0, 0])
        self.assertEqual(589631.7192194695, result.count_rates[2, 0, 0, 0])

        self.assertEqual((6, 4), result.settle_duration.shape)
        self.assertEqual(3333, result.settle_duration[0, 0])

    def test_read_l2_swe_data(self):
        result: SweL2Data = read_l2_swe_data(
            get_test_data_path('swe/imap_swe_l2_sci-with-fill-values_20250101_v002.cdf'))

        self.assertEqual(result.epoch[0], datetime(2025, 1, 1))
        self.assertEqual(len(result.epoch), 6)

        self.assertEqual(result.flux[3][13][12][6], 324526.5306847633)
        self.assertTrue(np.isnan(result.flux[1][2][3][4]))
        self.assertEqual(0, np.count_nonzero(result.flux == -1e31))
        self.assertEqual(result.flux.shape, (6, 24, 30, 7))

        self.assertEqual(result.inst_el[0], -63)
        self.assertEqual(len(result.inst_el), 7)

        self.assertEqual(result.energy[0], 2.66)
        self.assertEqual(len(result.energy), 24)

        self.assertEqual(result.inst_az_spin_sector.shape, (6, 24, 30))
        self.assertEqual(result.inst_az_spin_sector[0][0][0], 153.97713661193848)
        self.assertTrue(np.isnan(result.inst_az_spin_sector[4][5][6]))
        spiceypy.sct2e(-43, 0.1)
        self.assertEqual(result.phase_space_density.shape, (6, 24, 30, 7))
        self.assertEqual(result.phase_space_density[3][11][8][4], 1.8811969552023866e-26)
        self.assertTrue(np.isnan(result.phase_space_density[5][4][3][2]))

        self.assertEqual(result.acquisition_time.shape, (6, 24, 30))
        self.assertEqual(result.acquisition_time[0][0][0], np.datetime64('2024-12-31T23:59:30.099713922'))
        self.assertTrue(np.isnat(result.acquisition_time[1][2][3]))

        self.assertEqual(result.acquisition_duration.shape, (6, 24, 30))
        self.assertEqual(result.acquisition_duration[0][0][0], 80000)

    def test_read_l3a_swapi_proton_data(self):
        result = read_l3a_swapi_proton_data(get_test_data_path('swe/imap_swapi_l3a_proton-sw_20250101_v001.cdf'))
        self.assertIsInstance(result, SwapiL3aProtonData)
        self.assertEqual(10, len(result.epoch))
        self.assertEqual(10, len(result.epoch_delta))
        self.assertEqual(10, len(result.proton_sw_speed))
        self.assertEqual(10, len(result.proton_sw_clock_angle))
        self.assertEqual(10, len(result.proton_sw_deflection_angle))
        self.assertEqual(datetime(2025, 1, 1), result.epoch[0])
        self.assertEqual(timedelta(seconds=30), result.epoch_delta[0])
        self.assertEqual(498.4245091006667, result.proton_sw_speed[0])
        self.assertEqual(82.53712019721974, result.proton_sw_clock_angle[0])
        self.assertEqual(5.553957246800335e-06, result.proton_sw_deflection_angle[0])

    def test_read_l3a_swapi_proton_data_with_fill_values(self):
        with tempfile.TemporaryDirectory() as tempdir:
            cdf_with_fill_path = Path(tempdir, 'swe_file_with_fill.cdf')
            shutil.copyfile(get_test_data_path('swe/imap_swapi_l3a_proton-sw_20250101_v001.cdf'), cdf_with_fill_path)
            with CDF(str(cdf_with_fill_path), readonly=False) as cdf:
                proton_sw_speed_fill_value = cdf['proton_sw_speed'].attrs['FILLVAL']
                proton_sw_clock_angle_fill_value = cdf['proton_sw_clock_angle'].attrs['FILLVAL']
                proton_sw_deflection_angle_fill_value = cdf['proton_sw_deflection_angle'].attrs['FILLVAL']
                cdf['proton_sw_speed'][0] = proton_sw_speed_fill_value
                cdf['proton_sw_clock_angle'][0] = proton_sw_clock_angle_fill_value
                cdf['proton_sw_deflection_angle'][0] = proton_sw_deflection_angle_fill_value

            swapi_l3a_data = read_l3a_swapi_proton_data(cdf_with_fill_path)
            self.assertTrue(np.isnan(swapi_l3a_data.proton_sw_speed[0]))
            self.assertTrue(np.isnan(swapi_l3a_data.proton_sw_clock_angle[0]))
            self.assertTrue(np.isnan(swapi_l3a_data.proton_sw_deflection_angle[0]))

    def test_compute_epoch_delta_in_ns(self):
        acq_duration_microseconds = np.full((4, 24, 30), 80_000)
        settle_duration_microseconds = np.full((4, 4), 10000 / 3)
        result = compute_epoch_delta_in_ns(acq_duration_microseconds, settle_duration_microseconds)
        expected = np.full(4, 30 * 1e9)
        np.testing.assert_array_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
