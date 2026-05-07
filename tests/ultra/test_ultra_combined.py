import unittest

import numpy as np

from imap_l3_processing.ultra.science.ultra_combined import map_combined_rectangular_quantities_to_healpix_intensity_map
from tests.maps.test_builders import create_rectangular_intensity_map_data
from tests.ultra.test_ultra_processor import _create_ultra_l2_healpix_data


class TestUltraSurvivalProbability(unittest.TestCase):

    def test_map_combined_rectangular_quantities_to_healpix_intensity_map_data(self):
        healpix_input = _create_ultra_l2_healpix_data()
        rectangular_input = create_rectangular_intensity_map_data()
        healpix_input_intensity_map_data = healpix_input.intensity_map_data
        rectangular_input_intensity_map_data = rectangular_input.intensity_map_data

        result_map = map_combined_rectangular_quantities_to_healpix_intensity_map(healpix_input, rectangular_input)
        result_intensity_data = result_map.intensity_map_data

        np.testing.assert_array_equal(result_intensity_data.ena_intensity, healpix_input_intensity_map_data.ena_intensity)
        np.testing.assert_array_equal(result_intensity_data.ena_intensity_stat_uncert, healpix_input_intensity_map_data.ena_intensity_stat_uncert)
        np.testing.assert_array_equal(result_intensity_data.ena_intensity_sys_err, healpix_input_intensity_map_data.ena_intensity_sys_err)

        np.testing.assert_array_equal(result_intensity_data.epoch, rectangular_input_intensity_map_data.epoch)
        np.testing.assert_array_equal(result_intensity_data.epoch_delta, rectangular_input_intensity_map_data.epoch_delta)
        np.testing.assert_array_equal(result_intensity_data.energy, rectangular_input_intensity_map_data.energy)
        np.testing.assert_array_equal(result_intensity_data.energy_delta_plus, rectangular_input_intensity_map_data.energy_delta_plus)
        np.testing.assert_array_equal(result_intensity_data.energy_delta_minus, rectangular_input_intensity_map_data.energy_delta_minus)
        np.testing.assert_array_equal(result_intensity_data.energy_label, rectangular_input_intensity_map_data.energy_label)
        np.testing.assert_array_equal(result_intensity_data.latitude, rectangular_input_intensity_map_data.latitude)
        np.testing.assert_array_equal(result_intensity_data.longitude, rectangular_input_intensity_map_data.longitude)
        np.testing.assert_array_equal(result_intensity_data.obs_date, rectangular_input_intensity_map_data.obs_date)
        np.testing.assert_array_equal(result_intensity_data.obs_date_range, rectangular_input_intensity_map_data.obs_date_range)
        np.testing.assert_array_equal(result_intensity_data.solid_angle, rectangular_input_intensity_map_data.solid_angle)
        np.testing.assert_array_equal(result_intensity_data.exposure_factor, rectangular_input_intensity_map_data.exposure_factor)
        self.assertEqual(result_map.coords, healpix_input.coords)