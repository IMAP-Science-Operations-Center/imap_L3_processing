import os
from datetime import datetime
from unittest import TestCase

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.utils import chunk_l2_data, read_l2_swapi_data


class TestUtils(TestCase):
    def tearDown(self) -> None:
        if os.path.exists('temp_cdf.cdf'):
            os.remove('temp_cdf.cdf')

    def test_chunk_l2_data(self):
        epoch = np.array([0, 1, 2, 3])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        data = SwapiL2Data(epoch, energy, coincidence_count_rate, coincidence_count_rate_uncertainty)
        chunks = list(chunk_l2_data(data, 2))

        expected_count_rate_chunk_1 = np.array([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
        expected_count_rate_uncertainty_chunk_1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        first_chunk = chunks[0]
        np.testing.assert_array_equal(first_chunk.epoch, np.array([0, 1]))
        np.testing.assert_array_equal(energy, first_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_1, first_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_1,
                                      first_chunk.coincidence_count_rate_uncertainty)

        expected_count_rate_chunk_2 = np.array([[14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        expected_count_rate_uncertainty_chunk_2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        second_chunk = chunks[1]
        np.testing.assert_array_equal(np.array([2, 3]), second_chunk.epoch)
        np.testing.assert_array_equal(energy, second_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_2, second_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_2,
                                      second_chunk.coincidence_count_rate_uncertainty)

    def test_reading_l2_data_into_model(self):
        temp_cdf = CDF('temp_cdf', '')
        temp_cdf["epoch"] = np.array([datetime(2010, 1, 1, 0, 0, 46)])
        temp_cdf["energy"] = np.array([1, 2, 3, 4])
        temp_cdf["swp_coin_rate"] = np.array([5, 6, 7, 8])
        temp_cdf["swp_coin_unc"] = np.array([2, 2, 2, 2, 2, 2, 2, 2])

        temp_cdf.close()

        actual_swapi_l2_data = read_l2_swapi_data(CDF("temp_cdf.cdf"))

        epoch_as_tt2000 = 315576112184000000
        np.testing.assert_array_equal(np.array(epoch_as_tt2000), actual_swapi_l2_data.epoch)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), actual_swapi_l2_data.energy)
        np.testing.assert_array_equal(np.array([5, 6, 7, 8]), actual_swapi_l2_data.coincidence_count_rate)
        np.testing.assert_array_equal(np.array([2, 2, 2, 2, 2, 2, 2, 2]),
                                      actual_swapi_l2_data.coincidence_count_rate_uncertainty)
        os.remove('temp_cdf.cdf')
