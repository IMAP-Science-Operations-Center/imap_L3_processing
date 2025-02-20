import unittest
from datetime import datetime, timedelta

import numpy as np

from imap_processing.data_utils import rebin


class TestDataUtils(unittest.TestCase):
    def test_1d_rebin(self):
        hit_data_epoch = np.array([datetime(2020, 4, 4, 0, 5), datetime(2020, 4, 4, 0, 15)])
        hit_data_delta = np.array([timedelta(seconds=300), timedelta(seconds=300)])
        extra_data_at_beginning = [0, 0.5]
        extra_dates_at_beginning = [datetime(2020, 4, 3, 0, 23), datetime(2020, 4, 3, 0, 23, 30)]
        extra_dates_at_end = [datetime(2020, 4, 4, 1, 0), datetime(2020, 4, 4, 1, 23, 30)]
        extra_data_at_end = [1, 1.5]
        input_data = np.array(extra_data_at_beginning +
                              [0, 1, 0, 1, 0, 1, 0, 0] + extra_data_at_end)
        mag_epoch = np.array(extra_dates_at_beginning + [datetime(2020, 4, 4, 0, 1),
                                                         datetime(2020, 4, 4, 0, 3),
                                                         datetime(2020, 4, 4, 0, 6),
                                                         datetime(2020, 4, 4, 0, 9, 55),
                                                         datetime(2020, 4, 4, 0, 12, 45),
                                                         datetime(2020, 4, 4, 0, 13),
                                                         datetime(2020, 4, 4, 0, 15),
                                                         datetime(2020, 4, 4, 0, 17)] + extra_dates_at_end)
        expected_average = np.array([1 / 2, 1 / 4])

        actual_average = rebin(from_epoch=mag_epoch, from_data=input_data,
                               to_epoch=hit_data_epoch,
                               to_epoch_delta=hit_data_delta)
        np.testing.assert_array_equal(actual_average, expected_average)

    def test_rebin_with_extra_data_at_beginning_and_end(self):
        hit_data_epoch = np.array([datetime(2020, 4, 4, 0, 5), datetime(2020, 4, 4, 0, 15)])
        hit_data_delta = np.array([timedelta(seconds=300), timedelta(seconds=300)])
        extra_data_at_beginning = [[0.5, 0, 1], [0, 0.5, 1]]
        extra_dates_at_beginning = [datetime(2020, 4, 3, 0, 23), datetime(2020, 4, 3, 0, 23, 30)]
        extra_dates_at_end = [datetime(2020, 4, 4, 1, 0), datetime(2020, 4, 4, 1, 23, 30)]
        extra_data_at_end = [[0.5, 1, 1], [1, 0.5, 0.5]]
        mag_data = np.array(extra_data_at_beginning +
                            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                             [1, 0, 0]] + extra_data_at_end)
        mag_epoch = np.array(extra_dates_at_beginning + [datetime(2020, 4, 4, 0, 1),
                                                         datetime(2020, 4, 4, 0, 3), datetime(2020, 4, 4, 0, 6),
                                                         datetime(2020, 4, 4, 0, 9, 55),
                                                         datetime(2020, 4, 4, 0, 12, 45),
                                                         datetime(2020, 4, 4, 0, 13),
                                                         datetime(2020, 4, 4, 0, 15),
                                                         datetime(2020, 4, 4, 0, 17)] + extra_dates_at_end)
        expected_average = np.array([[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1 / 4, 1 / 4]])

        actual_average = rebin(from_epoch=mag_epoch, from_data=mag_data,
                               to_epoch=hit_data_epoch,
                               to_epoch_delta=hit_data_delta)
        np.testing.assert_array_equal(actual_average, expected_average)

    def test_rebin_handles_missing_data(self):
        hit_data_epoch = np.array([datetime(2020, 4, 4, 0, 5),
                                   datetime(2020, 4, 4, 0, 15),
                                   datetime(2020, 4, 4, 0, 25)])
        hit_data_delta = np.array([timedelta(seconds=300), timedelta(seconds=300), timedelta(seconds=300)])
        mag_data = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        mag_epoch = [datetime(2020, 4, 4, 0, 1),
                     datetime(2020, 4, 4, 0, 3),
                     datetime(2020, 4, 4, 0, 22),
                     datetime(2020, 4, 4, 0, 27)]
        expected_average = np.array([[0, 1 / 2, 1 / 2], [np.nan, np.nan, np.nan], [1 / 2, 1 / 2, 0]])

        actual_average = rebin(from_epoch=mag_epoch, from_data=mag_data,
                               to_epoch=hit_data_epoch,
                               to_epoch_delta=hit_data_delta)
        np.testing.assert_array_equal(actual_average, expected_average)
