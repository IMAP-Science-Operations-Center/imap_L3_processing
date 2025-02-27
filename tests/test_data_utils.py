import unittest
from datetime import datetime, timedelta

import numpy as np

from imap_processing.data_utils import rebin, find_closest_neighbor


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

    def test_rebin_with_missing_data_fills_with_nan(self):
        hit_data_epoch = np.array([datetime(2020, 4, 4, 0, 5),
                                   datetime(2020, 4, 4, 0, 15),
                                   datetime(2020, 4, 4, 0, 25)])
        hit_data_delta = np.array([timedelta(seconds=300), timedelta(seconds=300), timedelta(seconds=300)])
        mag_data = np.empty((0, 3))
        mag_epoch = []
        actual_average = rebin(from_epoch=mag_epoch, from_data=mag_data,
                               to_epoch=hit_data_epoch,
                               to_epoch_delta=hit_data_delta)

        expected_average = [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ]

        np.testing.assert_array_equal(expected_average, actual_average)

    def test_rebin_empty_bins_with_nan(self):
        hit_data_epoch = np.array([datetime(2020, 4, 4, 0, 5),
                                   datetime(2020, 4, 4, 0, 15),
                                   datetime(2020, 4, 4, 0, 25)])
        hit_data_delta = np.array([timedelta(seconds=300), timedelta(seconds=300), timedelta(seconds=300)])
        mag_data = np.array([[0, 0, 1], [0, 1, 0]])
        mag_epoch = [datetime(2020, 4, 4, 0, 1),
                     datetime(2020, 4, 4, 0, 3)]
        actual_average = rebin(from_epoch=mag_epoch, from_data=mag_data,
                               to_epoch=hit_data_epoch,
                               to_epoch_delta=hit_data_delta)
        expected_average = [
            [0, 0.5, 0.5],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ]
        np.testing.assert_array_equal(actual_average, expected_average)

    def test_gap_in_to_epoch(self):
        hit_data_epoch = np.array([datetime(2020, 4, 4, 0, 5),
                                   datetime(2020, 4, 4, 0, 15),
                                   datetime(2020, 4, 4, 0, 30)])
        hit_data_delta = np.array([timedelta(seconds=300), timedelta(seconds=300), timedelta(seconds=300)])
        mag_data = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        mag_epoch = [datetime(2020, 4, 4, 0, 1),
                     datetime(2020, 4, 4, 0, 7),
                     datetime(2020, 4, 4, 0, 17),
                     datetime(2020, 4, 4, 0, 21)
                     ]
        actual_average = rebin(from_epoch=mag_epoch, from_data=mag_data,
                               to_epoch=hit_data_epoch,
                               to_epoch_delta=hit_data_delta)
        expected_average = [
            [0, 0.5, 0.5],
            [0, 0, 1],
            [np.nan, np.nan, np.nan]
        ]
        np.testing.assert_array_equal(actual_average, expected_average)

    def test_find_closest_neighbor(self):
        test_cases = [
            ("matching cadence", [datetime(2020, 4, 4),
                                  datetime(2020, 4, 5),
                                  datetime(2020, 4, 6),
                                  datetime(2020, 4, 7)],
             [[0, 0, 1], [0, 2, 0], [0, 0, 3], [4, 0, 0]]),
            ("to slower cadence", [datetime(2020, 4, 4),
                                   datetime(2020, 4, 6),
                                   datetime(2020, 4, 8)],
             [[0, 0, 1], [0, 0, 3], [4, 0, 0]]),
            ("to faster cadence", [datetime(2020, 4, 4, hour=12),
                                   datetime(2020, 4, 5),
                                   datetime(2020, 4, 5, hour=12)],
             [[0, 0, 1], [0, 2, 0], [0, 2, 0]]),
            ("outside range", [datetime(2020, 4, 2, hour=23),
                               datetime(2020, 4, 5),
                               datetime(2020, 4, 8, hour=1)],
             [[np.nan, np.nan, np.nan], [0, 2, 0], [np.nan, np.nan, np.nan]]),
        ]

        for case, to_dates, expected_values in test_cases:
            with self.subTest(case):
                to_data_epoch = np.array(to_dates)
                from_data = np.array([[0, 0, 1], [0, 2, 0], [0, 0, 3], [4, 0, 0]])
                from_date_epoch = np.array([datetime(2020, 4, 4),
                                            datetime(2020, 4, 5),
                                            datetime(2020, 4, 6),
                                            datetime(2020, 4, 7)
                                            ])

                actual_neighbor_values = find_closest_neighbor(from_date_epoch, from_data, to_data_epoch,
                                                               timedelta(days=1))

                np.testing.assert_array_equal(actual_neighbor_values, expected_values)
