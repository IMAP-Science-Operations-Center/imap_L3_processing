import unittest
from datetime import datetime, timedelta

import numpy as np
from imap_data_access.file_validation import Version

from imap_l3_processing.models import MagData, VersionMap


class TestModels(unittest.TestCase):

    def test_calculate_average_mag_data_with_extra_data_at_beginning_and_end(self):
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

        mag_data_model = MagData(mag_epoch, mag_data)
        actual_average = mag_data_model.rebin_to(hit_data_epoch, hit_data_delta)

        np.testing.assert_array_equal(actual_average, expected_average)

    def test_calculate_average_mag_data_handles_missing_mag_data(self):
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

        mag_data_model = MagData(mag_epoch, mag_data)
        actual_average = mag_data_model.rebin_to(epoch=hit_data_epoch,
                                                 epoch_delta=hit_data_delta)
        np.testing.assert_array_equal(actual_average, expected_average)

    def test_version_map(self):
        version_map = VersionMap(
            {
                "sci": Version(1, 5),
                "other": Version(2, 90),
            },
            fallback=Version(1, 99),
        )
        self.assertEqual(version_map.lookup("sci"), Version(1, 5))
        self.assertEqual(version_map.lookup("other"), Version(2, 90))
        self.assertEqual(version_map.lookup("unknown"), Version(1, 99))

        fallback_version_map = VersionMap(
            {},
            fallback=Version(1, 1),
        )
        self.assertEqual(fallback_version_map.lookup("sci"), Version(1, 1))

    def test_version_map_throws_error_if_no_fallback(self):
        version_map = VersionMap(
            {
                "sci": Version(1, 5)
            },
        )

        with self.assertRaises(KeyError):
            version_map.lookup("other")


    def test_version_map_equality(self):
        version_map = VersionMap(
            {
                "sci": Version(1, 5),
                "other": Version(2, 90),
            },
            fallback=Version(1, 99),
        )
        equal_version_map = VersionMap(
            {
                "sci": Version(1, 5),
                "other": Version(2, 90),
            },
            fallback=Version(1, 99),
        )
        different_map = VersionMap(
            {
                "sci": Version(1, 3),
            },
            fallback=Version(1, 99),
        )
        different_fallback = VersionMap(
            {
                "sci": Version(1, 5),
                "other": Version(2, 90),
            },
            fallback=None,
        )
        self.assertEqual(version_map, equal_version_map)
        self.assertNotEqual(version_map, different_map)
        self.assertNotEqual(version_map, different_fallback)

if __name__ == '__main__':
    unittest.main()
