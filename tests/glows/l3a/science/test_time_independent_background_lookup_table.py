import unittest

import numpy as np

from imap_processing.glows.l3a.science.time_independent_background_lookup_table import \
    TimeIndependentBackgroundLookupTable


class TestTimeIndependentBackgroundLookupTable(unittest.TestCase):
    def test_interpolates_to_find_background(self):
        latitudes = np.array([90, 0, -90])
        longitudes = np.array([0, 90, 180, 270])
        bg_map = np.array([
            [4, 4, 4, 4],
            [5, 6, 7, 8],
            [9, 9, 9, 9],
        ])
        table = TimeIndependentBackgroundLookupTable(longitudes, latitudes, bg_map)

        self.assertEqual(5, table.lookup(lat=0, lon=0))
        self.assertEqual(7, table.lookup(lat=-45, lon=0))
        self.assertEqual(4.5, table.lookup(lat=45, lon=0))

        self.assertEqual(5.5, table.lookup(lat=0, lon=45))
        self.assertEqual(6.5, table.lookup(lat=0, lon=-45))
        self.assertEqual(6.5, table.lookup(lat=0, lon=315))

        self.assertEqual(4.75, table.lookup(lat=45, lon=45))

        np.testing.assert_equal([5, 5.5, 4.75], table.lookup(lat=np.array([0, 0, 45]), lon=np.array([0, 45, 45])))
