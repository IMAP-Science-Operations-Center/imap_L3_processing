import unittest

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, ElevationLookup


class TestCoDICEAngleLookup(unittest.TestCase):
    def test_spin_angle_lookup(self):
        spin_angle_lut = SpinAngleLookup()
        for i in range(24):
            self.assertEqual(i, spin_angle_lut.get_spin_angle_index(15.0 * i + 7.5))

    def test_elevation_lookup(self):
        elevation_angle_lut = ElevationLookup()
        for i in range(13):
            self.assertEqual(i, elevation_angle_lut.get_elevation_index(15.0 * i))
