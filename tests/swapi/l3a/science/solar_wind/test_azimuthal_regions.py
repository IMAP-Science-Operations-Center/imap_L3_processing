"""Tests for `solar_wind.azimuthal_regions` — three named `Region` constants
covering SWAPI's azimuthal regions: a sunglasses (SG) band centered on the
boresight and the two open-aperture (OA) wings on either side."""

import unittest

from imap_l3_processing.swapi.l3a.science.solar_wind.azimuthal_regions import (
    REGION_OPEN_APERTURE_NEG,
    REGION_OPEN_APERTURE_POS,
    REGION_SUNGLASSES,
    Region,
)


class TestRegionConstants(unittest.TestCase):
    def test_sunglasses_region_flags_and_azimuth_sign(self):
        # SG covers the full -20°..+20° band centered on the boresight, so its
        # azimuth sign is 0 (no half-selection).
        self.assertTrue(REGION_SUNGLASSES.is_sunglasses)
        self.assertFalse(REGION_SUNGLASSES.is_open_aperture)
        self.assertEqual(REGION_SUNGLASSES.azimuth_sign, 0)

    def test_open_aperture_negative_wing_flags_and_sign(self):
        self.assertFalse(REGION_OPEN_APERTURE_NEG.is_sunglasses)
        self.assertTrue(REGION_OPEN_APERTURE_NEG.is_open_aperture)
        self.assertEqual(REGION_OPEN_APERTURE_NEG.azimuth_sign, -1)

    def test_open_aperture_positive_wing_flags_and_sign(self):
        self.assertFalse(REGION_OPEN_APERTURE_POS.is_sunglasses)
        self.assertTrue(REGION_OPEN_APERTURE_POS.is_open_aperture)
        self.assertEqual(REGION_OPEN_APERTURE_POS.azimuth_sign, +1)

    def test_region_namedtuple_field_order_is_stable(self):
        # Some downstream code (e.g. numba forward-model loops) unpacks
        # Region tuples by position; the field order must not drift.
        self.assertEqual(
            Region._fields, ("is_sunglasses", "is_open_aperture", "azimuth_sign")
        )


if __name__ == "__main__":
    unittest.main()
