import unittest

from imap_l3_processing.swapi.l3a.science.solar_wind.azimuthal_regions import (
    REGION_OPEN_APERTURE_NEG,
    REGION_OPEN_APERTURE_POS,
    REGION_SUNGLASSES,
    Region,
)


class TestRegionConstants(unittest.TestCase):
    """Tests for the `REGION_SUNGLASSES`, `REGION_OPEN_APERTURE_NEG`, and `REGION_OPEN_APERTURE_POS` constants."""

    def test_sunglasses_region_flags_and_azimuth_sign(self):
        """The SG constant reports sunglasses=True, open-aperture=False, and azimuth_sign=0 since it spans both halves of the boresight band."""
        self.assertTrue(REGION_SUNGLASSES.is_sunglasses)
        self.assertFalse(REGION_SUNGLASSES.is_open_aperture)
        self.assertEqual(REGION_SUNGLASSES.azimuth_sign, 0)

    def test_open_aperture_negative_wing_flags_and_sign(self):
        """The OA- wing constant reports open-aperture=True and azimuth_sign=-1 to select the negative-azimuth half."""
        self.assertFalse(REGION_OPEN_APERTURE_NEG.is_sunglasses)
        self.assertTrue(REGION_OPEN_APERTURE_NEG.is_open_aperture)
        self.assertEqual(REGION_OPEN_APERTURE_NEG.azimuth_sign, -1)

    def test_open_aperture_positive_wing_flags_and_sign(self):
        """The OA+ wing constant reports open-aperture=True and azimuth_sign=+1 to select the positive-azimuth half."""
        self.assertFalse(REGION_OPEN_APERTURE_POS.is_sunglasses)
        self.assertTrue(REGION_OPEN_APERTURE_POS.is_open_aperture)
        self.assertEqual(REGION_OPEN_APERTURE_POS.azimuth_sign, +1)

    def test_region_namedtuple_field_order_is_stable(self):
        """The Region NamedTuple field order is locked so positional unpacking in numba forward-model loops keeps working."""
        self.assertEqual(
            Region._fields, ("is_sunglasses", "is_open_aperture", "azimuth_sign")
        )


if __name__ == "__main__":
    unittest.main()
