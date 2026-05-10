import unittest

import pandas as pd

from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
    interpolate_azimuthal_transmission,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path

_TRANSMISSION_CSV = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_SPACING_DEG = SwapiResponse.AZIMUTHAL_TRANSMISSION_SPACING_DEG


def _load_real_grid() -> AzimuthalTransmissionGrid:
    # Build the grid the same way `SwapiResponse.from_files` does.
    df = pd.read_csv(_TRANSMISSION_CSV)
    values = df["transmission"].fillna(0).values.astype(float)
    return AzimuthalTransmissionGrid(values=values, spacing=_SPACING_DEG)


class TestInterpolateAzimuthalTransmission(unittest.TestCase):
    """
    `interpolate_azimuthal_transmission` linearly interpolates the azimuthal
    transmission curve `T(|az|)` with two cases of special handling:

    (a) the input azimuth is wrapped into (-180°, 180°] before lookup, and
    (b) the lookup uses `|az|`, so the curve is implicitly mirrored at zero.
    """

    @classmethod
    def setUpClass(cls):
        cls.grid = _load_real_grid()
        cls.values = cls.grid.values
        cls.n = len(cls.values)
        cls.last_az = _SPACING_DEG * (cls.n - 1)

    def test_at_grid_point_returns_csv_value(self):
        """Exact-index lookups: interpolation weights collapse to 1, so the
        result must equal the CSV cell at that index. Sample indices span
        the SG attenuated region, the SG/OA shoulder, the OA plateau, and
        the zero-padded tail."""
        for idx in [0, 10, 199, 299, 500, 1000, self.n - 1]:
            with self.subTest(idx=idx):
                az = idx * _SPACING_DEG
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertAlmostEqual(got, float(self.values[idx]))

    def test_linear_interpolation_halfway_between_grid_points(self):
        """Halfway between two adjacent CSV rows must equal their average.
        Pick a pair on the SG→OA rising shoulder so the two endpoints
        differ meaningfully (rather than both sitting on the SG floor)."""
        idx = 250  # az ≈ 25°, mid-shoulder
        az_lower = idx * _SPACING_DEG
        midpoint_az = az_lower + 0.5 * _SPACING_DEG
        expected = 0.5 * (self.values[idx] + self.values[idx + 1])
        got = interpolate_azimuthal_transmission(self.grid, midpoint_az)
        self.assertAlmostEqual(got, float(expected))

    def test_linear_interpolation_off_center_between_grid_points(self):
        """Off-center fractions exercise the interpolation weights, not just
        the symmetric midpoint case: at fraction f between idx and idx+1, the
        result must be `(1-f)·v[idx] + f·v[idx+1]`. Pick the same SG→OA
        shoulder pair so the endpoints differ meaningfully."""
        idx = 250  # az ≈ 25°, mid-shoulder
        az_lower = idx * _SPACING_DEG
        for fraction in [0.25, 0.7, 0.9]:
            with self.subTest(fraction=fraction):
                az = az_lower + fraction * _SPACING_DEG
                expected = (1 - fraction) * self.values[idx] + fraction * self.values[idx + 1]
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertAlmostEqual(got, float(expected))

    def test_symmetric_about_zero(self):
        """The lookup uses |az|, so T(-x) must equal T(+x) for any x."""
        for az in [0.37, 12.5, 29.95, 88.123]:
            with self.subTest(az=az):
                positive = interpolate_azimuthal_transmission(self.grid, az)
                negative = interpolate_azimuthal_transmission(self.grid, -az)
                self.assertAlmostEqual(positive, negative)

    def test_clamps_at_and_past_the_last_index(self):
        """At and past the last stored index, both bracketing neighbors are
        clamped to the array end so the result equals the last table value.
        Covers three sub-cases that share the same expected outcome:
          - exactly on the last index (i_lower = n-1, i_upper = n)
          - just past the last index (both indices need clamping)
          - near the upper edge of the wrap interval
        """
        last_value = float(self.values[-1])
        for label, az in [
            ("exactly on last index", self.last_az),
            ("just past last index", self.last_az + 0.07),
            ("near upper wrap edge", 179.95),
        ]:
            with self.subTest(case=label, az=az):
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertEqual(got, last_value)

    def test_wraps_arguments_into_canonical_180_interval(self):
        """The wrap formula `(az + 180) % 360 - 180` maps any real azimuth into
        (-180°, 180°]. Some particular cases:
          - 359° wraps to -1° (|az| = 1°)
          - exact ±180° both wrap to 180°
          - -181° wraps to +179° (|az| = 179°)
          - multi-period inputs (±541°) wrap into the canonical interval
            (541 mod 360 = 181 → wraps to -179° → |az| = 179°)
        """
        idx_at_1deg = int(round(1.0 / _SPACING_DEG))
        expected_at_1deg = float(self.values[idx_at_1deg])
        last_value = float(self.values[-1])
        idx_at_179deg = int(round(179.0 / _SPACING_DEG))
        expected_at_179deg = float(self.values[idx_at_179deg])

        for label, az, expected in [
            ("almost full revolution", 359.0, expected_at_1deg),
            ("exact +180", 180.0, last_value),
            ("exact -180", -180.0, last_value),
            ("just past -180", -181.0, expected_at_179deg),
            ("multi-period positive", 541.0, expected_at_179deg),
            ("multi-period negative", -541.0, expected_at_179deg),
        ]:
            with self.subTest(case=label, az=az):
                got = interpolate_azimuthal_transmission(self.grid, az)
                self.assertAlmostEqual(got, expected)


class TestRealGridShape(unittest.TestCase):
    """Sanity-check the loaded CSV layout — the interpolator tests above
    assume 0.1° spacing and a fixed table length."""

    def test_csv_layout_matches_production_spacing(self):
        grid = _load_real_grid()
        self.assertEqual(grid.spacing, 0.1)
        # CSV spans 0° to 180° inclusive at 0.1° → 1801 rows.
        self.assertEqual(len(grid.values), 1801)
        # The SG region is attenuated by ≈1/1000 at az=0.
        self.assertAlmostEqual(float(grid.values[0]), 1e-3)


if __name__ == "__main__":
    unittest.main()
