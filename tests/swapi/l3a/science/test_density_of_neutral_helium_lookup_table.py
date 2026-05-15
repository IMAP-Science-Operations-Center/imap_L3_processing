"""Tests for `DensityOfNeutralHeliumLookupTable`."""

from unittest import TestCase

import numpy as np

from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import (
    DensityOfNeutralHeliumLookupTable,
    bilinear_density,
)


def _table_with_360_anchor():
    # Bilinear corner table: angle in {0, 360}, distance in {0.1, 0.2}.
    # Density at 0/360 must match by construction (table already wraps).
    return DensityOfNeutralHeliumLookupTable.from_calibration_table(
        np.array(
            [
                [0.0, 0.1, 0.01],
                [0.0, 0.2, 0.02],
                [360.0, 0.1, 0.015],
                [360.0, 0.2, 0.025],
            ]
        )
    )


def _table_without_360_anchor():
    # Table provided only up to angle = 180; class inserts a 360 anchor copying angle=0.
    return DensityOfNeutralHeliumLookupTable.from_calibration_table(
        np.array(
            [
                [0.0, 0.1, 0.01],
                [0.0, 0.2, 0.02],
                [180.0, 0.1, 0.03],
                [180.0, 0.2, 0.04],
            ]
        )
    )


class TestDensityArrayAndScalarInputs(TestCase):
    def test_array_input_returns_array_with_per_element_density(self):
        lut = _table_with_360_anchor()
        densities = lut.density(
            np.array([0.0, 0.0, 180.0, 540.0]), np.array([0.5, 0.15, 0.2, 0.2])
        )
        # angle=0, dist=0.5 → out of bounds (>0.2), fill 0.
        # angle=0, dist=0.15 → midpoint of (0.01, 0.02) = 0.015.
        # angle=180, dist=0.2 → midpoint over angle of (0.02, 0.025) = 0.0225.
        # angle=540 → wraps to 180 → 0.0225.
        np.testing.assert_array_equal(densities, [0.0, 0.015, 0.0225, 0.0225])

    def test_scalar_distance_returns_scalar_density(self):
        lut = _table_with_360_anchor()
        density = lut.density(180.0, 0.2)
        self.assertEqual(density, 0.0225)

    def test_negative_angle_wraps_via_modulo(self):
        lut = _table_with_360_anchor()
        # -180 % 360 = 180; identical to the explicit 180 case above.
        np.testing.assert_array_equal(
            lut.density(np.array([-180.0]), np.array([0.2])), [0.0225]
        )

    def test_table_extension_to_360_when_input_only_reaches_180(self):
        lut = _table_without_360_anchor()
        # The class auto-extends the angle axis to 360 by copying the angle=0 row,
        # so an input at 270 deg interpolates between 180 (value=0.03) and 360 (=0.01).
        density = lut.density(270.0, distance=0.2)
        # angle=270 is midway between 180 and 360 → density at distance 0.2 = (0.04+0.02)/2 = 0.03.
        # The original test expected 0.03 (without the 360 extension being correctly average); confirm.
        self.assertEqual(density, 0.03)

    def test_distance_below_grid_returns_fill(self):
        lut = _table_with_360_anchor()
        # Distance 0.05 < min grid distance 0.1 → out of bounds → fill_value=0.
        np.testing.assert_array_equal(
            lut.density(np.array([0.0]), np.array([0.05])), [0.0]
        )

    def test_distance_above_grid_returns_fill(self):
        lut = _table_with_360_anchor()
        np.testing.assert_array_equal(
            lut.density(np.array([0.0]), np.array([5.0])), [0.0]
        )

    def test_density_at_grid_point(self):
        lut = _table_with_360_anchor()
        # Exact corner returns the corner value (no interpolation noise).
        np.testing.assert_allclose(
            lut.density(np.array([0.0]), np.array([0.1])), [0.01]
        )


class TestGetMinimumDistance(TestCase):
    def test_returns_first_distance_in_grid(self):
        lut = _table_with_360_anchor()
        self.assertEqual(lut.get_minimum_distance(), 0.1)

    def test_irregular_distance_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            DensityOfNeutralHeliumLookupTable.from_calibration_table(
                np.array(
                    [
                        [0.0, 0.5, 0.1],
                        [0.0, 1.0, 0.2],
                        [0.0, 5.0, 0.3],
                    ]
                )
            )


class TestBilinearDensity(TestCase):
    """`bilinear_density` is the numba-friendly twin of `density()`. It must match
    `RegularGridInterpolator(method='linear', bounds_error=False, fill_value=0)`
    over the wrap-360 angle axis and the bounded distance axis."""

    def test_corner_returns_corner_value(self):
        lut = _table_with_360_anchor()
        self.assertEqual(
            bilinear_density(
                lut,0.0, 0.1
            ),
            0.01,
        )

    def test_distance_midpoint(self):
        lut = _table_with_360_anchor()
        self.assertAlmostEqual(
            bilinear_density(
                lut,0.0, 0.15
            ),
            0.015,
        )

    def test_angle_modulo_wraps(self):
        # 540 mod 360 = 180; matches the 180-degree case from the existing tests.
        lut = _table_with_360_anchor()
        self.assertAlmostEqual(
            bilinear_density(
                lut,540.0, 0.2
            ),
            0.0225,
        )

    def test_negative_angle_wraps(self):
        # -180 mod 360 = 180.
        lut = _table_with_360_anchor()
        self.assertAlmostEqual(
            bilinear_density(
                lut,-180.0, 0.2
            ),
            0.0225,
        )

    def test_distance_below_grid_returns_zero(self):
        lut = _table_with_360_anchor()
        self.assertEqual(
            bilinear_density(
                lut,0.0, 0.05
            ),
            0.0,
        )

    def test_distance_above_grid_returns_zero(self):
        lut = _table_with_360_anchor()
        self.assertEqual(
            bilinear_density(
                lut,0.0, 5.0
            ),
            0.0,
        )

    def test_matches_scipy_density_method(self):
        lut = _table_with_360_anchor()
        for angle, distance in [
            (10.0, 0.13),
            (123.4, 0.18),
            (-45.0, 0.11),
            (720.5, 0.17),
        ]:
            np.testing.assert_allclose(
                bilinear_density(lut, angle, distance),
                lut.density(float(angle), float(distance)),
                rtol=1e-12,
            )


class TestTableExtension(TestCase):
    def test_360_anchor_inserted_when_missing(self):
        lut = _table_without_360_anchor()
        # `from_calibration_table` inserts 360 so the angle axis wraps periodically.
        self.assertIn(360.0, lut.angle_grid)

    def test_360_anchor_density_copies_angle_zero(self):
        lut = _table_without_360_anchor()
        # angle=360 row should mirror angle=0: densities (0.01, 0.02) at distances (0.1, 0.2).
        np.testing.assert_allclose(lut.density_values[-1], lut.density_values[0])

    def test_asserts_table_starts_at_angle_zero(self):
        # If the table doesn't include angle=360 AND doesn't start at 0, it should fail
        # the documented `expected table to start at angle 0` assertion.
        with self.assertRaises(AssertionError):
            DensityOfNeutralHeliumLookupTable.from_calibration_table(
                np.array(
                    [
                        [10.0, 0.1, 0.01],
                        [10.0, 0.2, 0.02],
                    ]
                )
            )
