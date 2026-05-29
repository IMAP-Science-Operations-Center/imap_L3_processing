from unittest import TestCase

import numpy as np

from imap_l3_processing.swapi.l3a.science.pickup_ion.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable


def _table_with_360_anchor():
    return DensityOfNeutralHeliumLookupTable(
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
    return DensityOfNeutralHeliumLookupTable(
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
        np.testing.assert_array_equal(densities, [0.0, 0.015, 0.0225, 0.0225])

    def test_scalar_distance_returns_scalar_density(self):
        lut = _table_with_360_anchor()
        density = lut.density(180.0, 0.2)
        self.assertEqual(density, 0.0225)

    def test_negative_angle_wraps_via_modulo(self):
        lut = _table_with_360_anchor()
        np.testing.assert_array_equal(
            lut.density(np.array([-180.0]), np.array([0.2])), [0.0225]
        )

    def test_table_extension_to_360_when_input_only_reaches_180(self):
        lut = _table_without_360_anchor()
        density = lut.density(270.0, distance=0.2)
        self.assertEqual(density, 0.03)

    def test_distance_above_grid_returns_fill(self):
        lut = _table_with_360_anchor()
        np.testing.assert_array_equal(
            lut.density(np.array([0.0]), np.array([5.0])), [0.0]
        )

    def test_density_at_grid_point(self):
        lut = _table_with_360_anchor()
        np.testing.assert_allclose(
            lut.density(np.array([0.0]), np.array([0.1])), [0.01]
        )

    def test_density_below_lower_limit(self):
        lut = _table_with_360_anchor()
        np.testing.assert_allclose(
            lut.density(np.array([0.0]), np.array([0.1 / 2])), [0.01 / 2]
        )


class TestTableExtension(TestCase):
    def test_360_anchor_inserted_when_missing(self):
        lut = _table_without_360_anchor()
        self.assertIn(360.0, lut.grid[0])

    def test_360_anchor_density_copies_angle_zero(self):
        lut = _table_without_360_anchor()
        np.testing.assert_allclose(lut.densities[-1], lut.densities[0])

    def test_asserts_table_starts_at_angle_zero(self):
        with self.assertRaises(AssertionError):
            DensityOfNeutralHeliumLookupTable(
                np.array(
                    [
                        [10.0, 0.1, 0.01],
                        [10.0, 0.2, 0.02],
                    ]
                )
            )
