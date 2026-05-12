from unittest import TestCase
from unittest.mock import patch

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup import GeometricFactorLookup
from tests.test_helpers import get_test_data_folder


class TestGeometricFactorLookup(TestCase):

    def test_default_esa_step_end_index(self):
        geometric_factor_lookup = GeometricFactorLookup(2, 0.5)
        expected_esa_step_end_index = \
            np.array(
                [0, 1, 2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 27, 31, 35, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85, 91, 97,
                 103, 109, 115, 121, 127])
        np.testing.assert_array_equal(geometric_factor_lookup._esa_step_end_index, expected_esa_step_end_index)

    def test_read_from_csv(self):
        expected_num_esa_steps = 128
        expected_num_positions = 24

        path = get_test_data_folder() / 'codice' / 'imap_codice_l2-lo-gfactor_20251212_v003.csv'
        lookup = GeometricFactorLookup.read_from_csv(path)

        self.assertIsInstance(lookup, GeometricFactorLookup)

        self.assertEqual((expected_num_esa_steps, expected_num_positions), lookup._full_factor.shape)
        self.assertEqual((expected_num_esa_steps, expected_num_positions), lookup._reduced_factor.shape)
        self.assertEqual(8.71567e-05, lookup._full_factor[0, 0])
        self.assertEqual(9.7615504e-06, lookup._reduced_factor[0, 0])

    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_ESA_STEPS", 3)
    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_SPIN_SECTORS",
           3)
    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_AZIMUTH_BINS",
           2)
    def test_get_geometric_factors_gets_reduced_where_half_spin_greater_than_rgfo_half_spin(self):
        rgfo_half_spin_threshold = 5
        below_half_spin_threshold = 3
        above_half_spin_threshold = 7
        rgfo_half_spin = np.ma.masked_array(data=np.array([rgfo_half_spin_threshold, rgfo_half_spin_threshold]))
        half_spin = np.ma.masked_array(data=np.array([
            [below_half_spin_threshold, above_half_spin_threshold, above_half_spin_threshold],
            [above_half_spin_threshold, below_half_spin_threshold, above_half_spin_threshold],
        ]))
        full_factors = np.array([[10, 20], [30, 40], [50, 60]])
        reduced_factors = np.array([[1, 2], [3, 4], [5, 6]])
        geometric_factor_lookup = GeometricFactorLookup(
            _full_factor=full_factors,
            _reduced_factor=reduced_factors,
        )
        actual = geometric_factor_lookup.get_geometric_factors(
            rgfo_half_spin=rgfo_half_spin,
            rgfo_spin_sector=np.zeros(2),
            rgfo_esa_step=np.zeros(2),
            half_spin=half_spin,
        )
        all_spin_sectors_and_positions = (3, 2)

        epoch_esa_step_index_pairs_and_expected_factors = [
            (0, 0, full_factors),
            (0, 1, reduced_factors),
            (0, 2, reduced_factors),
            (1, 0, reduced_factors),
            (1, 1, full_factors),
            (1, 2, reduced_factors),
        ]
        for epoch, esa_step, expected_factors in epoch_esa_step_index_pairs_and_expected_factors:
            np.testing.assert_array_equal(
                actual[epoch, esa_step, :, :],
                np.broadcast_to(expected_factors[esa_step], all_spin_sectors_and_positions))

    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_ESA_STEPS", 3)
    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_SPIN_SECTORS",
           3)
    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_AZIMUTH_BINS",
           2)
    def test_get_geometric_factors_gets_reduced_where_equal_half_spin_greater_spin_sector(self):
        rgfo_half_spin = np.ma.masked_array(data=np.array([2, 2]))

        half_spin = np.ma.masked_array(data=np.array([
            [2, 2, 2],
            [2, 2, 2],
        ]))

        rgfo_spin_sector = np.ma.masked_array(data=np.array([0, 1]))

        full_factors = np.array([[10, 20], [30, 40], [50, 60]])
        reduced_factors = np.array([[1, 2], [3, 4], [5, 6]])
        geometric_factor_lookup = GeometricFactorLookup(
            _full_factor=full_factors,
            _reduced_factor=reduced_factors,
        )
        actual = geometric_factor_lookup.get_geometric_factors(
            rgfo_half_spin=rgfo_half_spin,
            rgfo_spin_sector=rgfo_spin_sector,
            rgfo_esa_step=np.zeros(2),
            half_spin=half_spin,
        )

        np.testing.assert_array_equal(
            actual[0, :, 1, :],
            reduced_factors, full_factors.shape
        )
        np.testing.assert_array_equal(
            actual[1, :, 0, :],
            full_factors
        )
        np.testing.assert_array_equal(
            actual[1, :, 2, :],
            reduced_factors
        )

    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_ESA_STEPS", 3)
    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_SPIN_SECTORS",
           3)
    @patch("imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup.CODICE_LO_NUM_AZIMUTH_BINS",
           2)
    def test_get_geometric_factors_gets_reduced_where_equal_half_spin_greater_spin_sector(self):
        rgfo_half_spin = np.ma.masked_array(data=np.array([2, 2]))

        half_spin = np.ma.masked_array(data=np.array([
            [2, 2, 2],
            [2, 2, 2],
        ]))

        rgfo_spin_sector = np.ma.masked_array(data=np.array([1, 1]))
        rgfo_esa_step = np.ma.masked_array(data=np.array([0, 1]))
        full_factors = np.array([[10, 20], [30, 40], [50, 60]])
        reduced_factors = np.array([[1, 2], [3, 4], [5, 6]])
        geometric_factor_lookup = GeometricFactorLookup(
            _full_factor=full_factors,
            _reduced_factor=reduced_factors,
        )
        actual = geometric_factor_lookup.get_geometric_factors(
            rgfo_half_spin=rgfo_half_spin,
            rgfo_spin_sector=rgfo_spin_sector,
            rgfo_esa_step=rgfo_esa_step,
            half_spin=half_spin,
        )

        epoch_esa_step_index_and_expected_factors = [
            (0, 0, full_factors),
            (0, 1, reduced_factors),
            (0, 2, reduced_factors),
            (1, 0, full_factors),
            (1, 1, full_factors),
            (1, 2, reduced_factors),
        ]
        for epoch, esa_step, expected_factors in epoch_esa_step_index_and_expected_factors:
            np.testing.assert_array_equal(
                actual[epoch, esa_step, 1, :],
                expected_factors[esa_step]
            )