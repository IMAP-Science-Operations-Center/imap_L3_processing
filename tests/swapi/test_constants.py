import unittest

from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_DISCARDED_BIN,
    SWAPI_FINE_SWEEP_BINS,
    SWAPI_K_FACTOR,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)


class TestConstants(unittest.TestCase):
    """Just because they're so subtly tricky..."""

    def test_discarded_bin_is_zero(self):
        self.assertEqual(SWAPI_DISCARDED_BIN, 0)

    def test_science_bins_excludes_discarded_bin(self):
        self.assertEqual(SWAPI_SCIENCE_BINS, slice(1, 72))

    def test_coarse_and_fine_partition_science_bins(self):
        # Coarse and fine bin slices must tile the science range with no overlap and no gap.
        self.assertEqual(SWAPI_COARSE_SWEEP_BINS.start, SWAPI_SCIENCE_BINS.start)
        self.assertEqual(SWAPI_COARSE_SWEEP_BINS.stop, SWAPI_FINE_SWEEP_BINS.start)
        self.assertEqual(SWAPI_FINE_SWEEP_BINS.stop, SWAPI_SCIENCE_BINS.stop)

    def test_coarse_sweep_has_62_bins_and_fine_has_9(self):
        self.assertEqual(
            SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start, 62
        )
        self.assertEqual(SWAPI_FINE_SWEEP_BINS.stop - SWAPI_FINE_SWEEP_BINS.start, 9)

    def test_simion_k_factor(self):
        self.assertAlmostEqual(SWAPI_K_FACTOR, 1.89)

    def test_l2_label_k_factor_differs_from_simion(self):
        # L2 esa_energy = SWAPI_L2_K_FACTOR × |V|, divided out by L3 to recover voltage.
        self.assertAlmostEqual(SWAPI_L2_K_FACTOR, 1.93)
        self.assertNotAlmostEqual(SWAPI_K_FACTOR, SWAPI_L2_K_FACTOR)


if __name__ == "__main__":
    unittest.main()
