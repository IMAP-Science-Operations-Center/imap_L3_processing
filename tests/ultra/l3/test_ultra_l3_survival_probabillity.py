import unittest

import imap_processing.tests.ultra.data.mock_data as umd
from imap_processing.ena_maps.ena_maps import UltraPointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry

from imap_l3_processing.ultra.l3.science.ultra_survival_probability import UltraSurvivalProbability


class TestUltraSurvivalProbability(unittest.TestCase):

    def test_ultra_survival_probability_pset_calls_super(self):
        l1cdataset = umd.mock_l1c_pset_product_healpix()
        prod = UltraSurvivalProbability(l1cdataset)
        self.assertIsInstance(prod, UltraPointingSet)

        self.assertIs(l1cdataset, prod.data)
        self.assertEqual(geometry.SpiceFrame.ECLIPJ2000, prod.spice_reference_frame)

    def test_ultra_survival_probability_pset_data_contains_survival_prob(self):
        l1cdataset = umd.mock_l1c_pset_product_healpix()
        prod = UltraSurvivalProbability(l1cdataset)

        self.assertIn("survival_probability_times_exposure", prod.data)
        self.assertIn("exposure_time", prod.data)

        self.assertEqual((CoordNames.TIME.value,
                          CoordNames.ENERGY.value,
                          CoordNames.HEALPIX_INDEX.value,), prod.data["survival_probability_times_exposure"].dims)

        self.assertEqual((CoordNames.HEALPIX_INDEX.value,), prod.data["exposure_time"].dims)

        self.assertEqual(l1cdataset["counts"].values.shape,
                         prod.data["survival_probability_times_exposure"].values.shape)


if __name__ == '__main__':
    unittest.main()
