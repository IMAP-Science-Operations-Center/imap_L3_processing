import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from tests.test_helpers import get_test_data_path


class TestEnergyLookup(unittest.TestCase):
    def test_read_from_csv(self):
        input_file = get_test_data_path("codice") / "imap_codice_lo-energy-per-charge_20241110_v002.csv"
        actual_energy_lookup = EnergyLookup.read_from_csv(input_file)

        np.testing.assert_array_equal(actual_energy_lookup.bin_centers[0], 81.216)
        np.testing.assert_array_equal(actual_energy_lookup.bin_centers[127], 0.507357072)
        np.testing.assert_array_equal(actual_energy_lookup.delta_minus[0], 81.216 - 79.60339125)
        np.testing.assert_array_equal(actual_energy_lookup.delta_minus[127], 0.507357072 - 0.497319345)
        np.testing.assert_array_equal(actual_energy_lookup.delta_plus[0], 82.855305 - 81.216)
        np.testing.assert_array_equal(actual_energy_lookup.delta_plus[127], 0.517446956 - 0.507357072)
