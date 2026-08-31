import unittest

import numpy as np

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_mass_correction_lookup_table import \
    CodiceHiMassCorrectionLookupTable
from tests.test_helpers import get_test_data_path


class TestCodiceHiMassCorrectionLookupTable(unittest.TestCase):
    def test_from_file(self):
        file_path = get_test_data_path("codice/truncated_codice_l3-hi-mass-correction-lut_20251008_v001.xlsx")
        mass_correction_lookup = CodiceHiMassCorrectionLookupTable.from_file(file_path)

        self.assertIsInstance(mass_correction_lookup, CodiceHiMassCorrectionLookupTable)

    def test_lookup_correction_factor(self):
        file_path = get_test_data_path("codice/truncated_codice_l3-hi-mass-correction-lut_20251008_v001.xlsx")
        mass_correction_lookup = CodiceHiMassCorrectionLookupTable.from_file(file_path)

        self.assertEqual(0.753777, mass_correction_lookup.lookup_correction_factor(ssd_id=7, energy_channel=40, tof=6))

        self.assertEqual(7.47142, mass_correction_lookup.lookup_correction_factor(ssd_id=3, energy_channel=24, tof=0))
        self.assertEqual(1, mass_correction_lookup.lookup_correction_factor(ssd_id=3, energy_channel=24, tof=1))
        self.assertEqual(12.8534, mass_correction_lookup.lookup_correction_factor(ssd_id=3, energy_channel=24, tof=6))

        missing_ssds = [2, 6, 8, 9, 10, 12, 13, 14, 15]
        for ssd_id in missing_ssds:
            self.assertEqual(1.0, mass_correction_lookup.lookup_correction_factor(ssd_id, energy_channel=24, tof=0))
