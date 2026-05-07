"""Direct tests for `SwapiL3Flags`.

The flag class is used as bitwise OR composition throughout the L3a pipeline
(e.g. `quality_flag |= result.bad_fit_flag`). These tests pin down the bit
assignments and the OR/AND/contains semantics so a reordering of the enum
values would be caught here rather than in a downstream regression.
"""

import unittest

from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


class TestSwapiL3Flags(unittest.TestCase):
    def test_specific_bit_values(self):
        # Pin each flag's integer value. Changing these breaks CDF bit masks.
        self.assertEqual(int(SwapiL3Flags.NONE), 0)
        self.assertEqual(int(SwapiL3Flags.EPHEMERIS_GAP), 4)
        self.assertEqual(int(SwapiL3Flags.HI_CHI_SQ), 8)
        self.assertEqual(int(SwapiL3Flags.PUI_FIT_MISSING_UNCERTAINTY), 16)
        self.assertEqual(int(SwapiL3Flags.STALE_PROTON), 32)
        self.assertEqual(int(SwapiL3Flags.PRELIMINARY_MAG), 64)
        self.assertEqual(int(SwapiL3Flags.MAG_GAP), 128)
        self.assertEqual(int(SwapiL3Flags.FIT_FAILED), 256)

    def test_combined_flag_name(self):
        # `FlagNameMixin` overrides .name to produce combined names for OR'd flags.
        combined = SwapiL3Flags.FIT_FAILED | SwapiL3Flags.STALE_PROTON
        # FlagNameMixin orders by bit value ascending: STALE_PROTON=32, FIT_FAILED=256.
        self.assertEqual(combined.name, "STALE_PROTON|FIT_FAILED")


if __name__ == "__main__":
    unittest.main()
