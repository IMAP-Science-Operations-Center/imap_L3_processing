import unittest

from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable, DetectorGain
from tests.test_helpers import get_test_data_path


class TestGainLookupTable(unittest.TestCase):
    def test_loads_from_file(self):
        lookup = GainLookupTable.from_file(
            high_gain_file_path=get_test_data_path(
                'hit/pha_events/imap_hit_hi-gain-lookup_20250203_v000.csv'),
            low_gain_file_path=get_test_data_path(
                'hit/pha_events/imap_hit_lo-gain-lookup_20250203_v000.csv'))

        self.assertEqual(2, len(lookup))

        self.assertEqual(58, len(lookup[DetectorGain.LOW]))
        self.assertEqual(0.263735, lookup[DetectorGain.LOW][17].a)
        self.assertEqual(-12.267, lookup[DetectorGain.LOW][17].b)

        self.assertEqual(58, len(lookup[DetectorGain.HIGH]))
        self.assertEqual(0.006271, lookup[DetectorGain.HIGH][17].a)
        self.assertEqual(0.083, lookup[DetectorGain.HIGH][17].b)
