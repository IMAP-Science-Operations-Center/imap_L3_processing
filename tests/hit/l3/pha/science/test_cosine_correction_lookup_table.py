import unittest

from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import CosineCorrectionLookupTable, Detector, \
    DetectedRange
from tests.test_helpers import get_test_data_path


class TestCosineCorrectionLookupTable(unittest.TestCase):
    def test_can_create_from_file(self):
        lookup_table = CosineCorrectionLookupTable(get_test_data_path(
            "hit/pha_events/imap_hit_l3_r2-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r3-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r4-cosines-text-not-cdf_20250203_v001.cdf"))

        self.assertEqual(150, len(lookup_table._range2_corrections))
        self.assertEqual(150, len(lookup_table._range3_corrections))
        self.assertEqual(150, len(lookup_table._range4_corrections))

        L1A0b_detector = Detector(name="L1A0b")
        L2A0_detector = Detector(name="L2A0")

        self.assertEqual(0.978643, lookup_table.get_cosine_correction(DetectedRange.R2, L1A0b_detector, L2A0_detector))
        self.assertEqual(0.674600, lookup_table.get_cosine_correction(DetectedRange.R3, L1A0b_detector, L2A0_detector))
        self.assertEqual(0.705122, lookup_table.get_cosine_correction(DetectedRange.R4, L1A0b_detector, L2A0_detector))
