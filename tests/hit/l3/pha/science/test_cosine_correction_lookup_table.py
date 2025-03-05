import unittest

from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import CosineCorrectionLookupTable, Detector, \
    DetectedRange, DetectorSide, DetectorRange
from tests.test_helpers import get_test_data_path


class TestCosineCorrectionLookupTable(unittest.TestCase):
    def test_can_create_from_file(self):
        lookup_table = CosineCorrectionLookupTable(get_test_data_path(
            "hit/pha_events/imap_hit_l3_r2A-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r3A-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r4A-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r2B-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r3B-cosines-text-not-cdf_20250203_v001.cdf"), get_test_data_path(
            "hit/pha_events/imap_hit_l3_r4B-cosines-text-not-cdf_20250203_v001.cdf")
        )

        self.assertEqual(150, len(lookup_table._range2A_corrections))
        self.assertEqual(150, len(lookup_table._range3A_corrections))
        self.assertEqual(150, len(lookup_table._range4A_corrections))

        self.assertEqual(150, len(lookup_table._range2B_corrections))
        self.assertEqual(150, len(lookup_table._range3B_corrections))
        self.assertEqual(150, len(lookup_table._range4B_corrections))

        L1A0b_detector = Detector(layer=1, side="A", segment="0b", address=1234, group="L1A")
        L2A0_detector = Detector(layer=2, side="A", segment="0", address=5678, group="L2A")

        L1B0b_detector = Detector(layer=1, side="B", segment="0b", address=1234, group="L1A")
        L2B0_detector = Detector(layer=2, side="B", segment="0", address=5678, group="L2A")

        self.assertEqual(0.978643, lookup_table.get_cosine_correction(DetectedRange(DetectorRange.R2, DetectorSide.A),
                                                                      L1A0b_detector, L2A0_detector))
        self.assertEqual(0.978643, lookup_table.get_cosine_correction(DetectedRange(DetectorRange.R2, DetectorSide.B),
                                                                      L1B0b_detector, L2B0_detector))

        self.assertEqual(0.674600, lookup_table.get_cosine_correction(DetectedRange(DetectorRange.R3, DetectorSide.A),
                                                                      L1A0b_detector, L2A0_detector))
        self.assertEqual(0.674600, lookup_table.get_cosine_correction(DetectedRange(DetectorRange.R3, DetectorSide.B),
                                                                      L1B0b_detector, L2B0_detector))

        self.assertEqual(0.705122, lookup_table.get_cosine_correction(DetectedRange(DetectorRange.R4, DetectorSide.A),
                                                                      L1A0b_detector, L2A0_detector))
        self.assertEqual(0.705122, lookup_table.get_cosine_correction(DetectedRange(DetectorRange.R4, DetectorSide.B),
                                                                      L1B0b_detector, L2B0_detector))
