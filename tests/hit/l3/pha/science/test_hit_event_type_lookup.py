import os
import unittest

from imap_processing.hit.l3.pha.science.hit_event_type_lookup import HitEventTypeLookup


class TestHitEventTypeLookup(unittest.TestCase):
    def setUp(self):
        self.fake_lookup_path = "test_lookup.csv"

    def tearDown(self):
        if os.path.exists(self.fake_lookup_path):
            os.remove(self.fake_lookup_path)

    def test_hit_event_type_lookup_from_csv(self):
        with open(self.fake_lookup_path, "w") as csvfile:
            csvfile.write("\n".join(["L1A4,L1B14,L2A,L3A,L2B,L40B,L1B14,Range",
                                     "1,0,1,0,0,0,0,2A",
                                     "1,0,1,1,0,*,0,3A",
                                     "1,0,1,1,1,*,0,3A",
                                     "0,1,0,*,1,*,1,2B",
                                     "*,0,0,1,1,*,0,NOCALC"]))

        lookup = HitEventTypeLookup.from_csv(self.fake_lookup_path)
        self.assertEqual(4, len(lookup._rules))

        test_cases = [
            ("2A", {"L1A4", "L2A"}),
            ("3A", {"L1A4", "L2A", "L3A"}),
            ("3A", {"L1A4", "L2A", "L3A", "L40B"}),
            ("2B", {"L2B", "L1B14"}),
            (None, {"L3A", "L2B", "L40B"})
        ]
        for expected, detector_groups in test_cases:
            with self.subTest(detector_groups):
                self.assertEqual(expected, lookup.lookup_range(detector_groups))
