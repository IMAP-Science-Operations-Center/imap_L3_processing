import csv
import tempfile
import unittest
from pathlib import Path

from imap_l3_processing.codice.l3.hi.direct_event.science.tof_lookup import TOFLookup


class TestTOFLookup(unittest.TestCase):
    def test_tof_lookup(self):
        tof_lookup_header_row = ["TOF Bit", "E/n (MeV/n)", "E/n Lower (MeV/n)", "E/n Upper (MeV/n)"]
        tof_lookup_rows = []
        for tof_bit in range(1024):
            energy = tof_bit * 1.0
            lower = tof_bit * 2.0
            upper = tof_bit * 3.0
            tof_lookup_rows.append([energy, lower, upper])

        tof_lookup_csv_path = "tof_lookup.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_tof_csv_file = Path(tmpdir) / tof_lookup_csv_path
            with open(temp_tof_csv_file, "w") as csv_file:
                csvwriter = csv.writer(csv_file)
                csvwriter.writerow(tof_lookup_header_row)
                csvwriter.writerows(tof_lookup_rows)

            tof_lookup = TOFLookup.read_from_file(temp_tof_csv_file)

            for tof_bit in range(1024):
                self.assertEqual(tof_lookup_rows[tof_bit][0], tof_lookup[tof_bit].lower_bound)
                self.assertEqual(tof_lookup_rows[tof_bit][1], tof_lookup[tof_bit].energy)
                self.assertEqual(tof_lookup_rows[tof_bit][2], tof_lookup[tof_bit].upper_bound)
