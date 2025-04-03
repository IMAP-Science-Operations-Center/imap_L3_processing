import csv
import tempfile
import unittest
from pathlib import Path

from imap_l3_processing.codice.l2.direct_event.science.energy_lookup import EnergyLookup


class TestEnergyLookup(unittest.TestCase):
    def test_find_energy_bin(self):
        test_energy_lookup_csv_path = "test_energy_lookup.csv"
        test_energy_bin_csv_path = " test_energy_bin.csv"

        header_row = ["SSD 0 - LG", "SSD 0 - MG", "SSD 0 - HG", "SSD 1 - LG", "SSD 1 - MG", "SSD 1 - HG"]

        ssd_lookup_table = []
        energy_bins_table = []

        for energy_index_row in range(2):
            ssd_lookup_row = []
            for ssd_id in range(2):
                for gain in range(3):
                    val = ssd_id * 3 + gain
                    energy_bin = energy_index_row * 6 + val
                    ssd_lookup_row.append(energy_bin)
                    energy_bins_table.append([
                        int(f"{ssd_id}{gain}{energy_index_row}1"),
                        int(f"{ssd_id}{gain}{energy_index_row}2"),
                        int(f"{ssd_id}{gain}{energy_index_row}3"),
                    ])

            ssd_lookup_table.append(ssd_lookup_row)

        with tempfile.TemporaryDirectory() as tempdir:
            temp_energy_file = Path(tempdir) / test_energy_lookup_csv_path
            with open(temp_energy_file, 'w', newline='') as test_energy_lookup_csv:
                csv_writer = csv.writer(test_energy_lookup_csv, delimiter=',')

                csv_writer.writerow(header_row)
                csv_writer.writerows(ssd_lookup_table)

            temp_bin_file = Path(tempdir) / test_energy_bin_csv_path
            with open(temp_bin_file, 'w', newline='') as test_energy_bin_lookup_csv:
                csv_writer = csv.writer(test_energy_bin_lookup_csv, delimiter=',')
                header_row = ["Energy (MeV)", "E lower (MeV)", "E upper (MeV)"]
                csv_writer.writerow(header_row)
                csv_writer.writerows(energy_bins_table)

            energy_lookup = EnergyLookup.from_files(temp_energy_file, temp_bin_file)

            for row in range(2):
                for ssd_id in range(2):
                    for gain in range(3):
                        energy_vals = energy_lookup.convert_to_mev(ssd_id, gain, row)
                        self.assertEqual(energy_vals[0], int(f"{ssd_id}{gain}{row}1"))
                        self.assertEqual(energy_vals[1], int(f"{ssd_id}{gain}{row}2"))
                        self.assertEqual(energy_vals[2], int(f"{ssd_id}{gain}{row}3"))
