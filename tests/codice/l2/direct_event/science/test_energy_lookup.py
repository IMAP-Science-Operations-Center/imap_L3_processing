import csv
import unittest


class EnergyLookup(unittest.TestCase):
    def test_find_energy_bin(self):
        test_energy_lookup_csv_path = "test_energy_lookup.csv"
        test_energy_bin_csv_path = " test_energy_bin.csv"

        header_row = ["SSD 0 - LG", "SSD 0 - MG", "SSD 0 - HG", "SSD 1 - LG", "SSD 1 - MG", "SSD 1 - HG"]
        energy_index_row_0 = [0, 1, 2, 10, 11, 12]
        energy_index_row_1 = [100, 101, 102, 1000, 1001, 1002]

        all_energy_index_rows = energy_index_row_0.copy()

        all_energy_index_rows.extend(energy_index_row_1)
        energy_index_rows = [energy_index_row_0, energy_index_row_1]

        mev_rows = []
        for i in all_energy_index_rows:
            mev_row = [i]
            mev_row.extend([(i * 10) + j for j in range(1, 4)])
            mev_rows.append(mev_row)

        with open(test_energy_lookup_csv_path, newline='') as test_energy_lookup_csv:
            csv_writer = csv.writer(test_energy_lookup_csv, delimiter=',')

            csv_writer.writerow(header_row)
            csv_writer.writerow(energy_index_row_0)
            csv_writer.writerow(energy_index_row_1)

        with open(test_energy_bin_csv_path, newline='') as test_energy_bin_lookup_csv:
            csv_writer = csv.writer(test_energy_bin_lookup_csv, delimiter=',')
            header_row = ["Energy (MeV)", "E lower (MeV)", "E upper (MeV)"]
            csv_writer.writerow(header_row)
            csv_writer.writerows(mev_rows)

        energy_lookup = EnergyLookup.from_files(test_energy_lookup_csv_path, test_energy_bin_csv_path)

        ssd_id = [0, 1]
        energy_range = [0, 1, 2]
        ssd_energy = [0, 1]

        energy_lower, energy, energy_upper = energy_lookup.convert_to_mev(ssd_id, ssd_energy, energy_range)
        self.assertEqual(mev_rows[0:], [energy_lower, energy, energy_upper])

        ssd_id = 0  # 0 #100
        energy_range = 1  # 10 #20 #30
        ssd_energy = 1  # 1 2 3
        expected_mev = [mev_row[1:] for mev_row in mev_rows if mev_row[0] == energy_index_rows[ssd_energy][ssd_id]]
        test_cases = [
            # sssid, energy_range, energy_row
            (0, 1, 1, mev_row)
        ]


if __name__ == '__main__':
    unittest.main()
