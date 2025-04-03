import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l2.direct_event.science.time_of_flight_lookup import TimeOfFlightLookup


class TestTimeOfFlightLookup(unittest.TestCase):
    def test_lookup_nanoseconds(self):
        tof_lookup_file = "tof_lookup.csv"
        tof_headers = ["TOF(ns)"]

        tof_first_col = np.arange(25)
        tof_values = np.stack([tof_first_col, np.random.rand(25)], axis=1)

        with tempfile.TemporaryDirectory() as tempdir:
            tof_lookup_path = Path(tempdir) / tof_lookup_file
            with open(tof_lookup_path, "w", newline='') as tof_file:
                csv_writer = csv.writer(tof_file, delimiter=',')
                csv_writer.writerow(tof_headers)
                csv_writer.writerows(list(tof_values))

            tof_lookup = TimeOfFlightLookup.from_files(tof_lookup_path)

            for i in range(25):
                np.testing.assert_array_equal(tof_values[i][1], tof_lookup.convert_to_nanoseconds(i))


if __name__ == '__main__':
    unittest.main()
