import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l2.direct_event.science.azimuth_lookup import AzimuthLookup


class TestAzimuthLookup(unittest.TestCase):
    def test_converts_ssd_id_to_azimuth(self):
        ssd_ids = np.arange(16)
        angles = [180, 210, 210, 240, 270, 300, 300, 330, 0, 30, 30, 60, 90, 90, 120, 150]

        test_azimuth_lookup_csv_path = 'test_azimuth_lookup.csv'
        with tempfile.TemporaryDirectory() as tempdir:
            temp_azimuth_csv_file = Path(tempdir) / test_azimuth_lookup_csv_path
            with open(temp_azimuth_csv_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["SSD ID", "Azimuth Angle"])
                writer.writerow(ssd_ids)
                writer.writerow(angles)

            azimuth_lookup = AzimuthLookup.from_files(temp_azimuth_csv_file)

            for ssd_ids, expected_azimuth in zip(ssd_ids, angles):
                actual_azimuth = azimuth_lookup.get_azimuth_by_ssd_id(ssd_ids)
                self.assertEqual(expected_azimuth, actual_azimuth)
