import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup


class TestEfficiencyLookup(unittest.TestCase):
    def test_read_from_csv(self):
        num_azimuths = 2
        num_energies = 3

        rng = np.random.default_rng()
        expected_efficiency_data = rng.random((num_azimuths, num_energies))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            output_csv = tmpdir / "efficiency_lookup.csv"
            with open(output_csv, "w") as csvfile:
                csv_writer = csv.writer(csvfile)

                csv_writer.writerow("# some header info goes here")
                csv_writer.writerows([
                    expected_efficiency_data[:, 0],
                    expected_efficiency_data[:, 1],
                    expected_efficiency_data[:, 2]
                ])

            efficiency_lookup = EfficiencyLookup.read_from_csv(output_csv)

            np.testing.assert_array_equal(efficiency_lookup.efficiency_data, expected_efficiency_data)
