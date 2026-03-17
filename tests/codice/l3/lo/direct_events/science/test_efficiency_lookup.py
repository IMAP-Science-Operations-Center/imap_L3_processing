import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup


class TestEfficiencyLookup(unittest.TestCase):
    def test_read_from_csv(self):
        num_positions = 2
        num_energies = 3

        rng = np.random.default_rng()
        expected_efficiency_data = rng.random((num_positions, num_energies))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            output_csv = tmpdir / "efficiency_lookup.csv"
            with open(output_csv, "w") as csvfile:
                csv_writer = csv.writer(csvfile)

                csv_writer.writerow(["species","product","esa_step"]+[f"position_{i}" for i in range(num_positions)])
                for i in range(num_energies):
                    csv_writer.writerow([
                        "hplus","sw",i, *expected_efficiency_data[:, i]
                    ])
                for i in range(num_energies):
                    csv_writer.writerow([
                        "cplus6","sw",i, *expected_efficiency_data[:, i]*6
                    ])
                for i in range(num_energies):
                    csv_writer.writerow([
                        "hplus","nsw",i, *expected_efficiency_data[:, i]*42
                    ])


            efficiency_lookup = EfficiencyLookup.read_from_csv(output_csv, "hplus")

            np.testing.assert_almost_equal(efficiency_lookup.efficiency_data, expected_efficiency_data)
