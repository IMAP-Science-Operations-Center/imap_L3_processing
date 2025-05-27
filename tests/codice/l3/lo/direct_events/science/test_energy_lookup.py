import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path


class TestEnergyLookup(unittest.TestCase):
    def test_read_from_csv(self):
        with (tempfile.TemporaryDirectory() as tmpdir):
            tmpdir = Path(tmpdir)

            output_path = tmpdir / "test_energy_lookup.csv"
            with open(output_path, "a") as test_energy_lookup_csv:
                csv_writer = csv.writer(test_energy_lookup_csv)

                csv_writer.writerow(["ESA Step", "Energy Bin Lower", "Energy Center", "Energy Bin Upper"])
                csv_writer.writerows([
                    [0, 1, 2, 3],
                    [1, 3, 5, 8],
                    [2, 8, 12, 15]
                ])

            actual_energy_lookup = EnergyLookup.read_from_csv(output_path)

            np.testing.assert_array_equal(actual_energy_lookup.bin_centers, np.array([2, 5, 12]))
            np.testing.assert_array_equal(actual_energy_lookup.bin_centers - actual_energy_lookup.delta_minus,
                                          np.array([1, 3, 8]))
            np.testing.assert_array_equal(actual_energy_lookup.bin_centers + actual_energy_lookup.delta_plus,
                                          np.array([3, 8, 15]))

            actual_energy_indices = actual_energy_lookup.get_energy_index([1, 2, 3, 4, 6, 7, 8, 9, 10])
            np.testing.assert_array_equal(actual_energy_indices, np.array([0, 0, 1, 1, 1, 1, 2, 2, 2]))

    def test_energy_lookup_with_imperfect_logspace_values(self):
        l1a_test_path = get_test_instrument_team_data_path(
            "codice/lo/imap_codice_l1a_lo-nsw-priority_20241110_v002.cdf")

        energies = CDF(str(l1a_test_path))["energy_table"][...]

        actual_energy_lookup = EnergyLookup.from_bin_centers(energies)

        bins = np.column_stack([np.arange(128),
                                actual_energy_lookup.bin_centers + actual_energy_lookup.delta_plus,
                                actual_energy_lookup.bin_centers,
                                actual_energy_lookup.bin_centers - actual_energy_lookup.delta_minus])

        np.savetxt(fname=get_test_data_path("codice/esa-to-energy-per-charge.csv"), X=bins,
                   fmt=["%i", "%f", "%f", "%f"], delimiter=",")

        np.testing.assert_array_equal(actual_energy_lookup.bin_centers, energies)

        for i, energy in enumerate(energies):
            self.assertEqual(i, actual_energy_lookup.get_energy_index(energy))

        lower_edges = energies - actual_energy_lookup.delta_minus
        upper_edges = energies + actual_energy_lookup.delta_plus

        np.testing.assert_array_equal(lower_edges[1:], upper_edges[:-1])

        self.assertAlmostEqual(np.average(np.diff(np.log(energies))), np.average(np.diff(np.log(lower_edges))),
                               places=5)
        self.assertAlmostEqual(np.average(np.diff(np.log(lower_edges))), np.average(np.diff(np.log(upper_edges))),
                               places=5)
