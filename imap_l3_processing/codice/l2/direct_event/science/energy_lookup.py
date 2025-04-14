from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyLookup:
    energy_lookup_table: np.ndarray
    energy_bin_table: np.ndarray

    def convert_to_mev(self, ssd_id=None, gain_id=None, ssd_energy=None):
        bin_id = self.energy_lookup_table[ssd_energy][ssd_id][gain_id]
        return self.energy_bin_table[bin_id][0], self.energy_bin_table[bin_id][1], self.energy_bin_table[bin_id][2],

    @classmethod
    def from_files(cls, energy_lookup_csv_path, energy_bin_csv_path):
        energy_lookup_table = np.loadtxt(energy_lookup_csv_path, delimiter=",", skiprows=1, dtype=int)
        rows_count, col_count = energy_lookup_table.shape

        energy_lookup_table = np.reshape(energy_lookup_table, (rows_count, int(col_count / 3), 3))
        energy_bin_table = np.loadtxt(energy_bin_csv_path, delimiter=",", skiprows=1)

        return cls(energy_lookup_table, energy_bin_table)
