from collections import namedtuple
from dataclasses import dataclass

import numpy as np

EnergyPerNuc = namedtuple("EnergyPerNuc", ["lower_bound", "energy", "upper_bound"])


@dataclass
class TOFLookup:
    energy_per_nuc_dictionary: dict[int, EnergyPerNuc]

    def __getitem__(self, item):
        return self.energy_per_nuc_dictionary[item]

    @classmethod
    def read_from_file(cls, filename):
        with open(filename, 'r') as f:
            tof_lookup = np.loadtxt(f, delimiter=',', skiprows=1, dtype=float)
            energy_per_nuc_dictionary = {tof_bit: EnergyPerNuc(lower_bound, energy, upper_bound)
                                         for tof_bit, (lower_bound, energy, upper_bound)
                                         in enumerate(tof_lookup)}

            return cls(energy_per_nuc_dictionary)
