from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SpeciesMassRange = namedtuple("SpeciesMassRange",
                              ["lower_mass", "upper_mass", "lower_mass_per_charge", "upper_mass_per_charge"])


@dataclass
class MassSpeciesBinLookup:
    _range_to_species: dict

    @classmethod
    def read_from_csv(cls, path: Path | str):
        with open(path, "r") as csvfile:
            loaded_csv = np.genfromtxt(csvfile, delimiter=",", dtype=None, skip_header=1)
            range_data = list(zip(*loaded_csv))
            species_dict: dict = {
                "species": range_data[0],
                "mass_ranges": list(zip(range_data[5], range_data[6])),
                "mass_per_charge_ranges": list(zip(range_data[3], range_data[4]))
            }

        return cls(species_dict)

    def get_species(self, mass: int, mass_per_charge: int) -> str:
        for i in range(len(self._range_to_species["species"])):
            lower_mass, upper_mass = self._range_to_species["mass_ranges"][i]
            lower_mass_per_charge, upper_mass_per_charge = self._range_to_species["mass_per_charge_ranges"][i]
            if lower_mass <= mass < upper_mass and lower_mass_per_charge <= mass_per_charge < upper_mass_per_charge:
                return self._range_to_species["species"][i]
