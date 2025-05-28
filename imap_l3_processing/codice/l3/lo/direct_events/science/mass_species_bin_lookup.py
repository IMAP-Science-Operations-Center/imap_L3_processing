from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MassSpeciesBinLookup:
    species: list[str]
    mass_per_charge: list[tuple[float, float]]
    mass_ranges: list[tuple[float, float]]

    @classmethod
    def read_from_csv(cls, path: Path | str):
        loaded_csv = np.genfromtxt(path, delimiter=",", dtype=None, skip_header=1)
        range_data = list(zip(*loaded_csv))
        [species, mpq_min, mpq_max, mass_min, mass_max] = [list(column) for column in range_data]

        return cls(
            species=species,
            mass_per_charge=list(zip(mpq_min, mpq_max)),
            mass_ranges=list(zip(mass_min, mass_max)),
        )

    def get_species(self, mass: float, mass_per_charge: float) -> Optional[str]:
        for i in range(len(self.species)):
            lower_mass, upper_mass = self.mass_ranges[i]
            lower_mass_per_charge, upper_mass_per_charge = self.mass_per_charge[i]
            if lower_mass <= mass < upper_mass and lower_mass_per_charge <= mass_per_charge < upper_mass_per_charge:
                return self.species[i]

    def get_species_index(self, species: str) -> int:
        return self.species.index(species)

    def get_num_species(self):
        return len(self.species)
