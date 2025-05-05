import enum
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class EventDirection(enum.Enum):
    Sunward = "sunward"
    NonSunward = "anti-sunward"


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
            range_data = [np.array(column) for column in range_data]

            sw_species_mask = np.logical_not(range_data[1])
            nsw_species_mask = range_data[1].astype(bool)

            mass_range = np.array(list((zip(range_data[5], range_data[6]))))
            mass_per_charge_ranges = np.array(list(zip(range_data[3], range_data[4])))

            species_dict: dict = {
                "sw_species": range_data[0][sw_species_mask],
                "nsw_species": range_data[0][nsw_species_mask],
                "sw_mass_ranges": mass_range[sw_species_mask],
                "nsw_mass_ranges": mass_range[nsw_species_mask],
                "sw_mass_per_charge_ranges": mass_per_charge_ranges[sw_species_mask],
                "nsw_mass_per_charge_ranges": mass_per_charge_ranges[nsw_species_mask],
            }

        return cls(species_dict)

    def get_species(self, mass: int, mass_per_charge: int, event_direction: EventDirection) -> str:
        if event_direction == EventDirection.Sunward:
            species_lookup = self._range_to_species["sw_species"]
            mass_range_lookup = self._range_to_species["sw_mass_ranges"]
            mass_per_charge_lookup = self._range_to_species["sw_mass_per_charge_ranges"]
        elif event_direction == EventDirection.NonSunward:
            species_lookup = self._range_to_species["nsw_species"]
            mass_range_lookup = self._range_to_species["nsw_mass_ranges"]
            mass_per_charge_lookup = self._range_to_species["nsw_mass_per_charge_ranges"]
        else:
            raise NotImplementedError

        for i in range(len(species_lookup)):
            lower_mass, upper_mass = mass_range_lookup[i]
            lower_mass_per_charge, upper_mass_per_charge = mass_per_charge_lookup[i]
            if lower_mass <= mass < upper_mass and lower_mass_per_charge <= mass_per_charge < upper_mass_per_charge:
                return species_lookup[i]

    def get_species_index(self, species: str, event_direction: EventDirection) -> int:
        if event_direction == EventDirection.NonSunward:
            num_sw_species = len(self._range_to_species['sw_species'])
            return np.where(self._range_to_species['nsw_species'] == species)[0] + num_sw_species
        else:
            return np.where(self._range_to_species['sw_species'] == species)[0]
