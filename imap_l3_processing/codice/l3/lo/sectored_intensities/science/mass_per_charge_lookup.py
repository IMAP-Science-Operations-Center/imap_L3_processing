import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MassPerChargeLookup:
    mass_per_charge: list[tuple[int, str, float]]

    @classmethod
    def read_from_file(cls, mass_per_charge_lookup_path: Path):
        with open(mass_per_charge_lookup_path) as file:
            mass_per_charge_lookup = csv.reader(file)
            next(mass_per_charge_lookup)
            mass_per_charge = []
            for index, entry in enumerate(mass_per_charge_lookup):
                mass_per_charge.append((int(entry[0]), entry[1], float(entry[2])))

        return cls(mass_per_charge=mass_per_charge)
