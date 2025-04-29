import csv
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

SpeciesMassRange = namedtuple("SpeciesMassRange",
                              ["lower_mass", "upper_mass", "lower_mass_per_charge", "upper_mass_per_charge"])


@dataclass
class MassSpeciesBinLookup:
    _species_ranges: dict[tuple[str, int], SpeciesMassRange]

    @classmethod
    def read_from_csv(cls, path: Path | str):
        with open(path, "r") as csvfile:
            species_dict: dict = {}
            data = csv.reader(csvfile, delimiter=",")
            next(data)
            for row in data:
                species_dict[(row[0], int(row[1]))] = SpeciesMassRange(float(row[5]), float(row[6]), float(row[3]),
                                                                       float(row[4]))

        return cls(species_dict)

    @property
    def h_plus_nsw(self):
        return self._species_ranges[("H+", 0)]

    @property
    def he_plus_plus_nsw(self):
        return self._species_ranges[("He++", 0)]

    @property
    def c_plus_4_nsw(self):
        return self._species_ranges[("C+4", 0)]

    @property
    def c_plus_5_nsw(self):
        return self._species_ranges[("C+5", 0)]

    @property
    def c_plus_6_nsw(self):
        return self._species_ranges[("C+6", 0)]

    @property
    def o_plus_5_nsw(self):
        return self._species_ranges[("O+5", 0)]

    @property
    def o_plus_6_nsw(self):
        return self._species_ranges[("O+6", 0)]

    @property
    def o_plus_7_nsw(self):
        return self._species_ranges[("O+7", 0)]

    @property
    def o_plus_8_nsw(self):
        return self._species_ranges[("O+8", 0)]

    @property
    def ne_nsw(self):
        return self._species_ranges[("Ne", 0)]

    @property
    def mg_nsw(self):
        return self._species_ranges[("Mg", 0)]

    @property
    def fe_lowq_nsw(self):
        return self._species_ranges[("Fe lowQ", 0)]

    @property
    def fe_highq_nsw(self):
        return self._species_ranges[("Fe highQ", 0)]

    @property
    def he_plus_nsw(self):
        return self._species_ranges[("He+ (PUI)", 0)]

    @property
    def cno_plus_nsw(self):
        return self._species_ranges[("CNO+ (PUI)", 0)]

    @property
    def si_nsw(self):
        return self._species_ranges[("Si", 0)]

    @property
    def h_plus_sw(self):
        return self._species_ranges[("H+", 1)]

    @property
    def he_plus_plus_sw(self):
        return self._species_ranges[("He++", 1)]

    @property
    def c_sw(self):
        return self._species_ranges[("C", 1)]

    @property
    def o_sw(self):
        return self._species_ranges[("O", 1)]

    @property
    def ne_sw(self):
        return self._species_ranges[("Ne", 1)]

    @property
    def si_mg_sw(self):
        return self._species_ranges[("Si and Mg", 1)]

    @property
    def fe_sw(self):
        return self._species_ranges[("Fe", 1)]

    @property
    def he_plus_sw(self):
        return self._species_ranges[("He+", 1)]

    @property
    def cno_plus_sw(self):
        return self._species_ranges[("CNO+", 1)]
