import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MassPerChargeLookup:
    hplus: float
    heplusplus: float
    cplus4: float
    cplus5: float
    cplus6: float
    oplus5: float
    oplus6: float
    oplus7: float
    oplus8: float
    mg: float
    si: float
    fe_loq: float
    fe_hiq: float

    @classmethod
    def read_from_file(cls, mass_per_charge_lookup_path: Path):
        with open(mass_per_charge_lookup_path) as file:
            mass_per_charge_lookup = csv.reader(file)
            next(mass_per_charge_lookup)
            mass_per_charge = {}
            for entry in mass_per_charge_lookup:
                mass_per_charge[entry[1]] = float(entry[2])

        return cls(
            hplus=mass_per_charge["H+"],
            heplusplus=mass_per_charge["He++"],
            cplus4=mass_per_charge["C+4"],
            cplus5=mass_per_charge["C+5"],
            cplus6=mass_per_charge["C+6"],
            oplus5=mass_per_charge["O+5"],
            oplus6=mass_per_charge["O+6"],
            oplus7=mass_per_charge["O+7"],
            oplus8=mass_per_charge["O+8"],
            mg=mass_per_charge["Mg"],
            si=mass_per_charge["Si"],
            fe_loq=mass_per_charge["Fe (low Q)"],
            fe_hiq=mass_per_charge["Fe (high Q)"],
        )
