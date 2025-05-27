from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from imap_l3_processing.codice.l3.lo.constants import CODICE_LO_NUM_AZIMUTH_BINS, CODICE_LO_NUM_ESA_STEPS
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection

SPECIES = TypeVar("SPECIES")
AZIMUTH = TypeVar("AZIMUTH")
ENERGIES = TypeVar("ENERGIES")


@dataclass
class EfficiencyLookup:
    efficiency_data: np.ndarray[(AZIMUTH, ENERGIES)]

    @classmethod
    def read_from_csv(cls, path: Path) -> EfficiencyLookup:
        return cls(efficiency_data=np.loadtxt(path, delimiter=",", skiprows=1).T)

    @classmethod
    def read_from_zip(cls, zip_path: Path, mass_species_bin_lut: MassSpeciesBinLookup) -> EfficiencyLookup:
        efficiency_data = np.full(
            (mass_species_bin_lut.get_num_species(), CODICE_LO_NUM_AZIMUTH_BINS, CODICE_LO_NUM_ESA_STEPS), np.nan)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            extracted_path = zip_path.parent / "extracted_efficiency_data"
            zip_file.extractall(extracted_path)

        all_species = mass_species_bin_lut._range_to_species['sw_species'] \
                      + mass_species_bin_lut._range_to_species['nsw_species']
        for species in all_species:
            filepath = extracted_path / f"{species}-efficiency.csv"
            species_efficiency = np.loadtxt(filepath, delimiter=',', skiprows=1)
            species_index = mass_species_bin_lut.get_species_index(
                species, EventDirection.Sunward if species[:2] == 'sw' else EventDirection.NonSunward
            )
            efficiency_data[species_index, ...] = species_efficiency.T
        return cls(efficiency_data)
