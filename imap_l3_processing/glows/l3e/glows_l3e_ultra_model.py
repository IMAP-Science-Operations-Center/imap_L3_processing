import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata

EPOCH_CDF_VAR_NAME = "epoch"
ENERGY_VAR_NAME = "energy_grid"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "surv_prob"
LATITUDE_VAR_NAME = "latitude"
LONGITUDE_VAR_NAME = "longitude"
HEALPIX_INDEX_VAR_NAME = "healpix_index"
ENERGY_LABEL_VAR_NAME = "energy_label"
SPIN_ANGLE_LABEL_VAR_NAME = "healpix_index_label"
SPIN_AXIS_LATITUDE_VAR_NAME = "spin_axis_latitude"
SPIN_AXIS_LONGITUDE_VAR_NAME = "spin_axis_longitude"
PROGRAM_VERSION_VAR_NAME = "program_version"


@dataclass
class GlowsL3EUltraData(DataProduct):
    epoch: np.ndarray[datetime]
    energy: np.ndarray
    healpix_index: np.ndarray
    probability_of_survival: np.ndarray
    spin_axis_lat: np.ndarray
    spin_axis_lon: np.ndarray
    program_version: np.ndarray

    @classmethod
    def convert_dat_to_glows_l3e_ul_product(cls, input_metadata: InputMetadata, file_path: Path,
                                            epoch: np.ndarray[datetime],
                                            args: GlowsL3eCallArguments):
        with open(file_path) as input_data:
            lines = input_data.readlines()

            energy_line = [line for line in lines if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

            code_version_line = [line for line in lines if line.startswith("# code version")]
            code_version = np.array([code_version_line[0].split(',')[0][14:].strip()])

        data_table = np.loadtxt(file_path, skiprows=200)

        healpix_indexes = np.arange(0, 3072)

        existing_healpix = data_table[:, 0]
        probability_of_survival = data_table[:, 3:]

        probability_of_survival_to_return = np.full((len(energies), len(healpix_indexes)), np.nan, dtype=float)

        for healpix, prob_sur in zip(existing_healpix, probability_of_survival):
            probability_of_survival_to_return[:, int(healpix)] = prob_sur

        transposed_prob_sur = np.array([probability_of_survival_to_return])

        input_metadata.start_date = epoch[0]

        return cls(input_metadata, epoch, energies,
                   healpix_indexes, transposed_prob_sur, np.array([args.spin_axis_latitude]),
                   np.array([args.spin_axis_longitude]), code_version)

    def to_data_product_variables(self) -> list[DataProductVariable]:
        energy_labels = [f"Energy Label {i}" for i in range(1, 21)]
        spin_angle_labels = [f"Heal Pixel Label {i}" for i in range(0, 3072)]
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(HEALPIX_INDEX_VAR_NAME, self.healpix_index),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, energy_labels),
            DataProductVariable(SPIN_ANGLE_LABEL_VAR_NAME, spin_angle_labels),
            DataProductVariable(SPIN_AXIS_LATITUDE_VAR_NAME, self.spin_axis_lat),
            DataProductVariable(SPIN_AXIS_LONGITUDE_VAR_NAME, self.spin_axis_lon),
            DataProductVariable(PROGRAM_VERSION_VAR_NAME, self.program_version)
        ]
