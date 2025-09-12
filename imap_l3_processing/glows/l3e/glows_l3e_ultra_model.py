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
SPACECRAFT_RADIUS_VAR_NAME = "spacecraft_radius"
SPACECRAFT_LONGITUDE_VAR_NAME = "spacecraft_longitude"
SPACECRAFT_LATITUDE_VAR_NAME = "spacecraft_latitude"
SPACECRAFT_VELOCITY_X_VAR_NAME = "spacecraft_velocity_x"
SPACECRAFT_VELOCITY_Y_VAR_NAME = "spacecraft_velocity_y"
SPACECRAFT_VELOCITY_Z_VAR_NAME = "spacecraft_velocity_z"


@dataclass
class GlowsL3EUltraData(DataProduct):
    epoch: np.ndarray[datetime]
    energy: np.ndarray
    healpix_index: np.ndarray
    probability_of_survival: np.ndarray
    spin_axis_lat: np.ndarray
    spin_axis_lon: np.ndarray
    program_version: np.ndarray
    spacecraft_radius: np.ndarray
    spacecraft_latitude: np.ndarray
    spacecraft_longitude: np.ndarray
    spacecraft_velocity_x: np.ndarray
    spacecraft_velocity_y: np.ndarray
    spacecraft_velocity_z: np.ndarray

    @classmethod
    def convert_dat_to_glows_l3e_ul_product(cls, input_metadata: InputMetadata, file_path: Path,
                                            epoch: datetime,
                                            args: GlowsL3eCallArguments):
        with open(file_path) as input_data:
            lines = input_data.readlines()

            energy_line = [line for line in lines if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

            code_version_line = [line for line in lines if line.startswith("# code version")]
            code_version = np.array([code_version_line[0].split(',')[0][14:].strip()])

        data_table = np.loadtxt(file_path, skiprows=200, dtype=np.float64)

        healpix_indexes = np.arange(0, 3072)

        existing_healpix = data_table[:, 0]
        probability_of_survival = data_table[:, 3:]

        probability_of_survival_to_return = np.full((len(energies), len(healpix_indexes)), np.nan, dtype=float)

        for healpix, prob_sur in zip(existing_healpix, probability_of_survival):
            probability_of_survival_to_return[:, int(healpix)] = prob_sur

        transposed_prob_sur = np.array([probability_of_survival_to_return])

        return cls(input_metadata,
            epoch=np.array([epoch]),
            energy=energies,
            healpix_index=healpix_indexes,
            probability_of_survival=transposed_prob_sur,
            spin_axis_lat=np.array([args.spin_axis_latitude]),
            spin_axis_lon=np.array([args.spin_axis_longitude]),
            program_version=np.array([code_version]),
            spacecraft_radius=np.array([args.spacecraft_radius]),
            spacecraft_longitude=np.array([args.spacecraft_longitude]),
            spacecraft_latitude=np.array([args.spacecraft_latitude]),
            spacecraft_velocity_x=np.array([args.spacecraft_velocity_x]),
            spacecraft_velocity_y=np.array([args.spacecraft_velocity_y]),
            spacecraft_velocity_z=np.array([args.spacecraft_velocity_z]),
        )

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
            DataProductVariable(PROGRAM_VERSION_VAR_NAME, self.program_version),
            DataProductVariable(SPACECRAFT_RADIUS_VAR_NAME, self.spacecraft_radius),
            DataProductVariable(SPACECRAFT_LATITUDE_VAR_NAME, self.spacecraft_latitude),
            DataProductVariable(SPACECRAFT_LONGITUDE_VAR_NAME, self.spacecraft_longitude),
            DataProductVariable(SPACECRAFT_VELOCITY_X_VAR_NAME, self.spacecraft_velocity_x),
            DataProductVariable(SPACECRAFT_VELOCITY_Y_VAR_NAME, self.spacecraft_velocity_y),
            DataProductVariable(SPACECRAFT_VELOCITY_Z_VAR_NAME, self.spacecraft_velocity_z),
        ]
