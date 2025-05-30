import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata

EPOCH_CDF_VAR_NAME = "epoch"
ENERGY_VAR_NAME = "energy_grid"
SPIN_ANGLE_VAR_NAME = "spin_angle"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "surv_prob"
ENERGY_LABEL_VAR_NAME = "energy_label"
SPIN_ANGLE_LABEL_VAR_NAME = "spin_angle_label"
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
class GlowsL3EHiData(DataProduct):
    epoch: np.ndarray[datetime]
    energy: np.ndarray
    spin_angle: np.ndarray
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
    def convert_dat_to_glows_l3e_hi_product(cls, input_metadata: InputMetadata, file_path: Path,
                                            epoch: np.ndarray[datetime], args: GlowsL3eCallArguments):
        with open(file_path) as input_data:
            lines = input_data.readlines()

            energy_line = [line for line in lines if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

            code_version_line = [line for line in lines if line.startswith("# code version")]
            code_version = np.array([code_version_line[0].split(',')[0][14:].strip()])

        spin_angle_and_survival_probabilities = np.loadtxt(file_path, skiprows=200)
        spin_angles = spin_angle_and_survival_probabilities[:, 0]
        survival_probabilities = np.array([spin_angle_and_survival_probabilities[:, 1:].T])

        input_metadata.start_date = epoch[0]

        return cls(input_metadata, epoch, energies, spin_angles,
                   survival_probabilities, np.array([args.spin_axis_latitude]), np.array([args.spin_axis_longitude]),
                   code_version,
                   spacecraft_radius=args.spacecraft_radius,
                   spacecraft_longitude=args.spacecraft_longitude,
                   spacecraft_latitude=args.spacecraft_latitude,
                   spacecraft_velocity_x=args.spacecraft_velocity_x,
                   spacecraft_velocity_y=args.spacecraft_velocity_y,
                   spacecraft_velocity_z=args.spacecraft_velocity_z,
                   )

    def to_data_product_variables(self) -> list[DataProductVariable]:
        spin_angle_labels = [f"Spin Angle Label {i}" for i in range(1, 361)]
        energy_labels = [f"Energy Label {i}" for i in range(1, 17)]
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
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
