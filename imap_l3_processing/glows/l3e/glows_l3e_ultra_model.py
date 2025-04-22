import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "probability_of_survival"
LATITUDE_VAR_NAME = "latitude"
LONGITUDE_VAR_NAME = "longitude"
HEALPIX_INDEX_VAR_NAME = "healpix_index"
ENERGY_LABEL_VAR_NAME = "energy_label"
SPIN_ANGLE_LABEL_VAR_NAME = "healpix_index_label"


@dataclass
class GlowsL3EUltraData(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[timedelta]
    energy: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    healpix_index: np.ndarray
    probability_of_survival: np.ndarray

    @classmethod
    def convert_dat_to_glows_l3e_ul_product(cls, input_metadata: InputMetadata, file_path: Path,
                                            epoch: np.ndarray[datetime],
                                            epoch_delta: np.ndarray[timedelta]):
        with open(file_path) as input_data:
            energy_line = [line for line in input_data.readlines() if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

        data_table = np.loadtxt(file_path, skiprows=200)

        healpix_indexes = np.arange(0, 3072)

        existing_healpix = data_table[:, 0]
        existing_latitudes = data_table[:, 1]
        existing_longitudes = data_table[:, 2]
        probability_of_survival = data_table[:, 3:-1]

        latitude_to_return = np.full_like(healpix_indexes, np.nan, dtype=float)
        longitude_to_return = np.full_like(healpix_indexes, np.nan, dtype=float)
        probability_of_survival_to_return = np.full((len(energies), len(healpix_indexes)), np.nan, dtype=float)

        for healpix, latitude, longitude, prob_sur in zip(existing_healpix, existing_latitudes, existing_longitudes,
                                                          probability_of_survival):
            latitude_to_return[int(healpix)] = latitude
            longitude_to_return[int(healpix)] = longitude
            probability_of_survival_to_return[:, int(healpix)] = prob_sur

        transposed_prob_sur = np.array([probability_of_survival_to_return])

        return cls(input_metadata, epoch, epoch_delta.astype('timedelta64[ns]').astype(float), energies,
                   latitude_to_return, longitude_to_return,
                   healpix_indexes, transposed_prob_sur)

    def to_data_product_variables(self) -> list[DataProductVariable]:
        energy_labels = [f"Energy Label {i}" for i in range(1, 21)]
        spin_angle_labels = [f"Heal Pixel Label {i}" for i in range(0, 3072)]
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(LATITUDE_VAR_NAME, self.latitude),
            DataProductVariable(LONGITUDE_VAR_NAME, self.longitude),
            DataProductVariable(HEALPIX_INDEX_VAR_NAME, self.healpix_index),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, energy_labels),
            DataProductVariable(SPIN_ANGLE_LABEL_VAR_NAME, spin_angle_labels)

        ]
