from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from imap_processing.models import DataProduct, DataProductVariable


@dataclass
class SweL2Data:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    phase_space_density: np.ndarray
    flux: np.ndarray
    energy: np.ndarray
    inst_el: np.ndarray
    inst_az_spin_sector: np.ndarray


@dataclass
class SwapiL3aProtonData:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    proton_sw_speed: np.ndarray[float]
    proton_sw_clock_angle: np.ndarray[float]
    proton_sw_deflection_angle: np.ndarray[float]


@dataclass
class SweL3Data(DataProduct):
    epoch: np.ndarray
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    pitch_angle: np.ndarray
    pitch_angle_delta: np.ndarray
    gyrophase: np.ndarray
    gyrophase_delta: np.ndarray
    flux_by_pitch_angle: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return []


class SweConfiguration(TypedDict):
    geometric_fractions: list[float]
    pitch_angle_bins: list[float]
    pitch_angle_delta: list[float]
    energy_bins: list[float]
    energy_delta_plus: list[float]
    energy_delta_minus: list[float]
    energy_bin_low_multiplier: float
    energy_bin_high_multiplier: float
