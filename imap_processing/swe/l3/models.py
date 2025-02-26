from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from imap_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
ENERGY_CDF_VAR_NAME = "energy"
ENERGY_DELTA_PLUS_CDF_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_CDF_VAR_NAME = "energy_delta_minus"
PITCH_ANGLE_CDF_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_CDF_VAR_NAME = "pitch_angle_delta"
FLUX_BY_PITCH_ANGLE_CDF_VAR_NAME = "flux_by_pitch_angle"
PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME = "phase_space_density_by_pitch_angle"
ENERGY_SPECTRUM_CDF_VAR_NAME = "energy_spectrum"


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
    flux_by_pitch_angle: np.ndarray
    phase_space_density_by_pitch_angle: np.ndarray
    energy_spectrum: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, value=self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME,
                                value=[delta.total_seconds() * 1e9 for delta in self.epoch_delta]),
            DataProductVariable(ENERGY_CDF_VAR_NAME, value=self.energy, record_varying=False),
            DataProductVariable(ENERGY_DELTA_PLUS_CDF_VAR_NAME, value=self.energy_delta_plus, record_varying=False),
            DataProductVariable(ENERGY_DELTA_MINUS_CDF_VAR_NAME, value=self.energy_delta_minus, record_varying=False),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, value=self.pitch_angle, record_varying=False),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, value=self.pitch_angle_delta, record_varying=False),
            DataProductVariable(FLUX_BY_PITCH_ANGLE_CDF_VAR_NAME, value=self.flux_by_pitch_angle),
            DataProductVariable(PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME,
                                value=self.phase_space_density_by_pitch_angle),
            DataProductVariable(ENERGY_SPECTRUM_CDF_VAR_NAME,
                                value=self.energy_spectrum),
        ]


class SweConfiguration(TypedDict):
    geometric_fractions: list[float]
    pitch_angle_bins: list[float]
    pitch_angle_delta: list[float]
    energy_bins: list[float]
    energy_delta_plus: list[float]
    energy_delta_minus: list[float]
    energy_bin_low_multiplier: float
    energy_bin_high_multiplier: float
