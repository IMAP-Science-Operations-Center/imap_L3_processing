from dataclasses import dataclass

import numpy as np

from imap_processing.models import DataProduct


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
class SweL3Data(DataProduct):
    epoch: np.ndarray
    epoch_delta: np.ndarray
    energy: np.ndarray
    pitch_angle: np.ndarray
    pitch_angle_delta: np.ndarray
    gyrophase: np.ndarray
    gyrophase_delta: np.ndarray
