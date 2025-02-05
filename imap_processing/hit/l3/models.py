from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass
class HitL2Data:
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[timedelta]
    hydrogen: np.ndarray[float]
    helium4: np.ndarray[float]
    CNO: np.ndarray[float]
    NeMgSi: np.ndarray[float]
    iron: np.ndarray[float]

    DELTA_MINUS_CNO: np.ndarray[float]
    DELTA_MINUS_HELIUM4: np.ndarray[float]
    DELTA_MINUS_HYDROGEN: np.ndarray[float]
    DELTA_MINUS_IRON: np.ndarray[float]
    DELTA_MINUS_NEMGSI: np.ndarray[float]
    DELTA_PLUS_CNO: np.ndarray[float]
    DELTA_PLUS_HELIUM4: np.ndarray[float]
    DELTA_PLUS_HYDROGEN: np.ndarray[float]
    DELTA_PLUS_IRON: np.ndarray[float]
    DELTA_PLUS_NEMGSI: np.ndarray[float]
    cno_energy_high: np.ndarray[float]
    cno_energy_idx: np.ndarray[int]
    cno_energy_low: np.ndarray[float]
    fe_energy_high: np.ndarray[float]
    fe_energy_idx: np.ndarray[int]
    fe_energy_low: np.ndarray[float]
    h_energy_high: np.ndarray[float]
    h_energy_idx: np.ndarray[int]
    h_energy_low: np.ndarray[float]
    he4_energy_high: np.ndarray[float]
    he4_energy_idx: np.ndarray[int]
    he4_energy_low: np.ndarray[float]
    nemgsi_energy_high: np.ndarray[float]
    nemgsi_energy_idx: np.ndarray[int]
    nemgsi_energy_low: np.ndarray[float]
