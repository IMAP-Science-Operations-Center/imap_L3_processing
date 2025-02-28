from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.models import DataProduct, DataProductVariable

EPOCH_VAR_NAME = "epoch"
CHARGE_VAR_NAME = "charge"
ENERGY_VAR_NAME = "energy"
ENERGY_AT_DETECTOR_VAR_NAME = "energy_at_detector"
E_DELTA_VAR_NAME = "e_delta"
E_PRIME_VAR_NAME = "e_prime"
DETECTED_RANGE_VAR_NAME = "detected_range"
PARTICLE_ID_VAR_NAME = "particle_id"
PRIORITY_BUFFER_NUMBER_VAR_NAME = "priority_buffer_number"
LATENCY_VAR_NAME = "latency"
STIM_TAG_VAR_NAME = "stim_tag"
LONG_EVENT_FLAG_VAR_NAME = "long_event_flag"
HAZ_TAG_VAR_NAME = "haz_tag"
A_B_SIDE_VAR_NAME = "a_b_side"
HAS_UNREAD_FLAG_VAR_NAME = "has_unread_flag"
CULLING_FLAG_VAR_NAME = "culling_flag"
PHA_VALUE_VAR_NAME = "pha_value"
DETECTOR_ADDRESS_VAR_NAME = "detector_address"
IS_LOW_GAIN_VAR_NAME = "is_low_gain"
LAST_PHA_VAR_NAME = "last_pha"
DETECTOR_FLAGS_VAR_NAME = "detector_flags"
DEINDEX_VAR_NAME = "deindex"
EPINDEX_VAR_NAME = "epindex"
STIM_GAIN_VAR_NAME = "stim_gain"
A_L_STIM_VAR_NAME = "a_l_stim"
STIM_STEP_VAR_NAME = "stim_step"
DAC_VALUE_VAR_NAME = "dac_value"


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


@dataclass
class HitDirectEventDataProduct(DataProduct):
    epoch: np.ndarray[datetime]
    charge: np.ndarray[float]
    energy: np.ndarray[float]
    e_delta: np.ndarray[float]
    e_prime: np.ndarray[float]
    detected_range: np.ndarray[int]
    particle_id: np.ndarray[int]
    priority_buffer_number: np.ndarray[int]
    latency: np.ndarray[int]
    stim_tag: np.ndarray[bool]
    long_event_flag: np.ndarray[bool]
    haz_tag: np.ndarray[bool]
    a_b_side: np.ndarray[bool]
    has_unread_adcs: np.ndarray[bool]
    culling_flag: np.ndarray[bool]
    pha_value: np.ndarray[int]
    energy_at_detector: np.ndarray[float]
    detector_address: np.ndarray[int]
    is_low_gain: np.ndarray[bool]
    detector_flags: np.ndarray[int]
    deindex: np.ndarray[int]
    epindex: np.ndarray[int]
    stim_gain: np.ndarray[float]
    a_l_stim: np.ndarray[bool]
    stim_step: np.ndarray[int]
    dac_value: np.ndarray[int]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(CHARGE_VAR_NAME, self.charge),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),

            DataProductVariable(E_DELTA_VAR_NAME, self.e_delta),
            DataProductVariable(E_PRIME_VAR_NAME, self.e_prime),
            DataProductVariable(DETECTED_RANGE_VAR_NAME, self.detected_range),

            DataProductVariable(PARTICLE_ID_VAR_NAME, self.particle_id),
            DataProductVariable(PRIORITY_BUFFER_NUMBER_VAR_NAME, self.priority_buffer_number),
            DataProductVariable(LATENCY_VAR_NAME, self.latency),

            DataProductVariable(STIM_TAG_VAR_NAME, self.stim_tag),
            DataProductVariable(LONG_EVENT_FLAG_VAR_NAME, self.long_event_flag),
            DataProductVariable(HAZ_TAG_VAR_NAME, self.haz_tag),
            DataProductVariable(A_B_SIDE_VAR_NAME, self.a_b_side),
            DataProductVariable(HAS_UNREAD_FLAG_VAR_NAME, self.has_unread_adcs),
            DataProductVariable(CULLING_FLAG_VAR_NAME, self.culling_flag),

            DataProductVariable(PHA_VALUE_VAR_NAME, self.pha_value),

            DataProductVariable(ENERGY_AT_DETECTOR_VAR_NAME, self.energy_at_detector),

            DataProductVariable(DETECTOR_ADDRESS_VAR_NAME, self.detector_address),

            DataProductVariable(IS_LOW_GAIN_VAR_NAME, self.is_low_gain),

            DataProductVariable(DETECTOR_FLAGS_VAR_NAME, self.detector_flags),
            DataProductVariable(DEINDEX_VAR_NAME, self.deindex),
            DataProductVariable(EPINDEX_VAR_NAME, self.epindex),
            DataProductVariable(STIM_GAIN_VAR_NAME, self.stim_gain),
            DataProductVariable(A_L_STIM_VAR_NAME, self.a_l_stim),
            DataProductVariable(STIM_STEP_VAR_NAME, self.stim_step),
            DataProductVariable(DAC_VALUE_VAR_NAME, self.dac_value)
        ]


@dataclass
class HitL1Data:
    epoch: np.ndarray[datetime]
    event_binary: np.ndarray[str]

    @classmethod
    def read_from_cdf(cls, cdf_file_path: Union[Path, str]):
        cdf = CDF(str(cdf_file_path))

        return cls(cdf["epoch"], cdf["pha_raw"][...])
