from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.models import DataProduct, DataProductVariable

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
CHARGE_VAR_NAME = "charge"
ENERGY_VAR_NAME = "energy"
ENERGY_AT_DETECTOR_VAR_NAME = "energy_at_detector"
E_DELTA_VAR_NAME = "delta_e"
E_PRIME_VAR_NAME = "e_prime"
DETECTED_RANGE_VAR_NAME = "range"
PARTICLE_ID_VAR_NAME = "particle_id"
PRIORITY_BUFFER_NUMBER_VAR_NAME = "priority_buffer_number"
LATENCY_VAR_NAME = "latency"
STIM_TAG_VAR_NAME = "stim_tag"
LONG_EVENT_FLAG_VAR_NAME = "long_event_flag"
HAZ_TAG_VAR_NAME = "haz_tag"
A_B_SIDE_VAR_NAME = "side"
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
    h: np.ndarray[float]
    he4: np.ndarray[float]
    cno: np.ndarray[float]
    nemgsi: np.ndarray[float]
    fe: np.ndarray[float]

    delta_minus_cno: np.ndarray[float]
    delta_minus_he4: np.ndarray[float]
    delta_minus_h: np.ndarray[float]
    delta_minus_fe: np.ndarray[float]
    delta_minus_nemgsi: np.ndarray[float]
    delta_plus_cno: np.ndarray[float]
    delta_plus_he4: np.ndarray[float]
    delta_plus_h: np.ndarray[float]
    delta_plus_fe: np.ndarray[float]
    delta_plus_nemgsi: np.ndarray[float]
    cno_energy: np.ndarray[float]
    cno_energy_delta_plus: np.ndarray[float]
    cno_energy_delta_minus: np.ndarray[float]
    fe_energy: np.ndarray[float]
    fe_energy_delta_plus: np.ndarray[float]
    fe_energy_delta_minus: np.ndarray[float]
    h_energy: np.ndarray[float]
    h_energy_delta_plus: np.ndarray[float]
    h_energy_delta_minus: np.ndarray[float]
    he4_energy: np.ndarray[float]
    he4_energy_delta_plus: np.ndarray[float]
    he4_energy_delta_minus: np.ndarray[float]
    nemgsi_energy: np.ndarray[float]
    nemgsi_energy_delta_plus: np.ndarray[float]
    nemgsi_energy_delta_minus: np.ndarray[float]


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
            DataProductVariable(EPOCH_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, 1, record_varying=False,
                                cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(CHARGE_VAR_NAME, self.charge, cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable(ENERGY_VAR_NAME, self.energy, cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable(E_DELTA_VAR_NAME, self.e_delta, cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable(E_PRIME_VAR_NAME, self.e_prime, cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable(DETECTED_RANGE_VAR_NAME, self.detected_range, cdf_data_type=pycdf.const.CDF_INT4),
            DataProductVariable(PARTICLE_ID_VAR_NAME, self.particle_id, cdf_data_type=pycdf.const.CDF_UINT2),
            DataProductVariable(PRIORITY_BUFFER_NUMBER_VAR_NAME, self.priority_buffer_number,
                                cdf_data_type=pycdf.const.CDF_UINT1),
            DataProductVariable(LATENCY_VAR_NAME, self.latency,
                                cdf_data_type=pycdf.const.CDF_UINT1),
            DataProductVariable(STIM_TAG_VAR_NAME, self.stim_tag, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(LONG_EVENT_FLAG_VAR_NAME, self.long_event_flag, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(HAZ_TAG_VAR_NAME, self.haz_tag, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(A_B_SIDE_VAR_NAME, self.a_b_side, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(HAS_UNREAD_FLAG_VAR_NAME, self.has_unread_adcs, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(CULLING_FLAG_VAR_NAME, self.culling_flag, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(PHA_VALUE_VAR_NAME, self.pha_value,
                                cdf_data_type=pycdf.const.CDF_UINT2),
            DataProductVariable(ENERGY_AT_DETECTOR_VAR_NAME, self.energy_at_detector,
                                cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable(IS_LOW_GAIN_VAR_NAME, self.is_low_gain, cdf_data_type=pycdf.const.CDF_INT1),
            DataProductVariable(DETECTOR_FLAGS_VAR_NAME, self.detector_flags, cdf_data_type=pycdf.const.CDF_UINT2),
            DataProductVariable(DEINDEX_VAR_NAME, self.deindex,
                                cdf_data_type=pycdf.const.CDF_UINT2),
            DataProductVariable(EPINDEX_VAR_NAME, self.epindex,
                                cdf_data_type=pycdf.const.CDF_UINT2),
            DataProductVariable(STIM_GAIN_VAR_NAME, self.stim_gain, cdf_data_type=pycdf.const.CDF_UINT1),
            DataProductVariable(A_L_STIM_VAR_NAME, self.a_l_stim, cdf_data_type=pycdf.const.CDF_UINT1),
            DataProductVariable(STIM_STEP_VAR_NAME, self.stim_step,
                                cdf_data_type=pycdf.const.CDF_UINT1),
            DataProductVariable(DAC_VALUE_VAR_NAME, self.dac_value,
                                cdf_data_type=pycdf.const.CDF_UINT2)
        ]


@dataclass
class HitL1Data:
    epoch: np.ndarray[datetime]
    event_binary: np.ndarray[str]

    @classmethod
    def read_from_cdf(cls, cdf_file_path: Union[Path, str]):
        cdf = CDF(str(cdf_file_path))

        return cls(cdf["epoch"], cdf["pha_raw"][...])
