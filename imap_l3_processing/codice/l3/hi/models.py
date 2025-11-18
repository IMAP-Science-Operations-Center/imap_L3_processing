from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable, read_variable_and_mask_fill_values
from imap_l3_processing.models import DataProduct, DataProductVariable

CODICE_HI_NUM_L2_PRIORITIES = 6


@dataclass
class PriorityEventL2:
    data_quality: ndarray
    multi_flag: ndarray
    number_of_events: ndarray
    ssd_energy: ndarray
    ssd_energy_plus: ndarray
    ssd_energy_minus: ndarray
    ssd_id: ndarray
    spin_angle: ndarray
    spin_number: ndarray
    time_of_flight: ndarray
    type: ndarray


@dataclass
class CodiceL2HiData:
    epoch: ndarray
    epoch_delta_plus: ndarray
    priority_events: list[PriorityEventL2]

    @classmethod
    def read_from_cdf(cls, filename):
        with CDF(str(filename)) as cdf:
            epoch = cdf["epoch"][...]
            epoch_delta_plus = cdf['epoch_delta_plus'][...]

            priority_events = []
            for p in range(CODICE_HI_NUM_L2_PRIORITIES):
                priority_event = PriorityEventL2(
                    data_quality=read_variable_and_mask_fill_values(cdf[f"p{p}_data_quality"]),
                    multi_flag=read_variable_and_mask_fill_values(cdf[f"p{p}_multi_flag"]),
                    number_of_events=read_variable_and_mask_fill_values(cdf[f"p{p}_num_events"]),
                    ssd_energy=read_numeric_variable(cdf[f"p{p}_ssd_energy"]),
                    ssd_energy_plus=cdf[f"p{p}_ssd_energy_plus"][...],
                    ssd_energy_minus=cdf[f"p{p}_ssd_energy_minus"][...],
                    ssd_id=read_numeric_variable(cdf[f"p{p}_ssd_id"]),
                    spin_angle=read_numeric_variable(cdf[f"p{p}_spin_sector"]),
                    spin_number=read_variable_and_mask_fill_values(cdf[f"p{p}_spin_number"]),
                    time_of_flight=read_numeric_variable(cdf[f"p{p}_tof"]),
                    type=read_variable_and_mask_fill_values(cdf[f"p{p}_type"]),
                )
                priority_events.append(priority_event)

            return cls(epoch, epoch_delta_plus, priority_events)


EPOCH_VAR_NAME = "epoch"
DATA_QUALITY_VAR_NAME = "data_quality"
MULTI_FLAG_VAR_NAME = "multi_flag"
NUM_EVENTS_VAR_NAME = "num_events"
SSD_ENERGY_VAR_NAME = "ssd_energy"
SSD_ENERGY_PLUS_VAR_NAME = "ssd_energy_plus"
SSD_ENERGY_MINUS_VAR_NAME = "ssd_energy_minus"
SSD_ID_VAR_NAME = "ssd_id"
SPIN_ANGLE_VAR_NAME = "spin_angle"
SPIN_NUMBER_VAR_NAME = "spin_number"
TOF_VAR_NAME = "tof"
TYPE_VAR_NAME = "type"
ENERGY_PER_NUC_LOWER_VAR_NAME = "energy_per_nuc_lower"
ENERGY_PER_NUC_VAR_NAME = "energy_per_nuc"
ENERGY_PER_NUC_UPPER_VAR_NAME = "energy_per_nuc_upper"
ESTIMATED_MASS_LOWER_VAR_NAME = "estimated_mass_lower"
ESTIMATED_MASS_VAR_NAME = "estimated_mass"
ESTIMATED_MASS_UPPER_VAR_NAME = "estimated_mass_upper"
PRIORITY_INDEX_VAR_NAME = "priority_index"
EVENT_INDEX_VAR_NAME = "event_index"
PRIORITY_INDEX_LABEL_VAR_NAME = "priority_index_label"
EVENT_INDEX_LABEL_VAR_NAME = "event_index_label"

EPOCH_DELTA_VAR_NAME = "epoch_delta"
ENERGY_H_VAR_NAME = "energy_h"
ENERGY_H_PLUS_VAR_NAME = "energy_h_plus"
ENERGY_H_MINUS_VAR_NAME = "energy_h_minus"
ENERGY_CNO_VAR_NAME = "energy_cno"
ENERGY_CNO_PLUS_VAR_NAME = "energy_cno_plus"
ENERGY_CNO_MINUS_VAR_NAME = "energy_cno_minus"
ENERGY_FE_VAR_NAME = "energy_fe"
ENERGY_FE_PLUS_VAR_NAME = "energy_fe_plus"
ENERGY_FE_MINUS_VAR_NAME = "energy_fe_minus"
ENERGY_HE3HE4_VAR_NAME = "energy_he3he4"
ENERGY_HE3HE4_PLUS_VAR_NAME = "energy_he3he4_plus"
ENERGY_HE3HE4_MINUS_VAR_NAME = "energy_he3he4_minus"
PITCH_ANGLE_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_VAR_NAME = "pitch_angle_delta"
GYROPHASE_VAR_NAME = "gyrophase"
GYROPHASE_DELTA_VAR_NAME = "gyrophase_delta"
H_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "h_intensity_by_pitch_angle"
H_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "h_intensity_by_pitch_angle_and_gyrophase"
HE3HE4_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "he3he4_intensity_by_pitch_angle"
HE3HE4_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "he3he4_intensity_by_pitch_angle_and_gyrophase"
CNO_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "cno_intensity_by_pitch_angle"
CNO_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "cno_intensity_by_pitch_angle_and_gyrophase"
FE_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "fe_intensity_by_pitch_angle"
FE_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "fe_intensity_by_pitch_angle_and_gyrophase"
ENERGY_H_LABEL_VAR_NAME = "energy_h_label"
ENERGY_CNO_LABEL_VAR_NAME = "energy_cno_label"
ENERGY_FE_LABEL_VAR_NAME = "energy_fe_label"
ENERGY_HE3HE4_LABEL_VAR_NAME = "energy_he3he4_label"
PITCH_ANGLE_LABEL_VAR_NAME = "pitch_angle_label"
GYROPHASE_LABEL_VAR_NAME = "gyrophase_label"


@dataclass
class CodiceL3HiDirectEvents(DataProduct):
    epoch: ndarray
    epoch_delta: ndarray
    data_quality: ndarray
    multi_flag: ndarray
    num_events: ndarray
    ssd_energy: ndarray
    ssd_energy_plus: ndarray
    ssd_energy_minus: ndarray
    ssd_id: ndarray
    spin_angle: ndarray
    spin_number: ndarray
    tof: ndarray
    type: ndarray
    energy_per_nuc: ndarray
    estimated_mass: ndarray
    priority_index: ndarray = field(init=False)
    event_index: ndarray = field(init=False)
    priority_index_label: ndarray = field(init=False)
    event_index_label: ndarray = field(init=False)

    def __post_init__(self):
        self.priority_index = np.arange(CODICE_HI_NUM_L2_PRIORITIES)
        self.event_index = np.arange(self.ssd_id.shape[-1])
        self.priority_index_label = np.array([str(i) for i in range(CODICE_HI_NUM_L2_PRIORITIES)])
        self.event_index_label = np.array([str(i) for i in range(len(self.event_index))])

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(DATA_QUALITY_VAR_NAME, self.data_quality),
            DataProductVariable(MULTI_FLAG_VAR_NAME, self.multi_flag),
            DataProductVariable(NUM_EVENTS_VAR_NAME, self.num_events),
            DataProductVariable(SSD_ENERGY_VAR_NAME, self.ssd_energy),
            DataProductVariable(SSD_ENERGY_PLUS_VAR_NAME, self.ssd_energy_plus),
            DataProductVariable(SSD_ENERGY_MINUS_VAR_NAME, self.ssd_energy_minus),
            DataProductVariable(SSD_ID_VAR_NAME, self.ssd_id),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(SPIN_NUMBER_VAR_NAME, self.spin_number),
            DataProductVariable(TOF_VAR_NAME, self.tof),
            DataProductVariable(TYPE_VAR_NAME, self.type),
            DataProductVariable(ENERGY_PER_NUC_VAR_NAME, self.energy_per_nuc),
            DataProductVariable(ESTIMATED_MASS_VAR_NAME, self.estimated_mass),
            DataProductVariable(PRIORITY_INDEX_VAR_NAME, self.priority_index),
            DataProductVariable(EVENT_INDEX_VAR_NAME, self.event_index),
            DataProductVariable(PRIORITY_INDEX_LABEL_VAR_NAME, self.priority_index_label),
            DataProductVariable(EVENT_INDEX_LABEL_VAR_NAME, self.event_index_label),
        ]


@dataclass
class CodiceHiL3PitchAngleDataProduct(DataProduct):
    epoch: ndarray
    epoch_delta: ndarray
    energy_h: ndarray
    energy_h_plus: ndarray
    energy_h_minus: ndarray
    energy_cno: ndarray
    energy_cno_plus: ndarray
    energy_cno_minus: ndarray
    energy_fe: ndarray
    energy_fe_plus: ndarray
    energy_fe_minus: ndarray
    energy_he3he4: ndarray
    energy_he3he4_plus: ndarray
    energy_he3he4_minus: ndarray
    pitch_angle: ndarray
    pitch_angle_delta: ndarray
    gyrophase: ndarray
    gyrophase_delta: ndarray
    h_intensity_by_pitch_angle: ndarray
    h_intensity_by_pitch_angle_and_gyrophase: ndarray
    he3he4_intensity_by_pitch_angle: ndarray
    he3he4_intensity_by_pitch_angle_and_gyrophase: ndarray
    cno_intensity_by_pitch_angle: ndarray
    cno_intensity_by_pitch_angle_and_gyrophase: ndarray
    fe_intensity_by_pitch_angle: ndarray
    fe_intensity_by_pitch_angle_and_gyrophase: ndarray

    energy_h_label: ndarray = field(init=False)
    energy_cno_label: ndarray = field(init=False)
    energy_fe_label: ndarray = field(init=False)
    energy_he3he4_label: ndarray = field(init=False)
    pitch_angle_label: ndarray = field(init=False)
    gyrophase_label: ndarray = field(init=False)

    def __post_init__(self):
        self.energy_h_label = self.energy_h.astype(str)
        self.energy_cno_label = self.energy_cno.astype(str)
        self.energy_fe_label = self.energy_fe.astype(str)
        self.energy_he3he4_label = self.energy_he3he4.astype(str)
        self.pitch_angle_label = self.pitch_angle.astype(str)
        self.gyrophase_label = self.gyrophase.astype(str)

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, np.array([t.total_seconds() for t in self.epoch_delta]) * 1e9),
            DataProductVariable(ENERGY_H_VAR_NAME, self.energy_h),
            DataProductVariable(ENERGY_H_PLUS_VAR_NAME, self.energy_h_plus),
            DataProductVariable(ENERGY_H_MINUS_VAR_NAME, self.energy_h_minus),
            DataProductVariable(ENERGY_CNO_VAR_NAME, self.energy_cno),
            DataProductVariable(ENERGY_CNO_PLUS_VAR_NAME, self.energy_cno_plus),
            DataProductVariable(ENERGY_CNO_MINUS_VAR_NAME, self.energy_cno_minus),
            DataProductVariable(ENERGY_FE_VAR_NAME, self.energy_fe),
            DataProductVariable(ENERGY_FE_PLUS_VAR_NAME, self.energy_fe_plus),
            DataProductVariable(ENERGY_FE_MINUS_VAR_NAME, self.energy_fe_minus),
            DataProductVariable(ENERGY_HE3HE4_VAR_NAME, self.energy_he3he4),
            DataProductVariable(ENERGY_HE3HE4_PLUS_VAR_NAME, self.energy_he3he4_plus),
            DataProductVariable(ENERGY_HE3HE4_MINUS_VAR_NAME, self.energy_he3he4_minus),
            DataProductVariable(PITCH_ANGLE_VAR_NAME, self.pitch_angle),
            DataProductVariable(PITCH_ANGLE_DELTA_VAR_NAME, self.pitch_angle_delta),
            DataProductVariable(GYROPHASE_VAR_NAME, self.gyrophase),
            DataProductVariable(GYROPHASE_DELTA_VAR_NAME, self.gyrophase_delta),
            DataProductVariable(H_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.h_intensity_by_pitch_angle),
            DataProductVariable(H_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.h_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(HE3HE4_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.he3he4_intensity_by_pitch_angle),
            DataProductVariable(HE3HE4_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.he3he4_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(CNO_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.cno_intensity_by_pitch_angle),
            DataProductVariable(CNO_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.cno_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(FE_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.fe_intensity_by_pitch_angle),
            DataProductVariable(FE_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.fe_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(ENERGY_H_LABEL_VAR_NAME, self.energy_h_label),
            DataProductVariable(ENERGY_CNO_LABEL_VAR_NAME, self.energy_cno_label),
            DataProductVariable(ENERGY_FE_LABEL_VAR_NAME, self.energy_fe_label),
            DataProductVariable(ENERGY_HE3HE4_LABEL_VAR_NAME, self.energy_he3he4_label),
            DataProductVariable(PITCH_ANGLE_LABEL_VAR_NAME, self.pitch_angle_label),
            DataProductVariable(GYROPHASE_LABEL_VAR_NAME, self.gyrophase_label),
        ]


@dataclass
class CodiceHiL2SectoredIntensitiesData:
    epoch: ndarray
    epoch_delta_plus: ndarray
    spin_angles: ndarray
    elevation_angle: ndarray
    data_quality: ndarray
    h_intensities: ndarray
    energy_h: ndarray
    energy_h_plus: ndarray
    energy_h_minus: ndarray
    cno_intensities: ndarray
    energy_cno: ndarray
    energy_cno_plus: ndarray
    energy_cno_minus: ndarray
    fe_intensities: ndarray
    energy_fe: ndarray
    energy_fe_plus: ndarray
    energy_fe_minus: ndarray
    he3he4_intensities: ndarray
    energy_he3he4: ndarray
    energy_he3he4_plus: ndarray
    energy_he3he4_minus: ndarray

    @classmethod
    def read_from_cdf(cls, l2_sectored_intensities_cdf):
        with CDF(str(l2_sectored_intensities_cdf)) as cdf:
            return cls(epoch=cdf["epoch"][...],
                       epoch_delta_plus=np.array([timedelta(seconds=ns / 1e9) for ns in cdf["epoch_delta_plus"][...]]),
                       spin_angles=cdf['spin_angles'][...],
                       elevation_angle=cdf['elevation_angle'][...],
                       data_quality=cdf['data_quality'][...],
                       h_intensities=read_numeric_variable(cdf['h']),
                       energy_h=cdf['energy_h'][...],
                       energy_h_plus=cdf['energy_h_plus'][...],
                       energy_h_minus=cdf['energy_h_minus'][...],
                       cno_intensities=read_numeric_variable(cdf['cno']),
                       energy_cno=cdf['energy_cno'][...],
                       energy_cno_plus=cdf['energy_cno_plus'][...],
                       energy_cno_minus=cdf['energy_cno_minus'][...],
                       fe_intensities=read_numeric_variable(cdf['fe']),
                       energy_fe=cdf['energy_fe'][...],
                       energy_fe_plus=cdf['energy_fe_plus'][...],
                       energy_fe_minus=cdf['energy_fe_minus'][...],
                       he3he4_intensities=read_numeric_variable(cdf['he3he4']),
                       energy_he3he4=cdf['energy_he3he4'][...],
                       energy_he3he4_plus=cdf['energy_he3he4_plus'][...],
                       energy_he3he4_minus=cdf['energy_he3he4_minus'][...],
                       )
