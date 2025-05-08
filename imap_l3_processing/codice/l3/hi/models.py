from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.models import DataProduct, DataProductVariable

CODICE_HI_NUM_L2_PRIORITIES = 6


@dataclass
class PriorityEventL2:
    data_quality: ndarray
    multi_flag: ndarray
    number_of_events: ndarray
    ssd_energy: ndarray
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
                    data_quality=cdf[f"p{p}_data_quality"][...],
                    multi_flag=cdf[f"p{p}_multi_flag"][...],
                    number_of_events=cdf[f"p{p}_num_events"][...],
                    ssd_energy=cdf[f"p{p}_ssd_energy"][...],
                    ssd_id=cdf[f"p{p}_ssd_id"][...],
                    spin_angle=cdf[f"p{p}_spin_sector"][...],
                    spin_number=cdf[f"p{p}_spin_number"][...],
                    time_of_flight=cdf[f"p{p}_tof"][...],
                    type=cdf[f"p{p}_type"][...],
                )
                priority_events.append(priority_event)

            return cls(epoch, epoch_delta_plus, priority_events)


EPOCH_VAR_NAME = "epoch"
DATA_QUALITY_VAR_NAME = "data_quality"
MULTI_FLAG_VAR_NAME = "multi_flag"
NUM_EVENTS_VAR_NAME = "num_events"
SSD_ENERGY_VAR_NAME = "ssd_energy"
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
ENERGY_VAR_NAME = "energy"
ENERGY_DELTA_PLUS_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_VAR_NAME = "energy_delta_minus"
PITCH_ANGLE_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_VAR_NAME = "pitch_angle_delta"
GYROPHASE_VAR_NAME = "gyrophase"
GYROPHASE_DELTA_VAR_NAME = "gyrophase_delta"
H_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "h_intensity_by_pitch_angle"
H_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "h_intensity_by_pitch_angle_and_gyrophase"
HE4_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "he4_intensity_by_pitch_angle"
HE4_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "he4_intensity_by_pitch_angle_and_gyrophase"
O_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "o_intensity_by_pitch_angle"
O_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "o_intensity_by_pitch_angle_and_gyrophase"
FE_INTENSITY_BY_PITCH_ANGLE_VAR_NAME = "fe_intensity_by_pitch_angle"
FE_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME = "fe_intensity_by_pitch_angle_and_gyrophase"


@dataclass
class CodiceL3HiDirectEvents(DataProduct):
    epoch: ndarray
    epoch_delta: ndarray
    data_quality: ndarray
    multi_flag: ndarray
    num_events: ndarray
    ssd_energy: ndarray
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
    energy: ndarray
    energy_delta_plus: ndarray
    energy_delta_minus: ndarray
    pitch_angle: ndarray
    pitch_angle_delta: ndarray
    gyrophase: ndarray
    gyrophase_delta: ndarray
    h_intensity_by_pitch_angle: ndarray
    h_intensity_by_pitch_angle_and_gyrophase: ndarray
    he4_intensity_by_pitch_angle: ndarray
    he4_intensity_by_pitch_angle_and_gyrophase: ndarray
    o_intensity_by_pitch_angle: ndarray
    o_intensity_by_pitch_angle_and_gyrophase: ndarray
    fe_intensity_by_pitch_angle: ndarray
    fe_intensity_by_pitch_angle_and_gyrophase: ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.energy_delta_minus),
            DataProductVariable(PITCH_ANGLE_VAR_NAME, self.pitch_angle),
            DataProductVariable(PITCH_ANGLE_DELTA_VAR_NAME, self.pitch_angle_delta),
            DataProductVariable(GYROPHASE_VAR_NAME, self.gyrophase),
            DataProductVariable(GYROPHASE_DELTA_VAR_NAME, self.gyrophase_delta),
            DataProductVariable(H_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.h_intensity_by_pitch_angle),
            DataProductVariable(H_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.h_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(HE4_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.he4_intensity_by_pitch_angle),
            DataProductVariable(HE4_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.he4_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(O_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.o_intensity_by_pitch_angle),
            DataProductVariable(O_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.o_intensity_by_pitch_angle_and_gyrophase),
            DataProductVariable(FE_INTENSITY_BY_PITCH_ANGLE_VAR_NAME, self.fe_intensity_by_pitch_angle),
            DataProductVariable(FE_INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_VAR_NAME,
                                self.fe_intensity_by_pitch_angle_and_gyrophase),
        ]


@dataclass
class CodiceHiL2SectoredIntensitiesData:
    epoch: ndarray
    epoch_delta: ndarray
    h_intensities: ndarray
    he4_intensities: ndarray
    o_intensities: ndarray
    fe_intensities: ndarray
    ssd_id: ndarray
    spin_sector: ndarray
    energy: ndarray
    energy_delta_minus: ndarray
    energy_delta_plus: ndarray

    @classmethod
    def read_from_cdf(cls, l2_sectored_intensities_cdf):
        with CDF(str(l2_sectored_intensities_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                epoch_delta=cdf["epoch_delta"][...],
                h_intensities=cdf["h_intensities"][...],
                he4_intensities=cdf["he4_intensities"][...],
                o_intensities=cdf["o_intensities"][...],
                fe_intensities=cdf["fe_intensities"][...],
                ssd_id=cdf["ssd_id"][...],
                spin_sector=cdf["spin_sector"][...],
                energy=cdf["energy"][...],
                energy_delta_minus=cdf["energy_delta_minus"][...],
                energy_delta_plus=cdf["energy_delta_plus"][...],
            )
