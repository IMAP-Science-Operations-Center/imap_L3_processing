from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.models import DataProduct, DataProductVariable


@dataclass
class PriorityEventL2:
    data_quality: ndarray
    energy_range: ndarray
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
    epochs: ndarray
    priority_event_0: PriorityEventL2
    priority_event_1: PriorityEventL2
    priority_event_2: PriorityEventL2
    priority_event_3: PriorityEventL2
    priority_event_4: PriorityEventL2
    priority_event_5: PriorityEventL2

    names = ["DataQuality",
             "ERGE",
             ("MultiFlag", "multi_flag"),
             ("NumEvents", "number_of_events"),
             ("SSDEnergy", "ssd_energy"),
             ("SSD_ID", "ssd_id"),
             ("SpinAngle", "spin_angle"),
             ("SpinNumber", "spin_number"),
             ("TOF", "tof"),
             ("Type", "type")]

    @property
    def priority_events(self):
        return [self.priority_event_0, self.priority_event_1, self.priority_event_2,
                self.priority_event_3, self.priority_event_4, self.priority_event_5]

    @classmethod
    def read_from_cdf(cls, filename):
        names = [("DataQuality", "data_quality"), ("ERGE", "energy_range"), ("MultiFlag", "multi_flag"),
                 ("NumEvents", "number_of_events"), ("SSDEnergy", "ssd_energy"), ("SSD_ID", "ssd_id"),
                 ("SpinAngle", "spin_angle"), ("SpinNumber", "spin_number"), ("TOF", "time_of_flight"),
                 ("Type", "type")]

        with CDF(filename) as cdf:
            epochs = cdf["epoch"]

            priority_event_data = {}
            priority_events = []
            for p in range(6):
                for cdf_name, class_attribute in names:
                    cdf_variable_name = f"P{p}_{cdf_name}"
                    priority_event_data[class_attribute] = np.array(cdf[cdf_variable_name])

                priority_events.append(PriorityEventL2(**priority_event_data))

            return cls(epochs, *priority_events)


EPOCH_VAR_NAME = "epoch"
P0_DATA_QUALITY_VAR_NAME = "p0_data_quality"
P0_ERGE_VAR_NAME = "p0_erge"
P0_MULTI_FLAG_VAR_NAME = "p0_multi_flag"
P0_NUM_OF_EVENTS_VAR_NAME = "p0_num_of_events"
P0_SSD_ENERGY_VAR_NAME = "p0_ssd_energy"
P0_SSD_ID_VAR_NAME = "p0_ssd_id"
P0_SPIN_ANGLE_VAR_NAME = "p0_spin_angle"
P0_SPIN_NUMBER_VAR_NAME = "p0_spin_number"
P0_TOF_VAR_NAME = "p0_tof"
P0_TYPE_VAR_NAME = "p0_type"
P1_DATA_QUALITY_VAR_NAME = "p1_data_quality"
P1_ERGE_VAR_NAME = "p1_erge"
P1_MULTI_FLAG_VAR_NAME = "p1_multi_flag"
P1_NUM_OF_EVENTS_VAR_NAME = "p1_num_of_events"
P1_SSD_ENERGY_VAR_NAME = "p1_ssd_energy"
P1_SSD_ID_VAR_NAME = "p1_ssd_id"
P1_SPIN_ANGLE_VAR_NAME = "p1_spin_angle"
P1_SPIN_NUMBER_VAR_NAME = "p1_spin_number"
P1_TOF_VAR_NAME = "p1_tof"
P1_TYPE_VAR_NAME = "p1_type"
P2_DATA_QUALITY_VAR_NAME = "p2_data_quality"
P2_ERGE_VAR_NAME = "p2_erge"
P2_MULTI_FLAG_VAR_NAME = "p2_multi_flag"
P2_NUM_OF_EVENTS_VAR_NAME = "p2_num_of_events"
P2_SSD_ENERGY_VAR_NAME = "p2_ssd_energy"
P2_SSD_ID_VAR_NAME = "p2_ssd_id"
P2_SPIN_ANGLE_VAR_NAME = "p2_spin_angle"
P2_SPIN_NUMBER_VAR_NAME = "p2_spin_number"
P2_TOF_VAR_NAME = "p2_tof"
P2_TYPE_VAR_NAME = "p2_type"
P3_DATA_QUALITY_VAR_NAME = "p3_data_quality"
P3_ERGE_VAR_NAME = "p3_erge"
P3_MULTI_FLAG_VAR_NAME = "p3_multi_flag"
P3_NUM_OF_EVENTS_VAR_NAME = "p3_num_of_events"
P3_SSD_ENERGY_VAR_NAME = "p3_ssd_energy"
P3_SSD_ID_VAR_NAME = "p3_ssd_id"
P3_SPIN_ANGLE_VAR_NAME = "p3_spin_angle"
P3_SPIN_NUMBER_VAR_NAME = "p3_spin_number"
P3_TOF_VAR_NAME = "p3_tof"
P3_TYPE_VAR_NAME = "p3_type"
P4_DATA_QUALITY_VAR_NAME = "p4_data_quality"
P4_ERGE_VAR_NAME = "p4_erge"
P4_MULTI_FLAG_VAR_NAME = "p4_multi_flag"
P4_NUM_OF_EVENTS_VAR_NAME = "p4_num_of_events"
P4_SSD_ENERGY_VAR_NAME = "p4_ssd_energy"
P4_SSD_ID_VAR_NAME = "p4_ssd_id"
P4_SPIN_ANGLE_VAR_NAME = "p4_spin_angle"
P4_SPIN_NUMBER_VAR_NAME = "p4_spin_number"
P4_TOF_VAR_NAME = "p4_tof"
P4_TYPE_VAR_NAME = "p4_type"
P5_DATA_QUALITY_VAR_NAME = "p5_data_quality"
P5_ERGE_VAR_NAME = "p5_erge"
P5_MULTI_FLAG_VAR_NAME = "p5_multi_flag"
P5_NUM_OF_EVENTS_VAR_NAME = "p5_num_of_events"
P5_SSD_ENERGY_VAR_NAME = "p5_ssd_energy"
P5_SSD_ID_VAR_NAME = "p5_ssd_id"
P5_SPIN_ANGLE_VAR_NAME = "p5_spin_angle"
P5_SPIN_NUMBER_VAR_NAME = "p5_spin_number"
P5_TOF_VAR_NAME = "p5_tof"
P5_TYPE_VAR_NAME = "p5_type"


@dataclass
class CodiceL3HiDirectEvents(DataProduct):
    epoch: ndarray
    p0_data_quality: ndarray
    p0_erge: ndarray
    p0_multi_flag: ndarray
    p0_num_of_events: ndarray
    p0_ssd_energy: ndarray
    p0_ssd_id: ndarray
    p0_spin_angle: ndarray
    p0_spin_number: ndarray
    p0_tof: ndarray
    p0_type: ndarray
    p0_energy_per_nuc_lower: ndarray
    p0_energy_per_nuc: ndarray
    p0_energy_per_nuc_upper: ndarray
    p0_estimated_mass_lower: ndarray
    p0_estimated_mass: ndarray
    p0_estimated_mass_upper: ndarray
    p1_data_quality: ndarray
    p1_erge: ndarray
    p1_multi_flag: ndarray
    p1_num_of_events: ndarray
    p1_ssd_energy: ndarray
    p1_ssd_id: ndarray
    p1_spin_angle: ndarray
    p1_spin_number: ndarray
    p1_tof: ndarray
    p1_type: ndarray
    p1_energy_per_nuc_lower: ndarray
    p1_energy_per_nuc: ndarray
    p1_energy_per_nuc_upper: ndarray
    p1_estimated_mass_lower: ndarray
    p1_estimated_mass: ndarray
    p1_estimated_mass_upper: ndarray
    p2_data_quality: ndarray
    p2_erge: ndarray
    p2_multi_flag: ndarray
    p2_num_of_events: ndarray
    p2_ssd_energy: ndarray
    p2_ssd_id: ndarray
    p2_spin_angle: ndarray
    p2_spin_number: ndarray
    p2_tof: ndarray
    p2_type: ndarray
    p2_energy_per_nuc_lower: ndarray
    p2_energy_per_nuc: ndarray
    p2_energy_per_nuc_upper: ndarray
    p2_estimated_mass_lower: ndarray
    p2_estimated_mass: ndarray
    p2_estimated_mass_upper: ndarray
    p3_data_quality: ndarray
    p3_erge: ndarray
    p3_multi_flag: ndarray
    p3_num_of_events: ndarray
    p3_ssd_energy: ndarray
    p3_ssd_id: ndarray
    p3_spin_angle: ndarray
    p3_spin_number: ndarray
    p3_tof: ndarray
    p3_type: ndarray
    p3_energy_per_nuc_lower: ndarray
    p3_energy_per_nuc: ndarray
    p3_energy_per_nuc_upper: ndarray
    p3_estimated_mass_lower: ndarray
    p3_estimated_mass: ndarray
    p3_estimated_mass_upper: ndarray
    p4_data_quality: ndarray
    p4_erge: ndarray
    p4_multi_flag: ndarray
    p4_num_of_events: ndarray
    p4_ssd_energy: ndarray
    p4_ssd_id: ndarray
    p4_spin_angle: ndarray
    p4_spin_number: ndarray
    p4_tof: ndarray
    p4_type: ndarray
    p4_energy_per_nuc_lower: ndarray
    p4_energy_per_nuc: ndarray
    p4_energy_per_nuc_upper: ndarray
    p4_estimated_mass_lower: ndarray
    p4_estimated_mass: ndarray
    p4_estimated_mass_upper: ndarray
    p5_data_quality: ndarray
    p5_erge: ndarray
    p5_multi_flag: ndarray
    p5_num_of_events: ndarray
    p5_ssd_energy: ndarray
    p5_ssd_id: ndarray
    p5_spin_angle: ndarray
    p5_spin_number: ndarray
    p5_tof: ndarray
    p5_type: ndarray
    p5_energy_per_nuc_lower: ndarray
    p5_energy_per_nuc: ndarray
    p5_energy_per_nuc_upper: ndarray
    p5_estimated_mass_lower: ndarray
    p5_estimated_mass: ndarray
    p5_estimated_mass_upper: ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(P0_DATA_QUALITY_VAR_NAME, self.p0_data_quality),
            DataProductVariable(P0_ERGE_VAR_NAME, self.p0_erge),
            DataProductVariable(P0_MULTI_FLAG_VAR_NAME, self.p0_multi_flag),
            DataProductVariable(P0_NUM_OF_EVENTS_VAR_NAME, self.p0_num_of_events),
            DataProductVariable(P0_SSD_ENERGY_VAR_NAME, self.p0_ssd_energy),
            DataProductVariable(P0_SSD_ID_VAR_NAME, self.p0_ssd_id),
            DataProductVariable(P0_SPIN_ANGLE_VAR_NAME, self.p0_spin_angle),
            DataProductVariable(P0_SPIN_NUMBER_VAR_NAME, self.p0_spin_number),
            DataProductVariable(P0_TOF_VAR_NAME, self.p0_tof),
            DataProductVariable(P0_TYPE_VAR_NAME, self.p0_type),
            DataProductVariable(P1_DATA_QUALITY_VAR_NAME, self.p1_data_quality),
            DataProductVariable(P1_ERGE_VAR_NAME, self.p1_erge),
            DataProductVariable(P1_MULTI_FLAG_VAR_NAME, self.p1_multi_flag),
            DataProductVariable(P1_NUM_OF_EVENTS_VAR_NAME, self.p1_num_of_events),
            DataProductVariable(P1_SSD_ENERGY_VAR_NAME, self.p1_ssd_energy),
            DataProductVariable(P1_SSD_ID_VAR_NAME, self.p1_ssd_id),
            DataProductVariable(P1_SPIN_ANGLE_VAR_NAME, self.p1_spin_angle),
            DataProductVariable(P1_SPIN_NUMBER_VAR_NAME, self.p1_spin_number),
            DataProductVariable(P1_TOF_VAR_NAME, self.p1_tof),
            DataProductVariable(P1_TYPE_VAR_NAME, self.p1_type),
            DataProductVariable(P2_DATA_QUALITY_VAR_NAME, self.p2_data_quality),
            DataProductVariable(P2_ERGE_VAR_NAME, self.p2_erge),
            DataProductVariable(P2_MULTI_FLAG_VAR_NAME, self.p2_multi_flag),
            DataProductVariable(P2_NUM_OF_EVENTS_VAR_NAME, self.p2_num_of_events),
            DataProductVariable(P2_SSD_ENERGY_VAR_NAME, self.p2_ssd_energy),
            DataProductVariable(P2_SSD_ID_VAR_NAME, self.p2_ssd_id),
            DataProductVariable(P2_SPIN_ANGLE_VAR_NAME, self.p2_spin_angle),
            DataProductVariable(P2_SPIN_NUMBER_VAR_NAME, self.p2_spin_number),
            DataProductVariable(P2_TOF_VAR_NAME, self.p2_tof),
            DataProductVariable(P2_TYPE_VAR_NAME, self.p2_type),
            DataProductVariable(P3_DATA_QUALITY_VAR_NAME, self.p3_data_quality),
            DataProductVariable(P3_ERGE_VAR_NAME, self.p3_erge),
            DataProductVariable(P3_MULTI_FLAG_VAR_NAME, self.p3_multi_flag),
            DataProductVariable(P3_NUM_OF_EVENTS_VAR_NAME, self.p3_num_of_events),
            DataProductVariable(P3_SSD_ENERGY_VAR_NAME, self.p3_ssd_energy),
            DataProductVariable(P3_SSD_ID_VAR_NAME, self.p3_ssd_id),
            DataProductVariable(P3_SPIN_ANGLE_VAR_NAME, self.p3_spin_angle),
            DataProductVariable(P3_SPIN_NUMBER_VAR_NAME, self.p3_spin_number),
            DataProductVariable(P3_TOF_VAR_NAME, self.p3_tof),
            DataProductVariable(P3_TYPE_VAR_NAME, self.p3_type),
            DataProductVariable(P4_DATA_QUALITY_VAR_NAME, self.p4_data_quality),
            DataProductVariable(P4_ERGE_VAR_NAME, self.p4_erge),
            DataProductVariable(P4_MULTI_FLAG_VAR_NAME, self.p4_multi_flag),
            DataProductVariable(P4_NUM_OF_EVENTS_VAR_NAME, self.p4_num_of_events),
            DataProductVariable(P4_SSD_ENERGY_VAR_NAME, self.p4_ssd_energy),
            DataProductVariable(P4_SSD_ID_VAR_NAME, self.p4_ssd_id),
            DataProductVariable(P4_SPIN_ANGLE_VAR_NAME, self.p4_spin_angle),
            DataProductVariable(P4_SPIN_NUMBER_VAR_NAME, self.p4_spin_number),
            DataProductVariable(P4_TOF_VAR_NAME, self.p4_tof),
            DataProductVariable(P4_TYPE_VAR_NAME, self.p4_type),
            DataProductVariable(P5_DATA_QUALITY_VAR_NAME, self.p5_data_quality),
            DataProductVariable(P5_ERGE_VAR_NAME, self.p5_erge),
            DataProductVariable(P5_MULTI_FLAG_VAR_NAME, self.p5_multi_flag),
            DataProductVariable(P5_NUM_OF_EVENTS_VAR_NAME, self.p5_num_of_events),
            DataProductVariable(P5_SSD_ENERGY_VAR_NAME, self.p5_ssd_energy),
            DataProductVariable(P5_SSD_ID_VAR_NAME, self.p5_ssd_id),
            DataProductVariable(P5_SPIN_ANGLE_VAR_NAME, self.p5_spin_angle),
            DataProductVariable(P5_SPIN_NUMBER_VAR_NAME, self.p5_spin_number),
            DataProductVariable(P5_TOF_VAR_NAME, self.p5_tof),
            DataProductVariable(P5_TYPE_VAR_NAME, self.p5_type),
        ]


class CodiceL3HiDirectEventsBuilder:
    def __init__(self, l2_data: CodiceL2HiData):
        self.l2_data = l2_data
        self.p0_estimated_mass_lower = None
        self.p0_estimated_mass = None
        self.p0_estimated_mass_upper = None
        self.p0_energy_per_nuc_lower = None
        self.p0_energy_per_nuc = None
        self.p0_energy_per_nuc_upper = None
        self.p1_energy_per_nuc_lower = None
        self.p1_energy_per_nuc = None
        self.p1_energy_per_nuc_upper = None
        self.p1_estimated_mass_lower = None
        self.p1_estimated_mass = None
        self.p1_estimated_mass_upper = None
        self.p2_energy_per_nuc_lower = None
        self.p2_energy_per_nuc = None
        self.p2_energy_per_nuc_upper = None
        self.p2_estimated_mass_lower = None
        self.p2_estimated_mass = None
        self.p2_estimated_mass_upper = None
        self.p3_energy_per_nuc_lower = None
        self.p3_energy_per_nuc = None
        self.p3_energy_per_nuc_upper = None
        self.p3_estimated_mass_lower = None
        self.p3_estimated_mass = None
        self.p3_estimated_mass_upper = None
        self.p4_energy_per_nuc_lower = None
        self.p4_energy_per_nuc = None
        self.p4_energy_per_nuc_upper = None
        self.p4_estimated_mass_lower = None
        self.p4_estimated_mass = None
        self.p4_estimated_mass_upper = None
        self.p5_energy_per_nuc_lower = None
        self.p5_energy_per_nuc = None
        self.p5_energy_per_nuc_upper = None
        self.p5_estimated_mass_lower = None
        self.p5_estimated_mass = None
        self.p5_estimated_mass_upper = None

    def updated_priority_event_0(self, energy_per_nuc_with_bounds: ndarray, estimated_mass_with_bounds: ndarray):
        (self.p0_energy_per_nuc_lower,
         self.p0_energy_per_nuc,
         self.p0_energy_per_nuc_upper) = self._split_into_lower_middle_and_upper(energy_per_nuc_with_bounds)

        (self.p0_estimated_mass_lower,
         self.p0_estimated_mass,
         self.p0_estimated_mass_upper) = self._split_into_lower_middle_and_upper(estimated_mass_with_bounds)
        return self

    def updated_priority_event_1(self, energy_per_nuc_with_bounds, estimated_mass_with_bounds):
        (self.p1_energy_per_nuc_lower,
         self.p1_energy_per_nuc,
         self.p1_energy_per_nuc_upper) = self._split_into_lower_middle_and_upper(energy_per_nuc_with_bounds)

        (self.p1_estimated_mass_lower,
         self.p1_estimated_mass,
         self.p1_estimated_mass_upper) = self._split_into_lower_middle_and_upper(estimated_mass_with_bounds)
        return self

    def updated_priority_event_2(self, energy_per_nuc_with_bounds, estimated_mass_with_bounds):
        (self.p2_energy_per_nuc_lower,
         self.p2_energy_per_nuc,
         self.p2_energy_per_nuc_upper) = self._split_into_lower_middle_and_upper(energy_per_nuc_with_bounds)

        (self.p2_estimated_mass_lower,
         self.p2_estimated_mass,
         self.p2_estimated_mass_upper) = self._split_into_lower_middle_and_upper(estimated_mass_with_bounds)
        return self

    def updated_priority_event_3(self, energy_per_nuc_with_bounds, estimated_mass_with_bounds):
        (self.p3_energy_per_nuc_lower,
         self.p3_energy_per_nuc,
         self.p3_energy_per_nuc_upper) = self._split_into_lower_middle_and_upper(energy_per_nuc_with_bounds)

        (self.p3_estimated_mass_lower,
         self.p3_estimated_mass,
         self.p3_estimated_mass_upper) = self._split_into_lower_middle_and_upper(estimated_mass_with_bounds)
        return self

    def updated_priority_event_4(self, energy_per_nuc_with_bounds, estimated_mass_with_bounds):
        (self.p4_energy_per_nuc_lower,
         self.p4_energy_per_nuc,
         self.p4_energy_per_nuc_upper) = self._split_into_lower_middle_and_upper(energy_per_nuc_with_bounds)

        (self.p4_estimated_mass_lower,
         self.p4_estimated_mass,
         self.p4_estimated_mass_upper) = self._split_into_lower_middle_and_upper(estimated_mass_with_bounds)
        return self

    def updated_priority_event_5(self, energy_per_nuc_with_bounds, estimated_mass_with_bounds):
        (self.p5_energy_per_nuc_lower,
         self.p5_energy_per_nuc,
         self.p5_energy_per_nuc_upper) = self._split_into_lower_middle_and_upper(energy_per_nuc_with_bounds)

        (self.p5_estimated_mass_lower,
         self.p5_estimated_mass,
         self.p5_estimated_mass_upper) = self._split_into_lower_middle_and_upper(estimated_mass_with_bounds)

        return self

    def _split_into_lower_middle_and_upper(self, energy_or_mass_with_bounds):
        return (energy_or_mass_with_bounds[:, :, 0],
                energy_or_mass_with_bounds[:, :, 1],
                energy_or_mass_with_bounds[:, :, 2])

    def convert(self):
        kwargs = {}
        for index, priority_event in enumerate(self.l2_data.priority_events):
            priority_attrs = {f'p{index}_erge': priority_event.energy_range,
                              f'p{index}_data_quality': priority_event.data_quality,
                              f'p{index}_multi_flag': priority_event.multi_flag,
                              f'p{index}_num_of_events': priority_event.number_of_events,
                              f'p{index}_spin_angle': priority_event.spin_angle,
                              f'p{index}_spin_number': priority_event.spin_number,
                              f'p{index}_ssd_energy': priority_event.ssd_energy,
                              f'p{index}_ssd_id': priority_event.ssd_id,
                              f'p{index}_tof': priority_event.time_of_flight,
                              f'p{index}_type': priority_event.type,
                              f'p{index}_energy_per_nuc_lower': getattr(self, f'p{index}_energy_per_nuc_lower'),
                              f'p{index}_energy_per_nuc': getattr(self, f'p{index}_energy_per_nuc'),
                              f'p{index}_energy_per_nuc_upper': getattr(self, f'p{index}_energy_per_nuc_upper'),
                              f'p{index}_estimated_mass_lower': getattr(self, f'p{index}_estimated_mass_lower'),
                              f'p{index}_estimated_mass': getattr(self, f'p{index}_estimated_mass'),
                              f'p{index}_estimated_mass_upper': getattr(self, f'p{index}_estimated_mass_upper')}
            kwargs.update(priority_attrs)

        return CodiceL3HiDirectEvents(input_metadata=None, epoch=self.l2_data.epochs, **kwargs)
