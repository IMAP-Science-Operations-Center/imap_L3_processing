from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF


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
DATA_QUALITY_VAR_NAME = "data_quality"
ENERGY_RANGE_VAR_NAME = "energy_range"
MULTI_FLAG_VAR_NAME = "multi_flag"
NUMBER_OF_EVENTS_VAR_NAME = "number_of_events"
ENERGY_VAR_NAME = "energy"
SPIN_SECTOR_VAR_NAME = "spin_sector"
SPIN_NUMBER_VAR_NAME = "spin_number"
TIME_OF_FLIGHT_VAR_NAME = "tof"
PRIORITY_VAR_NAME = "priority"
