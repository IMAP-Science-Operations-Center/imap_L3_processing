from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF


@dataclass
class PriorityEvent:
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
class CodiceL1aHiData:
    epochs: ndarray[datetime]
    priority_event_0: PriorityEvent
    priority_event_1: PriorityEvent
    priority_event_2: PriorityEvent
    priority_event_3: PriorityEvent
    priority_event_4: PriorityEvent
    priority_event_5: PriorityEvent

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

                priority_events.append(PriorityEvent(**priority_event_data))

            return cls(epochs, *priority_events)
