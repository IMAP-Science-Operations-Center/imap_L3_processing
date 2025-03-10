import numpy as np
from spacepy import pycdf


class EfficiencyCalibrationTable:
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=[("time", "M8[ns]"), ("MET", "i8"), ("efficiency", "f8")])

    def get_efficiency_for(self, time_as_tt2000):
        for d in reversed(self.data):
            if d[0] < np.datetime64(pycdf.lib.tt2000_to_datetime(time_as_tt2000), "ns"):
                return d[2]

        raise ValueError(f"No efficiency data for {pycdf.lib.tt2000_to_datetime(time_as_tt2000)}")
