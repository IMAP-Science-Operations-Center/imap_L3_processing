import numpy as np


class EfficiencyCalibrationTable:
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=[("time", "M8[ns]"), ("MET", "i8"), ("efficiency", "f8")])

    def get_efficiency_for(self, time):
        for d in reversed(self.data):
            if d[0] < np.datetime64(time):
                return d[2]

        raise ValueError(f"No efficiency data for {time}")
