import numpy as np
from spacepy import pycdf


class EfficiencyCalibrationTable:
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=[("time", "M8[ns]"), ("MET", "i8"), ("proton efficiency", "f8"), ("alpha efficiency", "f8")])

    def get_proton_efficiency_for(self, time_as_tt2000) -> float:
        return self._get_efficiency_for_index("proton efficiency", time_as_tt2000)

    def get_alpha_efficiency_for(self, time_as_tt2000) -> float:
        return self._get_efficiency_for_index("alpha efficiency", time_as_tt2000)

    def _get_efficiency_for_index(self, name, time_as_tt2000) -> float:
        for d in reversed(self.data):
            if d["time"] < np.datetime64(pycdf.lib.tt2000_to_datetime(int(time_as_tt2000)), "ns"):
                return d[name]

        raise ValueError(f"No efficiency data for {pycdf.lib.tt2000_to_datetime(time_as_tt2000)}")
