import numpy as np
from spacepy import pycdf


class EfficiencyCalibrationTable:
    """Detector efficiencies over the mission, relative to the lab calibration.

    The response multiplies the lab central effective area by these values
    directly, so a table holding absolute efficiencies is not valid input.
    """

    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=[("time", "M8[ns]"), ("MET", "i8"), ("proton efficiency", "f8"), ("helium efficiency", "f8")], ndmin=1)

    def relative_proton_efficiency(self, time_as_tt2000) -> float:
        return self._efficiency_at("proton efficiency", time_as_tt2000)

    def relative_helium_efficiency(self, time_as_tt2000) -> float:
        return self._efficiency_at("helium efficiency", time_as_tt2000)

    def _efficiency_at(self, column: str, time_as_tt2000) -> float:
        for d in reversed(self.data):
            if d["time"] < np.datetime64(pycdf.lib.tt2000_to_datetime(int(time_as_tt2000)), "ns"):
                return float(d[column])

        raise ValueError(f"No efficiency data for {pycdf.lib.tt2000_to_datetime(time_as_tt2000)}")
