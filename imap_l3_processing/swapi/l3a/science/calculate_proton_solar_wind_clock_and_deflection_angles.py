import numpy as np
import uncertainties
from scipy.interpolate import LinearNDInterpolator


class ClockAngleCalibrationTable:
    def __init__(self, calibration_table: np.ndarray):
        coords = calibration_table[:, 0:2]
        values = calibration_table[:, 2:4]
        self.interp = LinearNDInterpolator(coords, values)

    @uncertainties.wrap
    def lookup_clock_offset(self, sw_speed: float, a_over_b: float):
        return self.interp(sw_speed, a_over_b)[1]

    @uncertainties.wrap
    def lookup_flow_deflection(self, sw_speed: float, a_over_b: float):
        return self.interp(sw_speed, a_over_b)[0]

    @classmethod
    def from_file(cls, file):
        data = np.loadtxt(file)
        return cls(data)


def calculate_clock_angle(lookup_table: ClockAngleCalibrationTable, sw_speed, a, phi, b):
    a_over_b = a / b
    phi_offset = lookup_table.lookup_clock_offset(sw_speed, a_over_b)
    return (phi + phi_offset) % 360


def calculate_deflection_angle(lookup_table, sw_speed, a, phi, b):
    a_over_b = a / b
    deflection_angle = lookup_table.lookup_flow_deflection(sw_speed, a_over_b)
    return deflection_angle
