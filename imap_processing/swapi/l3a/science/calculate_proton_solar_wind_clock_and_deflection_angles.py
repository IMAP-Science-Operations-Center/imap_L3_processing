import numpy as np
import scipy
import uncertainties


class ClockAngleCalibrationTable:
    def __init__(self, calibration_table: np.ndarray):
        solar_wind_speed = np.unique(calibration_table[:, 0])
        a_over_b = np.unique(calibration_table[:, 1])

        self.grid = (solar_wind_speed, a_over_b)
        values_shape = tuple(len(x) for x in self.grid)

        self.deflection_angle = calibration_table[:, 2].reshape(values_shape)
        self.phi_offset = calibration_table[:, 3].reshape(values_shape)

    @uncertainties.wrap
    def lookup_clock_offset(self, sw_speed, a_over_b):
        return scipy.interpolate.interpn(self.grid, self.phi_offset,
                                         [sw_speed, a_over_b])[0]

    @uncertainties.wrap
    def lookup_flow_deflection(self, sw_speed, a_over_b):
        return scipy.interpolate.interpn(self.grid, self.deflection_angle, [sw_speed, a_over_b])[0]

    @classmethod
    def from_file(cls, file):
        data = np.loadtxt(file)
        return cls(data)


def calculate_clock_angle(lookup_table, sw_speed, a, phi, b):
    a_over_b = a / b
    phi_offset = lookup_table.lookup_clock_offset(sw_speed, a_over_b)
    return (phi + phi_offset) % 360


def calculate_deflection_angle(lookup_table, sw_speed, a, phi, b):
    a_over_b = a / b
    deflection_angle = lookup_table.lookup_flow_deflection(sw_speed, a_over_b)
    return deflection_angle
