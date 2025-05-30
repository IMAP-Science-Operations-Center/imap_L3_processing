from dataclasses import dataclass

import numpy as np


@dataclass
class GlowsL3eCallArguments:
    formatted_date: str
    decimal_date: str
    spacecraft_radius: np.float32
    spacecraft_longitude: np.float32
    spacecraft_latitude: np.float32
    spacecraft_velocity_x: np.float32
    spacecraft_velocity_y: np.float32
    spacecraft_velocity_z: np.float32
    spin_axis_longitude: np.float32
    spin_axis_latitude: np.float32
    elongation: float

    def to_argument_list(self):
        return (
                f"{self.formatted_date} {self.decimal_date} {self.spacecraft_radius} {self.spacecraft_longitude} {self.spacecraft_latitude} {self.spacecraft_velocity_x} " +
                f"{self.spacecraft_velocity_y} {self.spacecraft_velocity_z} {self.spin_axis_longitude} {self.spin_axis_latitude:.4f} {self.elongation:.3f}").split(
            " ")
