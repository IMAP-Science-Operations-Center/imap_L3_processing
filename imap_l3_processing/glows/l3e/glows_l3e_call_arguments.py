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
        return [
            self.formatted_date,
            self.decimal_date,
            str(self.spacecraft_radius),
            f"{self.spacecraft_longitude:.4f}",
            f"{self.spacecraft_latitude:.4f}",
            f"{self.spacecraft_velocity_x:.4f}",
            f"{self.spacecraft_velocity_y:.4f}",
            f"{self.spacecraft_velocity_z}",
            f"{self.spin_axis_longitude:.4f}",
            f"{self.spin_axis_latitude:.4f}",
            f"{self.elongation:.3f}"
        ]
