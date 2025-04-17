from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from spiceypy import spiceypy

from imap_l3_processing import spice_wrapper
from imap_l3_processing.constants import ONE_AU_IN_KM

spice_wrapper.furnish()


@dataclass
class GlowsL3eSpiceInput:
    radius: float
    ecliptic_longitude: float
    ecliptic_latitude: float
    vx: float
    vy: float
    vz: float
    spin_axis_longitude: float
    spin_axis_latitude: float

    def format(self):
        return f"{self.radius} {self.ecliptic_longitude} {self.ecliptic_latitude} {self.vx} {self.vy} {self.vz} {self.spin_axis_longitude} {self.spin_axis_latitude}"


def fetch_spice_for_glows_l3e(epoch: datetime) -> GlowsL3eSpiceInput:
    et_time = spiceypy.datetime2et(epoch)
    [x, y, z, vx, vy, vz], _ = spiceypy.spkezr("IMAP", et_time, "ECLIPJ2000", "NONE", "SUN")

    radius, longitude, latitude = spiceypy.reclat([x, y, z])

    rotation_matrix = spiceypy.pxform("IMAP_DPS", "ECLIPJ2000", et_time)
    spin_axis = rotation_matrix @ [0, 0, 1]

    _, spin_axis_long, spin_axis_lat = spiceypy.reclat(spin_axis)

    return GlowsL3eSpiceInput(
        radius=radius / ONE_AU_IN_KM,
        ecliptic_longitude=np.rad2deg(longitude) % 360,
        ecliptic_latitude=np.rad2deg(latitude),
        vx=vx,
        vy=vy,
        vz=vz,
        spin_axis_longitude=np.rad2deg(spin_axis_long),
        spin_axis_latitude=np.rad2deg(spin_axis_lat),
    )


def decimal_time(t: datetime) -> float:
    year_start = datetime(t.year, 1, 1)
    year_end = datetime(t.year + 1, 1, 1)
    return t.year + (t - year_start) / (year_end - year_start)


start_time = datetime(2025, 4, 15, 12)


def make_input_args(start_time, number_of_points, elongation):
    lines = []
    for i in range(number_of_points):
        t = start_time + timedelta(days=i)
        fake_time = t - timedelta(days=3650)

        formatted_time = fake_time.strftime("%Y%m%d_%H%M%S")
        spice_data = fetch_spice_for_glows_l3e(t)
        lines.append(f"{formatted_time} {decimal_time(fake_time)} {spice_data.format()} {elongation}")
    return lines


for line in make_input_args(start_time, 365, 90):
    print(line)
