from datetime import datetime

import numpy as np

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.spice_wrapper import spiceypy


def determine_call_args_for_l3e_executable(start_date: datetime, repointing_midpoint: datetime,
                                           elongation: float) -> list[str]:
    ephemeris_time = spiceypy.datetime2et(repointing_midpoint)

    [x, y, z, vx, vy, vz], _ = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")

    radius, longitude, latitude = spiceypy.reclat([x, y, z])

    rotation_matrix = spiceypy.pxform("IMAP_DPS", "ECLIPJ2000", ephemeris_time)
    spin_axis = rotation_matrix @ [0, 0, 1]

    _, spin_axis_long, spin_axis_lat = spiceypy.reclat(spin_axis)

    formatted_date = start_date.strftime("%Y%m%d_%H%M%S")
    decimal_date = _decimal_time(repointing_midpoint)

    return f"{formatted_date} {decimal_date} {radius / ONE_AU_IN_KM} {np.rad2deg(longitude) % 360} {np.rad2deg(latitude)} {vx} {vy} {vz} {np.rad2deg(spin_axis_long) % 360} {spin_axis_lat:.4f} {elongation:.3f}".split(
        " ")


def _decimal_time(t: datetime) -> str:
    year_start = datetime(t.year, 1, 1)
    year_end = datetime(t.year + 1, 1, 1)
    return "{:10.5f}".format(t.year + (t - year_start) / (year_end - year_start))
