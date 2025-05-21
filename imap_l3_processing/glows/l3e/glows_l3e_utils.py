from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.time import Time

from imap_l3_processing.constants import ONE_AU_IN_KM, TT2000_EPOCH, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import jd_fm_Carrington
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


def determine_repointing_numbers_for_cr(cr_number: int, path_to_csv: Path) -> list[int]:
    carrington_start_date = Time(jd_fm_Carrington(float(cr_number)), format='jd')
    carrington_end_date = Time(jd_fm_Carrington(float(cr_number + 1)), format='jd')

    repointing_data = np.loadtxt(path_to_csv, skiprows=1, delimiter=",", dtype=str)

    start_ns = (carrington_start_date.to_datetime() - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS
    end_ns = (carrington_end_date.to_datetime() - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS
    vectorized_date_conv = np.vectorize(lambda d: (Time(d, format="isot").to_datetime(
        leap_second_strict='silent') - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS)
    repointing_data[:, 3] = vectorized_date_conv(repointing_data[:, 3])
    repointing_data[:, 6] = vectorized_date_conv(repointing_data[:, 6])

    repointing_data = repointing_data.astype(float)

    pointing_numbers = []
    for i in range(len(repointing_data)):
        if (repointing_data[i, 6] > start_ns) & (repointing_data[i, 6] < end_ns):
            pointing_numbers.append(repointing_data[i, 7])
        elif i + 1 < len(repointing_data) and start_ns < repointing_data[i + 1, 3] < end_ns:
            pointing_numbers.append(repointing_data[i, 7])

    return pointing_numbers
