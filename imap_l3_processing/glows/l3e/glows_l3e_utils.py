from datetime import datetime
from pathlib import Path

import numpy as np
import spiceypy
from astropy.time import Time
from imap_data_access import query
from imap_processing.spice.repoint import set_global_repoint_table_paths, get_repoint_data

from imap_l3_processing.constants import ONE_AU_IN_KM, TT2000_EPOCH, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import jd_fm_Carrington
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments


def determine_call_args_for_l3e_executable(start_date: datetime, repointing_midpoint: datetime,
                                           elongation: float) -> GlowsL3eCallArguments:
    ephemeris_time = spiceypy.datetime2et(repointing_midpoint)

    [x, y, z, vx, vy, vz], _ = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")

    radius, longitude, latitude = spiceypy.reclat([x, y, z])

    rotation_matrix = spiceypy.pxform("IMAP_DPS", "ECLIPJ2000", ephemeris_time)
    spin_axis = rotation_matrix @ [0, 0, 1]

    _, spin_axis_long, spin_axis_lat = spiceypy.reclat(spin_axis)

    formatted_date = start_date.strftime("%Y%m%d_%H%M%S")
    decimal_date = _decimal_time(repointing_midpoint)

    return GlowsL3eCallArguments(
        formatted_date=formatted_date,
        decimal_date=decimal_date,
        spacecraft_radius=radius / ONE_AU_IN_KM,
        spacecraft_longitude=np.rad2deg(longitude) % 360,
        spacecraft_latitude=np.rad2deg(latitude),
        spacecraft_velocity_x=vx,
        spacecraft_velocity_y=vy,
        spacecraft_velocity_z=vz,
        spin_axis_longitude=np.rad2deg(spin_axis_long) % 360,
        spin_axis_latitude=np.rad2deg(spin_axis_lat),
        elongation=elongation
    )


def _decimal_time(t: datetime) -> str:
    year_start = datetime(t.year, 1, 1)
    year_end = datetime(t.year + 1, 1, 1)
    return "{:10.5f}".format(t.year + (t - year_start) / (year_end - year_start))


def determine_l3e_files_to_produce(descriptor: str, first_cr_processed: int, last_processed_cr: int, version: str,
                                   repointing_path: Path):
    l3e_files = query(instrument='glows', descriptor=descriptor, data_level='l3e', version=version)

    existing_pointings = [l3e['repointing'] for l3e in l3e_files]
    print(f"l3e {version} exists for repointings {set(existing_pointings)}")
    set_global_repoint_table_paths([repointing_path])
    repointing_data = get_repoint_data()

    first_carrington_start_date = Time(jd_fm_Carrington(float(first_cr_processed)), format='jd')
    last_cr_end_date = Time(jd_fm_Carrington(float(last_processed_cr + 1)), format='jd')

    start_ns = (first_carrington_start_date.to_datetime() - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS
    end_ns = (last_cr_end_date.to_datetime() - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS

    vectorized_date_conv = np.vectorize(lambda d: (Time(d, format="iso").to_datetime(
        leap_second_strict='silent') - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS)
    repoint_starts = vectorized_date_conv(repointing_data["repoint_start_utc"])
    repoint_ids = repointing_data["repoint_id"]
    print("repoint file has repoint ids", list(repoint_ids))

    pointing_numbers = []
    for i in range(len(repoint_ids)):
        if i + 1 < len(repoint_ids) and start_ns < repoint_starts[i + 1] < end_ns:
            pointing_numbers.append(int(repoint_ids[i]))
    print(f"We can make repointings {pointing_numbers}")

    pointing_numbers = [pointing for pointing in pointing_numbers if pointing not in existing_pointings]
    print("l3e files to make", pointing_numbers)
    return pointing_numbers
