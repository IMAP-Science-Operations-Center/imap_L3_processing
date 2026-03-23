from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
import numpy as np
import spiceypy
from astropy.time import Time
from imap_processing.spice.repoint import set_global_repoint_table_paths, get_repoint_data
from spacepy.pycdf import CDF

from typing import Optional

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


@dataclass
class GlowsL3eRepointings:
    repointing_numbers: list[int]
    hi_90_repointings: dict[int, int]
    hi_45_repointings: dict[int, int]
    lo_repointings: dict[int, int]
    ultra_repointings: dict[int, int]


def determine_l3e_files_to_produce(first_cr_processed: int, last_processed_cr: int,
                                   repointing_path: Path) -> GlowsL3eRepointings:
    descriptors = ['survival-probability-hi-90','survival-probability-hi-45','survival-probability-lo','survival-probability-ul']

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

    pointing_numbers = []
    for i in range(len(repoint_ids)):
        if i + 1 < len(repoint_ids) and start_ns < repoint_starts[i + 1] < end_ns:
            pointing_numbers.append(int(repoint_ids[i]))

    updated_pointings_per_instruments = []
    for descriptor in descriptors:
        l3e_files = imap_data_access.query(instrument='glows', data_level='l3e', version="latest", descriptor=descriptor)
        updated_pointing = {int(l3e['repointing']): int(l3e['version'][1:]) for l3e in l3e_files}

        for pointing_number in pointing_numbers:
            if pointing_number in updated_pointing:
                updated_pointing[pointing_number] = updated_pointing[pointing_number] + 1
            else:
                updated_pointing[pointing_number] = 1
        updated_pointings_per_instruments.append(updated_pointing)

    return GlowsL3eRepointings(pointing_numbers, *updated_pointings_per_instruments)


def find_first_updated_cr(new_l3d: Path, old_l3d: str) -> Optional[int]:
    downloaded_old_l3d = imap_data_access.download(old_l3d)

    old_l3d_cdf = CDF(str(downloaded_old_l3d))
    new_l3d_cdf = CDF(str(new_l3d))

    for i, cr in enumerate(old_l3d_cdf['cr_grid'][...]):
        lya_matches = np.isclose(old_l3d_cdf['lyman_alpha'][i], new_l3d_cdf['lyman_alpha'][i])
        phion_matches = np.isclose(old_l3d_cdf['phion'][i], new_l3d_cdf['phion'][i])
        plasma_speed_flag_matches = np.isclose(old_l3d_cdf['plasma_speed_flag'][i], new_l3d_cdf['plasma_speed_flag'][i])
        proton_density_flag_matches = np.isclose(old_l3d_cdf['proton_density_flag'][i], new_l3d_cdf['proton_density_flag'][i])
        uv_anisotropy_flag_matches = np.isclose(old_l3d_cdf['uv_anisotropy_flag'][i], new_l3d_cdf['uv_anisotropy_flag'][i])

        plasma_speed_matches = np.all(np.isclose(old_l3d_cdf['plasma_speed'][i], new_l3d_cdf['plasma_speed'][i]))
        proton_density_matches = np.all(np.isclose(old_l3d_cdf['proton_density'][i], new_l3d_cdf['proton_density'][i]))
        uv_anisotropy_matches = np.all(np.isclose(old_l3d_cdf['uv_anisotropy'][i], new_l3d_cdf['uv_anisotropy'][i]))

        if np.any(np.logical_not([lya_matches, phion_matches, plasma_speed_matches, plasma_speed_flag_matches, proton_density_matches, proton_density_flag_matches, uv_anisotropy_matches, uv_anisotropy_flag_matches])):
            return int(cr)

    if old_l3d_cdf['cr_grid'].shape != new_l3d_cdf['cr_grid'].shape:
        return int(old_l3d_cdf['cr_grid'][-1]) + 1

    return None

def get_lo_pivot_angle_from_l1b_file(path: Path) -> float:
    with CDF(str(path)) as cdf:
        epoch = cdf['epoch'][...]
        angles = cdf['pcc_coarse_pot_pri'][...]
    if len(epoch) == 0:
        return 90
    t0 = epoch[0]
    start = t0 + timedelta(hours=3)
    end = t0 + timedelta(hours=15)
    start_index, end_index = np.searchsorted(epoch, [start, end])
    angles_to_consider = angles[start_index:end_index]
    if len(angles_to_consider) == 0:
        return 90
    return np.round(np.median(angles_to_consider))

@dataclass
class LoPivotAngle:
    parent_filename: Optional[str]
    pivot_angle: float

def get_lo_pivot_angles(repointings: list[int]) -> dict[int, LoPivotAngle]:
    l1b_results = imap_data_access.query(
        instrument="lo",
        data_level="l1b",
        descriptor="nhk",
        version="latest",
    )
    paths_by_repointing = {f["repointing"]:f["file_path"] for f in l1b_results}
    result = {}
    for repointing in repointings:
        if path := paths_by_repointing.get(repointing):
            downloaded_path = imap_data_access.download(path)
            result[repointing] = LoPivotAngle(parent_filename=Path(path).name, pivot_angle=get_lo_pivot_angle_from_l1b_file(downloaded_path))
        else:
            result[repointing] = LoPivotAngle(parent_filename=None, pivot_angle=90)
    return result
