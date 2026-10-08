from __future__ import annotations

import typing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
import numpy as np
import spiceypy
from astropy.time import Time
from spiceypy import SpiceyError

from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_data_access.file_validation import Version
from spacepy.pycdf import CDF

from typing import Optional

from imap_l3_processing.constants import ONE_AU_IN_KM, TT2000_EPOCH, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import jd_fm_Carrington
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments, GlowsL3eSpacecraftInfo
from imap_l3_processing.utils import FurnishMetakernelOutput

if typing.TYPE_CHECKING:
    pass

def determine_call_args_for_l3e_executable(start_date: datetime, repointing_midpoint: datetime, elongation: float,
                                           spacecraft_info: GlowsL3eSpacecraftInfo) -> GlowsL3eCallArguments:
    formatted_date = start_date.strftime("%Y%m%d_%H%M%S")
    decimal_date = _decimal_time(repointing_midpoint)

    return GlowsL3eCallArguments(
        formatted_date=formatted_date,
        decimal_date=decimal_date,
        elongation=elongation,
        spacecraft_info=spacecraft_info,
    )

def determine_spacecraft_info_for_l3e_executable(repointing_midpoint: datetime) -> GlowsL3eSpacecraftInfo:
    ephemeris_time = spiceypy.datetime2et(repointing_midpoint)

    [x, y, z, vx, vy, vz], _ = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")

    radius, longitude, latitude = spiceypy.reclat([x, y, z])

    rotation_matrix = spiceypy.pxform("IMAP_DPS", "ECLIPJ2000", ephemeris_time)
    spin_axis = rotation_matrix @ [0, 0, 1]

    _, spin_axis_long, spin_axis_lat = spiceypy.reclat(spin_axis)

    return GlowsL3eSpacecraftInfo(
        spacecraft_radius=radius / ONE_AU_IN_KM,
        spacecraft_longitude=np.rad2deg(longitude) % 360,
        spacecraft_latitude=np.rad2deg(latitude),
        spacecraft_velocity_x=vx,
        spacecraft_velocity_y=vy,
        spacecraft_velocity_z=vz,
        spin_axis_longitude=np.rad2deg(spin_axis_long) % 360,
        spin_axis_latitude=np.rad2deg(spin_axis_lat),
    )


def determine_spacecraft_info_using_predict_if_needed(repointing_midpoint: datetime, spice_with_predict: FurnishMetakernelOutput, spice_without_predict: FurnishMetakernelOutput) -> \
    tuple[GlowsL3eSpacecraftInfo, GlowsL3Flags, list[str]]:
    try:
        with spiceypy.KernelPool([str(spice_without_predict.metakernel_path)]):
            spacecraft_info: GlowsL3eSpacecraftInfo = determine_spacecraft_info_for_l3e_executable(repointing_midpoint)

            glows_l3_flags: GlowsL3Flags = GlowsL3Flags.NONE
            kernel_names = [n.name for n in spice_without_predict.spice_kernel_paths]
    except SpiceyError:
        with spiceypy.KernelPool([str(spice_with_predict.metakernel_path)]):
            spacecraft_info: GlowsL3eSpacecraftInfo = determine_spacecraft_info_for_l3e_executable(
                repointing_midpoint)

            glows_l3_flags: GlowsL3Flags = GlowsL3Flags.PREDICTIVE_EPHEMERIS
            kernel_names = [n.name for n in spice_with_predict.spice_kernel_paths]

    return spacecraft_info, glows_l3_flags, kernel_names


def _decimal_time(t: datetime) -> str:
    year_start = datetime(t.year, 1, 1)
    year_end = datetime(t.year + 1, 1, 1)
    return "{:10.5f}".format(t.year + (t - year_start) / (year_end - year_start))


@dataclass
class GlowsL3eVersionsForRepointings:
    repointing_numbers: list[int]
    hi_90_repointings: dict[int, Version]
    hi_45_repointings: dict[int, Version]
    lo_repointings: dict[int, Version]
    ultra_sf_repointings: dict[int, Version]
    ultra_hf_repointings: dict[int, Version]


def compute_glows_flags_for_repoint(l3d_cdf_path: Path, repoint_midpoint: datetime) -> int:
    with CDF(str(l3d_cdf_path)) as cdf:
        cr_epochs = cdf['epoch'][...]
        flags = cdf['glows_flags'][...]

    cr_after_repoint = np.searchsorted(cr_epochs, repoint_midpoint)
    cr_before_repoint = cr_after_repoint - 1

    relevant_crs = [cr_before_repoint]
    if cr_after_repoint < len(cr_epochs):
        if cr_epochs[cr_after_repoint] != repoint_midpoint:
            relevant_crs.append(cr_after_repoint)
        else:
            relevant_crs = [cr_after_repoint]

    selected = flags[relevant_crs]

    return int(np.bitwise_or.reduce(selected.astype(np.uint16), initial=0))

def get_lo_pivot_angle_from_l1b_file(path: Path) -> float:
    with CDF(str(path)) as cdf:
        epoch = cdf['epoch'][...]
        angles = cdf['pcc_coarse_pot_pri'][...]
    if len(epoch) == 0:
        return 90.0
    t0 = epoch[0]
    start = t0 + timedelta(hours=0.5)
    end = t0 + timedelta(hours=22.5)
    start_index, end_index = np.searchsorted(epoch, [start, end])
    angles_to_consider = angles[start_index:end_index]
    if len(angles_to_consider) == 0:
        return 90.0
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
            result[repointing] = LoPivotAngle(parent_filename=None, pivot_angle=90.0)
    return result

def get_repoint_numbers_within_cr_window(start_cr_number: int | None, end_cr_number: int, repointing_data) -> set[int]:
    if start_cr_number is None:
        return set()
    first_carrington_start_date = Time(jd_fm_Carrington(float(start_cr_number)), format='jd')
    last_cr_end_date = Time(jd_fm_Carrington(float(end_cr_number + 0.5)), format='jd')

    start_ns = (first_carrington_start_date.to_datetime() - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS
    end_ns = (last_cr_end_date.to_datetime() - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS

    vectorized_date_conv = np.vectorize(lambda d: (Time(d, format="iso").to_datetime(
        leap_second_strict='silent') - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS)
    repoint_starts = vectorized_date_conv(repointing_data["repoint_start_utc"])
    repoint_ends = vectorized_date_conv(repointing_data["repoint_end_utc"])
    repoint_ids = repointing_data["repoint_id"]

    repoint_numbers = set()
    for i in range(len(repoint_ids)):
        if i + 1 < len(repoint_ids) and start_ns < (repoint_starts[i + 1] + repoint_ends[i])/2 < end_ns:
            repoint_numbers.add(int(repoint_ids[i]))

    return repoint_numbers

def calculate_energy_deltas(centers: np.ndarray):
    edges = np.empty_like(centers, shape=(len(centers) + 1,))
    edges[1:-1] = np.sqrt(centers[1:] * centers[:-1])
    edges[0] = np.sqrt(centers[0] / centers[1]) * centers[0]
    edges[-1] = np.sqrt(centers[-1] / centers[-2]) * centers[-1]

    delta_plus = edges[1:] - centers
    delta_minus = centers - edges[:-1]

    return delta_plus, delta_minus

