import json
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional
from zipfile import ZipFile, ZIP_DEFLATED

import imap_data_access
import numpy as np
import pandas as pd
from imap_processing.spice.repoint import get_repoint_data
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3a.models import GlowsL3LightCurve, PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME, \
    PHOTON_FLUX_CDF_VAR_NAME, RAW_HISTOGRAM_CDF_VAR_NAME, EXPOSURE_TIMES_CDF_VAR_NAME, EPOCH_CDF_VAR_NAME, \
    EPOCH_DELTA_CDF_VAR_NAME, SPIN_ANGLE_CDF_VAR_NAME, SPIN_ANGLE_DELTA_CDF_VAR_NAME, LATITUDE_CDF_VAR_NAME, \
    LONGITUDE_CDF_VAR_NAME, EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME, TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME, \
    SPACECRAFT_VELOCITY_STD_DEV_CDF_VAR_NAME, FILTER_TEMPERATURE_AVERAGE_CDF_VAR_NAME, \
    FILTER_TEMPERATURE_STD_DEV_CDF_VAR_NAME, HV_VOLTAGE_AVERAGE_CDF_VAR_NAME, HV_VOLTAGE_STD_DEV_CDF_VAR_NAME, \
    SPIN_PERIOD_AVERAGE_CDF_VAR_NAME, SPIN_PERIOD_STD_DEV_CDF_VAR_NAME, SPIN_PERIOD_GROUND_AVERAGE_CDF_VAR_NAME, \
    SPIN_PERIOD_GROUND_STD_DEV_CDF_VAR_NAME, PULSE_LENGTH_AVERAGE_CDF_VAR_NAME, PULSE_LENGTH_STD_DEV_CDF_VAR_NAME, \
    POSITION_ANGLE_OFFSET_AVERAGE_CDF_VAR_NAME, POSITION_ANGLE_OFFSET_STD_DEV_CDF_VAR_NAME, \
    SPIN_AXIS_ORIENTATION_AVERAGE_CDF_VAR_NAME, SPIN_AXIS_ORIENTATION_STD_DEV_CDF_VAR_NAME, \
    SPACECRAFT_LOCATION_AVERAGE_CDF_VAR_NAME, SPACECRAFT_LOCATION_STD_DEV_CDF_VAR_NAME, \
    SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME, NUM_OF_BINS_CDF_VAR_NAME
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import ExternalDependencies


def read_cdf_parents(server_file_name: str) -> set[str]:
    downloaded_path = imap_data_access.download(server_file_name)

    with CDF(str(downloaded_path)) as cdf:
        parents = set(cdf.attrs["Parents"])
    return parents


def get_best_ancillary(start_date: datetime, end_date: datetime, ancillary_query_results: list[dict]) -> Optional[str]:
    valid_ancillaries = []
    for ancillary_file in ancillary_query_results:
        ancillary_start_date = datetime.strptime(ancillary_file["start_date"], "%Y%m%d")
        ancillary_end_date = datetime.strptime(ancillary_file["end_date"], "%Y%m%d") if ancillary_file[
            "end_date"] else None

        if ancillary_start_date <= end_date and (ancillary_end_date is None or ancillary_end_date >= start_date):
            valid_ancillaries.append(ancillary_file)

    if len(valid_ancillaries) == 0:
        return None
    else:
        latest_ancillary = max(valid_ancillaries, key=lambda x: x["ingestion_date"])
        return Path(latest_ancillary["file_path"]).name


jd_carrington_first = 2091
jd_carrington_start_date = datetime(2009, 12, 7, 4)
carrington_length = timedelta(days=27.2753)


def get_date_range_of_cr(cr_number: int) -> tuple[datetime, datetime]:
    start_date = jd_carrington_start_date + (cr_number - jd_carrington_first) * carrington_length
    return start_date, start_date + carrington_length


def get_cr_for_date_time(datetime_to_check: datetime) -> int:
    return int(jd_carrington_first + (datetime_to_check - jd_carrington_start_date) / carrington_length)


def read_glows_l3a_data(cdf: CDF) -> GlowsL3LightCurve:
    epoch_delta_s = cdf[EPOCH_DELTA_CDF_VAR_NAME][...] / 1e9
    epoch_delta = np.array([timedelta(seconds=x).total_seconds() for x in epoch_delta_s])
    return GlowsL3LightCurve(None,
                             photon_flux=cdf[PHOTON_FLUX_CDF_VAR_NAME][...],
                             photon_flux_uncertainty=cdf[PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME][...],
                             raw_histogram=cdf[RAW_HISTOGRAM_CDF_VAR_NAME][...],
                             number_of_bins=cdf[NUM_OF_BINS_CDF_VAR_NAME][...],
                             exposure_times=cdf[EXPOSURE_TIMES_CDF_VAR_NAME][...],
                             epoch=cdf[EPOCH_CDF_VAR_NAME][...],
                             epoch_delta=epoch_delta,
                             spin_angle=cdf[SPIN_ANGLE_CDF_VAR_NAME][...],
                             spin_angle_delta=cdf[SPIN_ANGLE_DELTA_CDF_VAR_NAME][...],
                             latitude=cdf[LATITUDE_CDF_VAR_NAME][...],
                             longitude=cdf[LONGITUDE_CDF_VAR_NAME][...],
                             extra_heliospheric_background=cdf[EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME][...],
                             time_dependent_background=cdf[TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME][...],
                             filter_temperature_average=cdf[FILTER_TEMPERATURE_AVERAGE_CDF_VAR_NAME][...],
                             filter_temperature_std_dev=cdf[FILTER_TEMPERATURE_STD_DEV_CDF_VAR_NAME][...],
                             hv_voltage_average=cdf[HV_VOLTAGE_AVERAGE_CDF_VAR_NAME][...],
                             hv_voltage_std_dev=cdf[HV_VOLTAGE_STD_DEV_CDF_VAR_NAME][...],
                             spin_period_average=cdf[SPIN_PERIOD_AVERAGE_CDF_VAR_NAME][...],
                             spin_period_std_dev=cdf[SPIN_PERIOD_STD_DEV_CDF_VAR_NAME][...],
                             spin_period_ground_average=cdf[SPIN_PERIOD_GROUND_AVERAGE_CDF_VAR_NAME][...],
                             spin_period_ground_std_dev=cdf[SPIN_PERIOD_GROUND_STD_DEV_CDF_VAR_NAME][...],
                             pulse_length_average=cdf[PULSE_LENGTH_AVERAGE_CDF_VAR_NAME][...],
                             pulse_length_std_dev=cdf[PULSE_LENGTH_STD_DEV_CDF_VAR_NAME][...],
                             position_angle_offset_average=cdf[POSITION_ANGLE_OFFSET_AVERAGE_CDF_VAR_NAME][...],
                             position_angle_offset_std_dev=cdf[POSITION_ANGLE_OFFSET_STD_DEV_CDF_VAR_NAME][...],
                             spin_axis_orientation_average=cdf[SPIN_AXIS_ORIENTATION_AVERAGE_CDF_VAR_NAME][...],
                             spin_axis_orientation_std_dev=cdf[SPIN_AXIS_ORIENTATION_STD_DEV_CDF_VAR_NAME][...],
                             spacecraft_location_average=cdf[SPACECRAFT_LOCATION_AVERAGE_CDF_VAR_NAME][...],
                             spacecraft_location_std_dev=cdf[SPACECRAFT_LOCATION_STD_DEV_CDF_VAR_NAME][...],
                             spacecraft_velocity_average=cdf[SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME][...],
                             spacecraft_velocity_std_dev=cdf[SPACECRAFT_VELOCITY_STD_DEV_CDF_VAR_NAME][...],
                             )


def archive_dependencies(l3bc_deps: GlowsL3BCDependencies, external_dependencies: ExternalDependencies) -> Path:
    start_date = l3bc_deps.start_date.strftime("%Y%m%d")
    zip_path = TEMP_CDF_FOLDER_PATH / f"imap_glows_l3b-archive_{start_date}_v{l3bc_deps.version:03}.zip"
    json_filename = "cr_to_process.json"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as file:
        file.write(external_dependencies.lyman_alpha_path, "lyman_alpha_composite.nc")
        file.write(external_dependencies.omni2_data_path, "omni2_all_years.dat")
        file.write(external_dependencies.f107_index_file_path, "f107_fluxtable.txt")
        cr = {"cr_rotation_number": l3bc_deps.carrington_rotation_number,
              "l3a_paths": [l3a['filename'] for l3a in l3bc_deps.l3a_data],
              "cr_start_date": str(l3bc_deps.start_date),
              "cr_end_date": str(l3bc_deps.end_date),
              "bad_days_list": l3bc_deps.ancillary_files['bad_days_list'].name,
              "pipeline_settings": l3bc_deps.ancillary_files['pipeline_settings'].name,
              "waw_helioion_mp": l3bc_deps.ancillary_files['WawHelioIonMP_parameters'].name,
              "uv_anisotropy": l3bc_deps.ancillary_files['uv_anisotropy'].name,
              "repointing_file": None
              }
        json_string = json.dumps(cr)
        file.writestr(json_filename, json_string)
    return zip_path


def get_pointing_date_range(repointing: int) -> (np.datetime64, np.datetime64):
    repointing_df: pd.DataFrame = get_repoint_data()
    matching_rows_start = repointing_df[repointing_df['repoint_id'] == repointing]
    matching_rows_end = repointing_df[repointing_df['repoint_id'] == repointing + 1]
    if len(matching_rows_start) == 0 or len(matching_rows_end) == 0:
        raise ValueError(f"No pointing found for pointing: {repointing}")
    repointing_data_start = matching_rows_start.iloc[0]
    repointing_data_end = matching_rows_end.iloc[0]
    start_time = repointing_data_start['repoint_end_utc']
    end_time = repointing_data_end['repoint_start_utc']

    return np.datetime64(start_time), np.datetime64(end_time)
