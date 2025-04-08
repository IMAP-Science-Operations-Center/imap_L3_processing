import json
from collections import defaultdict
from datetime import timedelta
from json import dump
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from astropy.time import Time, TimeDelta
from spacepy.pycdf import CDF

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
    SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME
from imap_l3_processing.glows.l3bc.dependency_validator import validate_dependencies
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import carrington, jd_fm_Carrington
from imap_l3_processing.glows.l3bc.models import CRToProcess


def read_glows_l3a_data(cdf: CDF) -> GlowsL3LightCurve:
    epoch_delta_s = cdf[EPOCH_DELTA_CDF_VAR_NAME][...] / 1e9
    epoch_delta = [timedelta(seconds=x) for x in epoch_delta_s]
    return GlowsL3LightCurve(None,
                             photon_flux=cdf[PHOTON_FLUX_CDF_VAR_NAME][...],
                             photon_flux_uncertainty=cdf[PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME][...],
                             raw_histogram=cdf[RAW_HISTOGRAM_CDF_VAR_NAME][...],
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


def find_unprocessed_carrington_rotations(l3a_inputs: list[dict], l3b_inputs: list[dict],
                                          dependencies: GlowsInitializerAncillaryDependencies) -> [
    CRToProcess]:
    l3bs_carringtons: set = set()
    for l3b in l3b_inputs:
        current_date = get_astropy_time_from_yyyymmdd(l3b["start_date"]) + TimeDelta(1, format='jd')
        current_rounded_cr = int(carrington(current_date.jd))
        l3bs_carringtons.add(current_rounded_cr)

    sorted_l3a_inputs = sorted(l3a_inputs, key=lambda entry: entry['start_date'])

    l3as_by_carrington: dict = defaultdict(list)

    latest_l3a_file = get_astropy_time_from_yyyymmdd(sorted_l3a_inputs[-1]["start_date"])

    for index, l3a in enumerate(sorted_l3a_inputs):
        current_date: Time = get_astropy_time_from_yyyymmdd(l3a["start_date"])
        current_rounded_cr = int(carrington(current_date.jd))

        tomorrow = current_date + TimeDelta(1, format="jd")
        tomorrow_rounded_cr = int(carrington(tomorrow.jd))

        if tomorrow_rounded_cr - current_rounded_cr == 1:
            l3as_by_carrington[tomorrow_rounded_cr].append(l3a['file_path'])

        l3as_by_carrington[current_rounded_cr].append(l3a['file_path'])

    crs_to_process = []
    for carrington_number, l3a_files in l3as_by_carrington.items():
        if carrington_number not in l3bs_carringtons:
            carrington_start_date = jd_fm_Carrington(float(carrington_number))
            date_time = Time(carrington_start_date, format='jd')
            date_time.format = 'iso'
            carrington_end_date_non_inclusive = jd_fm_Carrington(carrington_number + 1)
            date_time_end_date = Time(carrington_end_date_non_inclusive, format='jd')
            date_time_end_date.format = 'iso'

            if latest_l3a_file < date_time_end_date + dependencies.initializer_time_buffer:
                continue

            is_valid = validate_dependencies(date_time_end_date, dependencies.initializer_time_buffer,
                                             dependencies.omni2_data_path, dependencies.f107_index_file_path,
                                             dependencies.lyman_alpha_path)

            if not is_valid:
                continue

            crs_to_process.append(CRToProcess(
                l3a_paths=l3a_files,
                cr_start_date=date_time,
                cr_end_date=date_time_end_date,
                cr_rotation_number=carrington_number,
            ))

    return crs_to_process


def get_astropy_time_from_yyyymmdd(date_string: str) -> Time:
    return Time(f'{date_string[0:4]}-{date_string[4:6]}-{date_string[6:8]}')


def archive_dependencies(cr_to_process: CRToProcess, version: str,
                         ancillary_dependencies: GlowsInitializerAncillaryDependencies) -> Path:
    start_date = cr_to_process.cr_start_date.strftime("%Y%m%d")
    filename = f"imap_glows_l3b-archive_{start_date}_{version}.zip"
    json_filename = "cr_to_process.json"
    with ZipFile(filename, "w", ZIP_DEFLATED) as file:
        file.write(ancillary_dependencies.lyman_alpha_path)
        file.write(ancillary_dependencies.omni2_data_path)
        file.write(ancillary_dependencies.f107_index_file_path)
        with open(json_filename, "w") as json_file:
            cr = {"cr_rotation_number": cr_to_process.cr_rotation_number,
                  "l3a_paths": cr_to_process.l3a_paths,
                  "cr_start_date": cr_to_process.cr_start_date.value,
                  "cr_end_date": cr_to_process.cr_end_date.value,
                  "bad_days_list": ancillary_dependencies.bad_days_list,
                  "pipeline_settings": ancillary_dependencies.pipeline_settings,
                  "waw_helioion_mp": ancillary_dependencies.waw_helioion_mp_path,
                  "uv_anisotropy": ancillary_dependencies.uv_anisotropy_path
                  }
            dump(cr, json_file)
        file.write(json_filename)
    return Path(filename)


def make_l3b_data_with_fill(dependencies: GlowsL3BCDependencies):
    model = {}
    model['header'] = {
        'ancillary_data_files': dependencies.ancillary_files,
        'external_dependeciens': dependencies.external_files,
        'l3a_input_files_name': []
    }
    uv_anisotropy_file = dependencies.ancillary_files['uv_anisotropy']
    model['ion_rate_profile'] = {}
    model['CR'] = dependencies.carrington_rotation_number

    with open(uv_anisotropy_file, 'r') as file:
        text = file.read()
        contents = json.loads(text)
        model['uv_anisotropy_factor'] = contents['anisotropy_factor']
        model['ion_rate_profile']['lat_grid'] = contents['lat_grid']

    model['ion_rate_profile']['sum_rate'] = np.full(len(contents['lat_grid']), np.nan)
    model['ion_rate_profile']['ph_rate'] = np.full(len(contents['lat_grid']), np.nan)
    model['ion_rate_profile']['cx_rate'] = np.full(len(contents['lat_grid']), np.nan)
    model['ion_rate_profile']['sum_uncert'] = np.full(len(contents['lat_grid']), np.nan)
    model['ion_rate_profile']['ph_uncert'] = np.full(len(contents['lat_grid']), np.nan)
    model['ion_rate_profile']['cx_uncert'] = np.full(len(contents['lat_grid']), np.nan)

    return model


def make_l3c_data_with_fill():
    raise NotImplemented
