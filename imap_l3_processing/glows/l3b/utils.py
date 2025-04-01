from collections import defaultdict
from datetime import timedelta
from json import dump
from zipfile import ZipFile, ZIP_DEFLATED

from astropy.time import Time
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
from imap_l3_processing.glows.l3b.dependency_validator import validate_dependencies
from imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3b.l3bc_toolkit.funcs import carrington, jd_fm_Carrington
from imap_l3_processing.glows.l3b.models import CRToProcess


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
        current_date = get_astropy_time_from_yyyymmdd(l3b["start_date"])
        current_rounded_cr = int(carrington(current_date.jd))
        l3bs_carringtons.add(current_rounded_cr)

    sorted_l3a_inputs = sorted(l3a_inputs, key=lambda entry: entry['start_date'])

    l3as_by_carrington: dict = defaultdict(list)

    prior_cr: int = 0
    prior_jd: float = 1.0
    for index, l3a in enumerate(sorted_l3a_inputs):
        current_date: Time = get_astropy_time_from_yyyymmdd(l3a["start_date"])
        current_rounded_cr = int(carrington(current_date.jd))
        if (current_rounded_cr - prior_cr == 1) and (current_date.jd - prior_jd == 1):
            l3as_by_carrington[current_rounded_cr].append(sorted_l3a_inputs[index - 1]['file_path'])
        l3as_by_carrington[current_rounded_cr].append(l3a['file_path'])
        prior_jd = current_date.jd
        prior_cr = current_rounded_cr

    crs_to_process = []
    for carrington_number, l3a_files in l3as_by_carrington.items():
        if carrington_number not in l3bs_carringtons and len(l3a_files) == 28:
            carrington_start_date = jd_fm_Carrington(float(carrington_number))
            date_time = Time(carrington_start_date, format='jd')
            date_time.format = 'iso'
            carrington_end_date_non_inclusive = jd_fm_Carrington(carrington_number + 1)
            date_time_end_date = Time(carrington_end_date_non_inclusive, format='jd')
            date_time_end_date.format = 'iso'
            is_valid = validate_dependencies(date_time, date_time_end_date + timedelta(days=1),
                                             dependencies.omni2_data_path, dependencies.f107_index_file_path,
                                             dependencies.lyman_alpha_path)

            if not is_valid:
                continue

            date_midpoint = (date_time.jd + date_time_end_date.jd) / 2
            date_time_midpoint = Time(date_midpoint, format='jd')

            crs_to_process.append(CRToProcess(
                l3a_paths=l3a_files,
                cr_midpoint=date_time_midpoint.strftime('%Y%m%d'),
                cr_rotation_number=carrington_number,
                uv_anisotropy=dependencies.uv_anisotropy_path,
                waw_helioion_mp=dependencies.waw_helioion_mp_path
            ))

    return crs_to_process


def get_astropy_time_from_yyyymmdd(date_string: str) -> Time:
    return Time(f'{date_string[0:4]}-{date_string[4:6]}-{date_string[6:8]}')


def archive_dependencies(cr_to_process: CRToProcess, version: str,
                         ancillary_dependencies: GlowsInitializerAncillaryDependencies):
    filename = f"imap_glows_l3pre-b_l3b-archive_{cr_to_process.cr_midpoint}_{version}.zip"
    json_filename = "cr_to_process.json"
    with ZipFile(filename, "w", ZIP_DEFLATED) as file:
        file.write(ancillary_dependencies.lyman_alpha_path)
        file.write(ancillary_dependencies.omni2_data_path)
        file.write(ancillary_dependencies.f107_index_file_path)
        with open(json_filename, "w") as json_file:
            dump(cr_to_process, json_file)
        file.write(json_filename)
