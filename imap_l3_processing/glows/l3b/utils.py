from datetime import timedelta

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
