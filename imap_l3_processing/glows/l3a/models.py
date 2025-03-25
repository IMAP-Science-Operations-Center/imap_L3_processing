from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
from spacepy import pycdf

from imap_l3_processing.models import DataProduct, DataProductVariable

PHOTON_FLUX_CDF_VAR_NAME = 'photon_flux'
PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME = 'photon_flux_uncertainty'
RAW_HISTOGRAM_CDF_VAR_NAME = 'raw_histogram'
EXPOSURE_TIMES_CDF_VAR_NAME = 'exposure_times'
NUM_OF_BINS_CDF_VAR_NAME = 'number_of_bins'
EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
SPIN_ANGLE_CDF_VAR_NAME = "spin_angle"
SPIN_ANGLE_DELTA_CDF_VAR_NAME = "spin_angle_delta"
LATITUDE_CDF_VAR_NAME = "ecliptic_lat"
LONGITUDE_CDF_VAR_NAME = "ecliptic_lon"
EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME = "extra_heliospheric_bckgrd"
TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME = "time_dependent_bckgrd"

FILTER_TEMPERATURE_AVERAGE_CDF_VAR_NAME = "filter_temperature_average"
FILTER_TEMPERATURE_STD_DEV_CDF_VAR_NAME = "filter_temperature_std_dev"
HV_VOLTAGE_AVERAGE_CDF_VAR_NAME = "hv_voltage_average"
HV_VOLTAGE_STD_DEV_CDF_VAR_NAME = "hv_voltage_std_dev"
SPIN_PERIOD_AVERAGE_CDF_VAR_NAME = "spin_period_average"
SPIN_PERIOD_STD_DEV_CDF_VAR_NAME = "spin_period_std_dev"
SPIN_PERIOD_GROUND_AVERAGE_CDF_VAR_NAME = "spin_period_ground_average"
SPIN_PERIOD_GROUND_STD_DEV_CDF_VAR_NAME = "spin_period_ground_std_dev"
PULSE_LENGTH_AVERAGE_CDF_VAR_NAME = "pulse_length_average"
PULSE_LENGTH_STD_DEV_CDF_VAR_NAME = "pulse_length_std_dev"
POSITION_ANGLE_OFFSET_AVERAGE_CDF_VAR_NAME = "position_angle_offset_average"
POSITION_ANGLE_OFFSET_STD_DEV_CDF_VAR_NAME = "position_angle_offset_std_dev"
SPIN_AXIS_ORIENTATION_AVERAGE_CDF_VAR_NAME = "spin_axis_orientation_average"
SPIN_AXIS_ORIENTATION_STD_DEV_CDF_VAR_NAME = "spin_axis_orientation_std_dev"
SPACECRAFT_LOCATION_AVERAGE_CDF_VAR_NAME = "spacecraft_location_average"
SPACECRAFT_LOCATION_STD_DEV_CDF_VAR_NAME = "spacecraft_location_std_dev"
SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME = "spacecraft_velocity_average"
SPACECRAFT_VELOCITY_STD_DEV_CDF_VAR_NAME = "spacecraft_velocity_std_dev"


class GlowsL2LightCurve(TypedDict):
    spin_angle: np.ndarray[float]
    histogram_flag_array: np.ndarray[bool]
    photon_flux: np.ndarray[float]
    exposure_times: np.ndarray[float]
    flux_uncertainties: np.ndarray[float]
    raw_histogram: np.ndarray[int]
    ecliptic_lon: np.ndarray[float]
    ecliptic_lat: np.ndarray[float]


class GlowsL2Header(TypedDict):
    flight_software_version: int
    pkts_file_name: str
    ancillary_data_files: list[str]


class GlowsLatLon(TypedDict):
    lat: float
    lon: float


class XYZ(TypedDict):
    x: float
    y: float
    z: float


class GlowsL2Data(TypedDict):
    start_time: str
    end_time: str
    daily_lightcurve: GlowsL2LightCurve
    number_of_bins: int
    spin_axis_orientation_average: GlowsLatLon
    spin_axis_orientation_std_dev: GlowsLatLon
    identifier: int
    filter_temperature_average: float
    filter_temperature_std_dev: float
    hv_voltage_average: float
    hv_voltage_std_dev: float
    spin_period_average: float
    spin_period_std_dev: float
    spin_period_ground_average: float
    spin_period_ground_std_dev: float
    pulse_length_average: float
    pulse_length_std_dev: float
    position_angle_offset_average: float
    position_angle_offset_std_dev: float
    spacecraft_location_average: XYZ
    spacecraft_location_std_dev: XYZ
    spacecraft_velocity_average: XYZ
    spacecraft_velocity_std_dev: XYZ
    header: GlowsL2Header
    l2_file_name: str


@dataclass
class GlowsL3LightCurve(DataProduct):
    photon_flux: np.ndarray[float]
    photon_flux_uncertainty: np.ndarray[float]
    raw_histogram: np.ndarray[float]
    exposure_times: np.ndarray[float]
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[float]
    spin_angle: np.ndarray[float]
    spin_angle_delta: np.ndarray[float]
    latitude: np.ndarray[float]
    longitude: np.ndarray[float]
    extra_heliospheric_background: np.ndarray[float]
    time_dependent_background: np.ndarray[float]

    filter_temperature_average: np.ndarray[float]
    filter_temperature_std_dev: np.ndarray[float]
    hv_voltage_average: np.ndarray[float]
    hv_voltage_std_dev: np.ndarray[float]
    spin_period_average: np.ndarray[float]
    spin_period_std_dev: np.ndarray[float]
    spin_period_ground_average: np.ndarray[float]
    spin_period_ground_std_dev: np.ndarray[float]
    pulse_length_average: np.ndarray[float]
    pulse_length_std_dev: np.ndarray[float]
    position_angle_offset_average: np.ndarray[float]
    position_angle_offset_std_dev: np.ndarray[float]

    spin_axis_orientation_average: np.ndarray[float]
    spin_axis_orientation_std_dev: np.ndarray[float]
    spacecraft_location_average: np.ndarray[float]
    spacecraft_location_std_dev: np.ndarray[float]
    spacecraft_velocity_average: np.ndarray[float]
    spacecraft_velocity_std_dev: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(PHOTON_FLUX_CDF_VAR_NAME, self.photon_flux),
            DataProductVariable(PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME, self.photon_flux_uncertainty),
            DataProductVariable(RAW_HISTOGRAM_CDF_VAR_NAME, self.raw_histogram, cdf_data_type=pycdf.const.CDF_UINT4),
            DataProductVariable(EXPOSURE_TIMES_CDF_VAR_NAME, self.exposure_times),
            DataProductVariable(NUM_OF_BINS_CDF_VAR_NAME, len(self.photon_flux[-1]), record_varying=False,
                                cdf_data_type=pycdf.const.CDF_UINT2),
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta, cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable(SPIN_ANGLE_CDF_VAR_NAME, self.spin_angle, record_varying=False),
            DataProductVariable(SPIN_ANGLE_DELTA_CDF_VAR_NAME, self.spin_angle_delta, record_varying=False),
            DataProductVariable(LATITUDE_CDF_VAR_NAME, self.latitude),
            DataProductVariable(LONGITUDE_CDF_VAR_NAME, self.longitude),
            DataProductVariable(EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME, self.extra_heliospheric_background),
            DataProductVariable(TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME, self.time_dependent_background),

            DataProductVariable(FILTER_TEMPERATURE_AVERAGE_CDF_VAR_NAME, self.filter_temperature_average),
            DataProductVariable(FILTER_TEMPERATURE_STD_DEV_CDF_VAR_NAME, self.filter_temperature_std_dev),
            DataProductVariable(HV_VOLTAGE_AVERAGE_CDF_VAR_NAME, self.hv_voltage_average),
            DataProductVariable(HV_VOLTAGE_STD_DEV_CDF_VAR_NAME, self.hv_voltage_std_dev),
            DataProductVariable(SPIN_PERIOD_AVERAGE_CDF_VAR_NAME, self.spin_period_average),
            DataProductVariable(SPIN_PERIOD_STD_DEV_CDF_VAR_NAME, self.spin_period_std_dev),
            DataProductVariable(SPIN_PERIOD_GROUND_AVERAGE_CDF_VAR_NAME, self.spin_period_ground_average),
            DataProductVariable(SPIN_PERIOD_GROUND_STD_DEV_CDF_VAR_NAME, self.spin_period_ground_std_dev),
            DataProductVariable(PULSE_LENGTH_AVERAGE_CDF_VAR_NAME, self.pulse_length_average),
            DataProductVariable(PULSE_LENGTH_STD_DEV_CDF_VAR_NAME, self.pulse_length_std_dev),
            DataProductVariable(POSITION_ANGLE_OFFSET_AVERAGE_CDF_VAR_NAME, self.position_angle_offset_average),
            DataProductVariable(POSITION_ANGLE_OFFSET_STD_DEV_CDF_VAR_NAME, self.position_angle_offset_std_dev),
            DataProductVariable(SPIN_AXIS_ORIENTATION_AVERAGE_CDF_VAR_NAME, self.spin_axis_orientation_average),
            DataProductVariable(SPIN_AXIS_ORIENTATION_STD_DEV_CDF_VAR_NAME, self.spin_axis_orientation_std_dev),
            DataProductVariable(SPACECRAFT_LOCATION_AVERAGE_CDF_VAR_NAME, self.spacecraft_location_average),
            DataProductVariable(SPACECRAFT_LOCATION_STD_DEV_CDF_VAR_NAME, self.spacecraft_location_std_dev),
            DataProductVariable(SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME, self.spacecraft_velocity_average),
            DataProductVariable(SPACECRAFT_VELOCITY_STD_DEV_CDF_VAR_NAME, self.spacecraft_velocity_std_dev),

            DataProductVariable("lon_lat", np.arange(2), record_varying=False),
            DataProductVariable("lon_lat_labels", ["lon", "lat"], record_varying=False),
            DataProductVariable("x_y_z", np.arange(3), record_varying=False),
            DataProductVariable("x_y_z_labels", ["X", "Y", "Z"], record_varying=False),
        ]
