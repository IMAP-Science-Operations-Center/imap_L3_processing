from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
from spacepy import pycdf

from imap_processing.models import DataProduct, DataProductVariable

PHOTON_FLUX_CDF_VAR_NAME = 'photon_flux'
PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME = 'photon_flux_uncertainty'
EXPOSURE_TIMES_CDF_VAR_NAME = 'exposure_times'
NUM_OF_BINS_CDF_VAR_NAME = 'number_of_bins'
BINS_CDF_VAR_NAME = 'bins'
EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
SPIN_ANGLE_CDF_VAR_NAME = "spin_angle"


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
    exposure_times: np.ndarray[float]
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[float]
    spin_angle: np.ndarray[float]
    latitude: np.ndarray[float]
    longitude: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(PHOTON_FLUX_CDF_VAR_NAME, self.photon_flux),
            DataProductVariable(PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME, self.photon_flux_uncertainty),
            DataProductVariable(EXPOSURE_TIMES_CDF_VAR_NAME, self.exposure_times),
            DataProductVariable(NUM_OF_BINS_CDF_VAR_NAME, len(self.photon_flux[-1]), record_varying=False,
                                cdf_data_type=pycdf.const.CDF_INT2),
            DataProductVariable(BINS_CDF_VAR_NAME, np.arange(len(self.photon_flux[-1])), record_varying=False),
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta, cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable(SPIN_ANGLE_CDF_VAR_NAME, self.spin_angle)
        ]
