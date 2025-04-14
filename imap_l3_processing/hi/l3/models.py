from dataclasses import dataclass
from datetime import datetime

import numpy as np
from spacepy import pycdf

from imap_l3_processing.models import InputMetadata, DataProduct, DataProductVariable

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
LAT_VAR_NAME = "lat"
LONG_VAR_NAME = "lon"
ENERGY_VAR_NAME = "energy"
ENERGY_DELTAS_VAR_NAME = "energy_deltas"
INTENSITY_VAR_NAME = "intensity"
VARIANCE_VAR_NAME = "variance"
COUNTS_VAR_NAME = "counts"
COUNTS_UNCERTAINTY_VAR_NAME = "counts_uncertainty"
EXPOSURE_VAR_NAME = "exposure"
SENSITIVITY_VAR_NAME = "sensitivity"
SPECTRAL_FIT_INDEX_VAR_NAME = "spectral_fit_index"
SPECTRAL_FIT_INDEX_ERROR_VAR_NAME = "spectral_fit_index_error"
ENERGY_LABEL_VAR_NAME = "energy_label"
LON_LABEL_VAR_NAME = "lon_label"
LAT_LABEL_VAR_NAME = "lat_label"


@dataclass
class HiMapData:
    epoch: np.ndarray
    energy: np.ndarray
    energy_deltas: np.ndarray
    counts: np.ndarray
    counts_uncertainty: np.ndarray
    epoch_delta: np.ndarray
    exposure: np.ndarray
    flux: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    sensitivity: np.ndarray
    variance: np.ndarray


@dataclass
class HiL3SpectralIndexDataProduct(DataProduct, HiMapData):
    spectral_fit_index: np.ndarray
    spectral_fit_index_error: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(LAT_VAR_NAME, self.lat, cdf_data_type=pycdf.const.CDF_DOUBLE,
                                record_varying=False),
            DataProductVariable(LONG_VAR_NAME, self.lon, cdf_data_type=pycdf.const.CDF_DOUBLE,
                                record_varying=False),
            DataProductVariable(ENERGY_VAR_NAME, self.energy, cdf_data_type=pycdf.const.CDF_DOUBLE,
                                record_varying=False),
            DataProductVariable(ENERGY_DELTAS_VAR_NAME, self.energy_deltas, cdf_data_type=pycdf.const.CDF_DOUBLE,
                                record_varying=False),
            DataProductVariable(INTENSITY_VAR_NAME, self.flux, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(VARIANCE_VAR_NAME, self.variance, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(COUNTS_VAR_NAME, self.counts, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(COUNTS_UNCERTAINTY_VAR_NAME, self.counts_uncertainty,
                                cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(EXPOSURE_VAR_NAME, self.exposure, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(SENSITIVITY_VAR_NAME, self.sensitivity, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(SPECTRAL_FIT_INDEX_VAR_NAME, self.spectral_fit_index,
                                cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(SPECTRAL_FIT_INDEX_ERROR_VAR_NAME, self.spectral_fit_index_error,
                                cdf_data_type=pycdf.const.CDF_DOUBLE),
        ]


@dataclass
class HiL3SurvivalCorrectedDataProduct(DataProduct, HiMapData):
    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_DELTAS_VAR_NAME, self.energy_deltas),
            DataProductVariable(COUNTS_VAR_NAME, self.counts),
            DataProductVariable(COUNTS_UNCERTAINTY_VAR_NAME, self.counts_uncertainty),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(EXPOSURE_VAR_NAME, self.exposure),
            DataProductVariable(INTENSITY_VAR_NAME, self.flux),
            DataProductVariable(LAT_VAR_NAME, self.lat),
            DataProductVariable(LONG_VAR_NAME, self.lon),
            DataProductVariable(SENSITIVITY_VAR_NAME, self.sensitivity),
            DataProductVariable(VARIANCE_VAR_NAME, self.variance),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, [f"Energy Bin {energy}" for energy in self.energy]),
            DataProductVariable(LON_LABEL_VAR_NAME, [f"Lon {lon}" for lon in self.lon]),
            DataProductVariable(LAT_LABEL_VAR_NAME, [f"Lat {lat}" for lat in self.lat]),
        ]


@dataclass
class HiL1cData:
    epoch: datetime
    epoch_j2000: np.ndarray
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray


@dataclass
class GlowsL3eData:
    epoch: datetime
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray
