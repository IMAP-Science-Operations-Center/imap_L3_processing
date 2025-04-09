from dataclasses import dataclass
from datetime import datetime

import numpy as np
from spacepy import pycdf

from imap_l3_processing.models import InputMetadata, DataProduct, DataProductVariable

EPOCH_VAR_NAME = "Epoch"
LAT_VAR_NAME = "lat"
LONG_VAR_NAME = "lon"
ENERGY_VAR_NAME = "bin"
FLUX_VAR_NAME = "intensity"
VARIANCE_VAR_NAME = "variance"
ENERGY_DELTAS_VAR_NAME = "energy_deltas"
COUNTS_VAR_NAME = "counts"
COUNTS_UNCERTAINTY_VAR_NAME = "counts_uncertainty"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
EXPOSURE_VAR_NAME = "exposure"
SENSITIVITY_VAR_NAME = "sensitivity"
SPECTRAL_FIT_INDEX_VAR_NAME = "spectral_fit_index"
SPECTRAL_FIT_INDEX_ERROR_VAR_NAME = "spectral_fit_index_error"


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
            DataProductVariable(LAT_VAR_NAME, self.lat, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(LONG_VAR_NAME, self.lon, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(ENERGY_VAR_NAME, self.energy, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(FLUX_VAR_NAME, self.flux, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(VARIANCE_VAR_NAME, self.variance, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(ENERGY_DELTAS_VAR_NAME, self.energy_deltas, cdf_data_type=pycdf.const.CDF_DOUBLE),
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


class HiL3SurvivalCorrectedDataProduct(DataProduct, HiMapData):
    def to_data_product_variables(self) -> list[DataProductVariable]:
        return []


@dataclass
class HiL1cData:
    epoch: datetime
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray


@dataclass
class GlowsL3eData:
    epoch: datetime
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray
