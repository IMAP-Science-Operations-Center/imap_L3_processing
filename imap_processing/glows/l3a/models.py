from dataclasses import dataclass
from datetime import datetime

import numpy as np

from imap_processing.models import DataProduct, DataProductVariable

PHOTON_FLUX_CDF_VAR_NAME = 'photon_flux'
EXPOSURE_TIMES_CDF_VAR_NAME = 'exposure_times'
NUM_OF_BINS_CDF_VAR_NAME = 'number_of_bins'
BINS_CDF_VAR_NAME = 'bins'
EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"


@dataclass
class GlowsL2Data:
    start_time: datetime
    end_time: datetime
    histogram_flag_array: np.ndarray[bool]
    photon_flux: np.ndarray[float]
    flux_uncertainties: np.ndarray[float]
    spin_angle: np.ndarray[float]
    exposure_times: np.ndarray[float]
    epoch: np.ndarray[datetime]


@dataclass
class GlowsL3LightCurve(DataProduct):
    photon_flux: np.ndarray[float]
    exposure_times: np.ndarray[float]
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(PHOTON_FLUX_CDF_VAR_NAME, self.photon_flux),
            DataProductVariable(EXPOSURE_TIMES_CDF_VAR_NAME, self.exposure_times),
            DataProductVariable(NUM_OF_BINS_CDF_VAR_NAME, len(self.photon_flux[-1]), record_varying=False),
            DataProductVariable(BINS_CDF_VAR_NAME, np.arange(len(self.photon_flux[-1])), record_varying=False),
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta)
        ]
