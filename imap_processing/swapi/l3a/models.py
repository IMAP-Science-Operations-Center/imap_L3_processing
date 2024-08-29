import abc
from dataclasses import dataclass, field
from typing import List

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF
from uncertainties.unumpy import nominal_values, std_devs

from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"
PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_delta"


class DataProduct(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write_cdf(self, file_path: str, version: str):
        raise NotImplemented


@dataclass
class SwapiL3ProtonSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_speed: np.ndarray[float]

    def write_cdf(self, file_path: str, version: str):
        with CDF(file_path, '') as l3_cdf:
            attribute_manager = ImapAttributeManager()
            attribute_manager.add_global_attribute("Logical_file_id", file_path)
            attribute_manager.add_global_attribute("Data_version", version)
            attribute_manager.add_instrument_attrs("swapi", "l3a")
            for k, v in attribute_manager.get_global_attributes().items():
                l3_cdf.attrs[k] = v

            l3_cdf[EPOCH_CDF_VAR_NAME] = self.epoch
            l3_cdf[EPOCH_CDF_VAR_NAME].type(pycdf.const.CDF_TIME_TT2000)
            _add_variable_attributes(attribute_manager, l3_cdf, EPOCH_CDF_VAR_NAME)

            l3_cdf[PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME] = nominal_values(self.proton_sw_speed)
            _add_variable_attributes(attribute_manager, l3_cdf, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME)

            l3_cdf[PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME] = std_devs(self.proton_sw_speed)
            _add_variable_attributes(attribute_manager, l3_cdf, PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME)

            l3_cdf.new(EPOCH_DELTA_CDF_VAR_NAME, THIRTY_SECONDS_IN_NANOSECONDS, recVary=False)
            _add_variable_attributes(attribute_manager, l3_cdf, EPOCH_DELTA_CDF_VAR_NAME)


def _add_variable_attributes(attribute_manager, cdf, variable_name):
    for k, v in attribute_manager.get_variable_attributes(variable_name).items():
        cdf[variable_name].attrs[k] = v


@dataclass
class SwapiL3AlphaSolarWindData(DataProduct):
    def write_cdf(self, file_path: str, version: str):
        pass

    epoch: np.ndarray[float]
    alpha_sw_speed: np.ndarray[float]


@dataclass
class SwapiL2Data:
    epoch: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    spin_angles: np.ndarray[float]  # not currently in the L2 cdf, is in the sample data provided by Bishwas
    coincidence_count_rate_uncertainty: np.ndarray[float]