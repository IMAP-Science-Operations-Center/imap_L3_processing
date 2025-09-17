from typing import Iterable

from spacepy.pycdf import CDF
from uncertainties.unumpy import uarray

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.swapi.l3a.models import SwapiL2Data, SwapiL3AlphaSolarWindData, EPOCH_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME, ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME


def read_l2_swapi_data(cdf: CDF) -> SwapiL2Data:
    return SwapiL2Data(cdf.raw_var("sci_start_time")[...],
                       read_numeric_variable(cdf["swp_esa_energy"]),
                       read_numeric_variable(cdf["swp_coin_rate"]),
                       read_numeric_variable(cdf["swp_coin_rate_stat_uncert_plus"]))


def read_l3a_alpha_sw_swapi_data(cdf: CDF) -> SwapiL3AlphaSolarWindData:
    alpha_sw_speed = uarray(cdf[ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME],
                            cdf[ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME])
    alpha_sw_temperature = uarray(cdf[ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME],
                                  cdf[ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME])
    alpha_sw_density = uarray(cdf[ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME],
                              cdf[ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME])
    return SwapiL3AlphaSolarWindData(None,
                                     epoch=cdf[EPOCH_CDF_VAR_NAME],
                                     alpha_sw_speed=alpha_sw_speed,
                                     alpha_sw_temperature=alpha_sw_temperature,
                                     alpha_sw_density=alpha_sw_density)


def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    i = 0
    while i < len(data.sci_start_time):
        yield SwapiL2Data(
            data.sci_start_time[i:i + chunk_size],
            data.energy,
            data.coincidence_count_rate[i:i + chunk_size],
            data.coincidence_count_rate_uncertainty[i:i + chunk_size]
        )
        i += chunk_size
