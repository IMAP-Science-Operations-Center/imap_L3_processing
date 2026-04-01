from datetime import datetime
from typing import Iterable

from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.swapi.l3a.models import SwapiL2Data


def read_l2_swapi_data(cdf: CDF) -> SwapiL2Data:
    sci_start_times = pycdf.lib.v_datetime_to_tt2000(
        [datetime.fromisoformat(x) for x in cdf["sci_start_time"][...]])
    return SwapiL2Data(sci_start_times,
                       read_numeric_variable(cdf["esa_energy"]),
                       read_numeric_variable(cdf["swp_coin_rate"]),
                       read_numeric_variable(cdf["swp_coin_rate_stat_uncert_plus"]))

def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    i = 0
    while i < len(data.sci_start_time):
        yield SwapiL2Data(
            data.sci_start_time[i:i + chunk_size],
            data.energy[i:i + chunk_size],
            data.coincidence_count_rate[i:i + chunk_size],
            data.coincidence_count_rate_uncertainty[i:i + chunk_size]
        )
        i += chunk_size
