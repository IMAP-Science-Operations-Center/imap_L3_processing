from typing import Iterable

from spacepy.pycdf import CDF

from imap_l3_processing.swapi.l3a.models import SwapiL2Data, SwapiL3AlphaSolarWindData


def read_l2_swapi_data(cdf: CDF) -> SwapiL2Data:
    return SwapiL2Data(cdf.raw_var("epoch")[...],
                       cdf["energy"][...],
                       cdf["swp_coin_rate"][...],
                       cdf["swp_coin_unc"][...])


def read_l3a_alpha_sw_swapi_data(cdf: CDF) -> SwapiL3AlphaSolarWindData:
    return


def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    i = 0
    while i < len(data.epoch):
        yield SwapiL2Data(
            data.epoch[i:i + chunk_size],
            data.energy,
            data.coincidence_count_rate[i:i + chunk_size],
            data.coincidence_count_rate_uncertainty[i:i + chunk_size]
        )
        i += chunk_size
