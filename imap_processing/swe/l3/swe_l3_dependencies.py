from dataclasses import dataclass

from imap_processing.models import MagL1dData
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData
from imap_processing.swe.l3.models import SweL2Data


@dataclass
class SweL3Dependencies:
    swe_l2_data: SweL2Data
    mag_l1d_data: MagL1dData
    swapi_l3a_proton_data: SwapiL3ProtonSolarWindData
