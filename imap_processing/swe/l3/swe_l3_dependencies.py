from dataclasses import dataclass
from typing import TypedDict

from imap_processing.models import MagL1dData
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData
from imap_processing.swe.l3.models import SweL2Data


class SweConfiguration(TypedDict):
    geometric_fractions: list[float]
    pitch_angle_bins: list[float]
    pitch_angle_delta: list[float]
    energy_bins: list[float]
    energy_delta_plus: list[float]
    energy_delta_minus: list[float]


@dataclass
class SweL3Dependencies:
    swe_l2_data: SweL2Data
    mag_l1d_data: MagL1dData
    swapi_l3a_proton_data: SwapiL3ProtonSolarWindData
    configuration: SweConfiguration
