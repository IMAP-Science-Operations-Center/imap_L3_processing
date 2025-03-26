from dataclasses import dataclass

from astropy.time import Time


@dataclass
class CRToProcess:
    l3a_paths: list[str]
    carrington_end_date: Time
    carrington_start_date: Time
    carrington_rotation: int
