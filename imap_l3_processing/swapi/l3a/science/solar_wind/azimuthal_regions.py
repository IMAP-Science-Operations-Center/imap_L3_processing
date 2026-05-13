from typing import NamedTuple


class Region(NamedTuple):
    # exactly one of the two flags is True (see the REGION_* constants below)
    is_sunglasses: bool
    is_open_aperture: bool
    azimuth_sign: int  # 0 for SG (centered on boresight), ±1 for OA halves


REGION_SUNGLASSES = Region(True, False, 0)
REGION_OPEN_APERTURE_NEG = Region(False, True, -1)
REGION_OPEN_APERTURE_POS = Region(False, True, +1)
