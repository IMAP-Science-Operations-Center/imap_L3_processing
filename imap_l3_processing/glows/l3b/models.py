from dataclasses import dataclass


@dataclass
class CRToProcess:
    l3a_paths: list[str]
    cr_midpoint: str
    cr_rotation_number: int
    uv_anisotropy: str
    waw_helioion_mp: str
