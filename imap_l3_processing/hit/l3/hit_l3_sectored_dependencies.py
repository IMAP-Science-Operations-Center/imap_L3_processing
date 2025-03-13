from __future__ import annotations
from dataclasses import dataclass

from imap_l3_processing.hit.l3.models import HitL2Data
from imap_l3_processing.hit.l3.utils import read_l2_hit_data
from imap_l3_processing.models import UpstreamDataDependency, MagL1dData
from imap_l3_processing.utils import download_dependency, read_l1d_mag_data

HIT_L2_DESCRIPTOR = "sectoredrates"
MAG_L1D_DESCRIPTOR = "mago-normal"


@dataclass
class HITL3SectoredDependencies:
    data: HitL2Data
    mag_l1d_data: MagL1dData

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]) -> HITL3SectoredDependencies:
        try:
            hit_data_dependency = next(
                dependency for dependency in dependencies if dependency.descriptor == HIT_L2_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {HIT_L2_DESCRIPTOR} dependency.")
        try:
            mag_dependency = next(
                dependency for dependency in dependencies if dependency.descriptor == MAG_L1D_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {MAG_L1D_DESCRIPTOR} dependency.")

        hit_data_path = download_dependency(hit_data_dependency)
        mag_data_path = download_dependency(mag_dependency)

        mag_data = read_l1d_mag_data(mag_data_path)
        hit_data = read_l2_hit_data(hit_data_path)

        return HITL3SectoredDependencies(data=hit_data, mag_l1d_data=mag_data)
