from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.hi.l3.models import HiL3Data
from imap_l3_processing.hi.l3.utils import read_hi_l2_data
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency

HI_L3_SPECTRAL_FIT_DESCRIPTOR = "spectral-fit-index"


@dataclass
class HiL3SpectralFitDependencies:
    hi_l3_data: HiL3Data

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        try:
            hi_l3_dependency = next(
                dependency for dependency in dependencies if dependency.instrument == "hi"
                and dependency.descriptor == HI_L3_SPECTRAL_FIT_DESCRIPTOR)
        except StopIteration:
            raise ValueError("Missing Hi dependency.")

        hi_l3_file = download_dependency(hi_l3_dependency)
        return cls.from_file_paths(hi_l3_file)

    @classmethod
    def from_file_paths(cls, hi_l3_path: Path):
        return cls(read_hi_l2_data(hi_l3_path))
