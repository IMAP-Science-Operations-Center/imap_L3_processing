from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imap_data_access
from imap_data_access import ScienceFilePath
from imap_data_access.file_validation import generate_imap_file_path
from spacepy.pycdf import CDF

from imap_l3_processing.hi.l3.models import HiMapData, HiL1cData, GlowsL3eData, HiIntensityMapData
from imap_l3_processing.hi.l3.utils import read_hi_l2_data, read_hi_l1c_data, read_glows_l3e_data
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency, download_dependency_from_path


def find_glows_l3e_dependencies(l1c_filenames: list[str]) -> list[str]:
    dates = [datetime.strptime(ScienceFilePath(l1c_filename).start_date, "%Y%m%d") for l1c_filename in l1c_filenames]

    start_date = min(dates).strftime("%Y%m%d")
    end_date = max(dates).strftime("%Y%m%d")

    sensor = l1c_filenames[0].split("_")[3][:2]
    descriptor = f"survival-probabilities-hi-{sensor}"

    survival_probabilities = [result["file_path"] for result in imap_data_access.query(instrument="glows",
                                                                                       data_level="l3e",
                                                                                       descriptor=descriptor,
                                                                                       start_date=start_date,
                                                                                       end_date=end_date,
                                                                                       version="latest")]

    return survival_probabilities


@dataclass
class HiL3SurvivalDependencies:
    l2_data: HiIntensityMapData
    hi_l1c_data: list[HiL1cData]
    glows_l3e_data: list[GlowsL3eData]

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]) -> HiL3SurvivalDependencies:
        upstream_map_dependency = next(dep for dep in dependencies if dep.data_level == "l2")
        map_file_path = download_dependency(upstream_map_dependency)
        hi_l1c_paths = []
        with CDF(str(map_file_path)) as l2_map:
            map_input_paths = [generate_imap_file_path(file) for file in l2_map.attrs["Parents"]]
            l1c_file_names = [map_input_file.filename.name for map_input_file in map_input_paths if
                              isinstance(map_input_file, ScienceFilePath) and map_input_file.data_level == "l1c"]
            for parent in l1c_file_names:
                hi_l1c_paths.append(download_dependency_from_path(parent))

        glows_l3e_file_names = find_glows_l3e_dependencies(l1c_file_names)
        glows_file_paths = [download_dependency_from_path(path) for path in glows_l3e_file_names]
        return cls.from_file_paths(map_file_path, hi_l1c_paths, glows_file_paths)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, hi_l1c_paths: list[Path],
                        glows_l3e_paths: list[Path]) -> HiL3SurvivalDependencies:
        glows_l3e_data = list(map(read_glows_l3e_data, glows_l3e_paths))
        l1c_data = list(map(read_hi_l1c_data, hi_l1c_paths))

        return cls(l2_data=read_hi_l2_data(map_file_path), hi_l1c_data=l1c_data, glows_l3e_data=glows_l3e_data)
