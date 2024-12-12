from dataclasses import dataclass
from pathlib import Path

from spacepy.pycdf import CDF

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR, GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR, \
    GLOWS_CALIBRATION_DATA_DESCRIPTOR, \
    GLOWS_EXTRA_HELIOSPHERIC_BACKGROUND_DESCRIPTOR, GLOWS_PIPELINE_SETTINGS_DESCRIPTOR
from imap_processing.glows.l3a.models import GlowsL2Data
from imap_processing.glows.l3a.utils import read_l2_glows_data
from imap_processing.models import UpstreamDataDependency
from imap_processing.utils import download_dependency


@dataclass
class GlowsL3ADependencies:
    data: GlowsL2Data
    ancillary_files: dict[str, Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        dependency = next(dep
                          for dep in dependencies if dep.descriptor.startswith(GLOWS_L2_DESCRIPTOR))

        l2_cdf_path = download_dependency(dependency)
        cdf = CDF(str(l2_cdf_path))
        l2_glows_data = read_l2_glows_data(cdf)

        calibration_dependency = UpstreamDataDependency("glows", "l3a", None, None, "latest",
                                                        descriptor=GLOWS_CALIBRATION_DATA_DESCRIPTOR)
        time_dependent_background_dependency = UpstreamDataDependency("glows", "l3a", dependency.start_date,
                                                                      dependency.end_date,
                                                                      "latest",
                                                                      descriptor=GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR)
        extra_heliospheric_background_dependency = UpstreamDataDependency("glows", "l3a", None, None,
                                                                          "latest",
                                                                          descriptor=GLOWS_EXTRA_HELIOSPHERIC_BACKGROUND_DESCRIPTOR)
        settings_dependency = UpstreamDataDependency("glows", "l3a", None, None,
                                                     "latest",
                                                     descriptor=GLOWS_PIPELINE_SETTINGS_DESCRIPTOR)
        calibration_path = download_dependency(calibration_dependency)
        extra_heliospheric_background = download_dependency(extra_heliospheric_background_dependency)
        time_dependent_background_path = download_dependency(time_dependent_background_dependency)
        settings_path = download_dependency(settings_dependency)

        ancillary_files = {
            "calibration_data": calibration_path,
            "settings": settings_path,
            "time_dependent_bckgrd": time_dependent_background_path,
            "extra_heliospheric_bckgrd": extra_heliospheric_background,
        }
        return cls(l2_glows_data, ancillary_files)
