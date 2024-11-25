from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR, GLOWS_NUMBER_OF_BINS_ANCILLARY_DESCRIPTOR, \
    GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR, GLOWS_TIME_INDEPENDENT_BACKGROUND_DESCRIPTOR, \
    GLOWS_BAD_ANGLE_FLAG_CONFIG_DESCRIPTOR
from imap_processing.glows.l3a.models import GlowsL2Data
from imap_processing.glows.l3a.science.bad_angle_flag_configuration import BadAngleFlagConfiguration
from imap_processing.glows.l3a.science.time_independent_background_lookup_table import \
    TimeIndependentBackgroundLookupTable
from imap_processing.glows.l3a.utils import read_l2_glows_data
from imap_processing.models import UpstreamDataDependency
from imap_processing.utils import download_dependency


@dataclass
class GlowsL3ADependencies:
    data: GlowsL2Data
    number_of_bins: int
    time_dependent_background: ndarray[float]
    time_independent_background_table: TimeIndependentBackgroundLookupTable
    bad_angle_flag_configuration: BadAngleFlagConfiguration

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        dependency = next(dep
                          for dep in dependencies if dep.descriptor.startswith(GLOWS_L2_DESCRIPTOR))

        l2_cdf_path = download_dependency(dependency)
        cdf = CDF(str(l2_cdf_path))
        l2_glows_data = read_l2_glows_data(cdf)

        num_bin_dependency = UpstreamDataDependency("glows", "l2", None, None, "latest",
                                                    descriptor=GLOWS_NUMBER_OF_BINS_ANCILLARY_DESCRIPTOR)
        num_of_bin = int(download_dependency(num_bin_dependency).read_text())

        background_dependency = UpstreamDataDependency("glows", "l2", dependency.start_date, dependency.end_date,
                                                       "latest",
                                                       descriptor=GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR)
        background_text = download_dependency(background_dependency)
        background = np.loadtxt(background_text)

        time_independent_background_dependency = UpstreamDataDependency("glows", "l2", None, None,
                                                                        "latest",
                                                                        descriptor=GLOWS_TIME_INDEPENDENT_BACKGROUND_DESCRIPTOR)
        time_independent_background_file_path = download_dependency(time_independent_background_dependency)
        flag_configuration_dependency = UpstreamDataDependency("glows", "l2", None, None,
                                                               "latest",
                                                               descriptor=GLOWS_BAD_ANGLE_FLAG_CONFIG_DESCRIPTOR)
        flag_configuration_file_path = download_dependency(flag_configuration_dependency)

        return cls(l2_glows_data, num_of_bin, background,
                   TimeIndependentBackgroundLookupTable.from_file(time_independent_background_file_path),
                   BadAngleFlagConfiguration.from_file(flag_configuration_file_path))
