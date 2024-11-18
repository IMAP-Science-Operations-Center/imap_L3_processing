from dataclasses import dataclass

from spacepy.pycdf import CDF

from imap_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR
from imap_processing.models import UpstreamDataDependency
from imap_processing.utils import download_dependency


@dataclass
class GlowsL3ADependencies:
    data: CDF
    number_of_bins: int

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        dependency = next(dep
                          for dep in dependencies if dep.descriptor.startswith(GLOWS_L2_DESCRIPTOR))

        l2_cdf_path = download_dependency(dependency)
        cdf = CDF(str(l2_cdf_path))

        num_bin_dependency = UpstreamDataDependency("glows", "l2", None, None, "latest",
                                                    descriptor="histogram-number-of-bins-text-not-cdf")
        num_of_bin = int(download_dependency(num_bin_dependency).read_text())
        return cls(cdf, num_of_bin)
