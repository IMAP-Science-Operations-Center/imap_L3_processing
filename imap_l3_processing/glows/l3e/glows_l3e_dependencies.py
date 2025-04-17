from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.glows.descriptors import GLOWS_L3D_DESCRIPTOR
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency


@dataclass
class GlowsL3EDependencies:
    l3d_data: Path
    energy_grid: Path
    tess_xyz_8: Path
    tess_ang16: Path

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        l3d_upstream_dependency = next(dep for dep in dependencies if dep.descriptor.startswith(GLOWS_L3D_DESCRIPTOR))

        l3d_dependency = download_dependency(l3d_upstream_dependency)

        energy_grid_dependency = download_dependency(UpstreamDataDependency("glows", None, None, None,
                                                                            "v001",
                                                                            descriptor="energy-grid"))
        tess_xyz_8_dependency = download_dependency(UpstreamDataDependency("glows", None, None, None,
                                                                           "v001",
                                                                           descriptor="tess-xyz-8"))
        tess_ang16_dependency = download_dependency(UpstreamDataDependency("glows", None, None, None,
                                                                           "v001",
                                                                           descriptor="tess-ang-16"))

        return cls(l3d_dependency, energy_grid_dependency, tess_xyz_8_dependency, tess_ang16_dependency)
