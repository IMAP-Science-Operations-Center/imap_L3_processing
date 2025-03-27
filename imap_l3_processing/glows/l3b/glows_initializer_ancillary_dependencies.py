from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency, download_external_dependency

F107_FLUX_TABLE_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
LYMAN_ALPHA_COMPOSITE_INDEX_URL = "http://lasp.colorado.edu/data/timed_see/composite_lya/lyman_alpha_composite.nc"
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"


@dataclass
class GlowsInitializerAncillaryDependencies:
    uv_anisotropy_path: Path
    waw_helioion_mp_path: Path
    f107_index_file_path: Path
    lyman_alpha_path: Path
    omni2_data_path: Path

    @classmethod
    def fetch_dependencies(cls):
        uv_anisotropy_factor_dependency = cls.create_ancillary_dependency("uv-anisotropy-factor")
        waw_helioion_mp_dependency = cls.create_ancillary_dependency("waw-helioion-mp")

        uv_anisotropy_path = download_dependency(uv_anisotropy_factor_dependency)
        waw_helioion_mp_path = download_dependency(waw_helioion_mp_dependency)

        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL, 'f107_fluxtable.txt')
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL, 'lyman_alpha_composite.nc')
        omni2_data_path = download_external_dependency(OMNI2_URL, 'omni2_all_years.dat')

        return cls(uv_anisotropy_path, waw_helioion_mp_path, f107_index_file_path, lyman_alpha_path, omni2_data_path)

    @classmethod
    def create_ancillary_dependency(cls, descriptor: str):
        return UpstreamDataDependency(
            descriptor=descriptor,
            instrument="glows",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="latest",
        )
