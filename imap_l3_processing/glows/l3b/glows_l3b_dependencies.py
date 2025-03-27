from dataclasses import dataclass
from pathlib import Path

from spacepy.pycdf import CDF

from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR
from imap_l3_processing.glows.l3a.models import GlowsL3LightCurve
from imap_l3_processing.glows.l3b.utils import read_glows_l3a_data
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.swapi.descriptors import SWAPI_L3A_ALPHA_SW_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL3AlphaSolarWindData
from imap_l3_processing.swapi.l3a.utils import read_l3a_alpha_sw_swapi_data
from imap_l3_processing.utils import download_dependency, download_external_dependency

F107_FLUX_TABLE_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
LYMAN_ALPHA_COMPOSITE_INDEX_URL = "http://lasp.colorado.edu/data/timed_see/composite_lya/lyman_alpha_composite.nc"
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"
UV_ANISOTROPY_KEY = "uv_anisotropy_factor"
WAW_HELIOION_MP_KEY = "waw_helioion_mp"
F107_INDEX_KEY = "f107_index"
LYMAN_ALPHA_COMPOSITE_KEY = "lyman_alpha_composite_index"
OMNI2_DATA_KEY = "omni2_data"


@dataclass
class GlowsInitializerDependencies:
    glows_l3a_data: GlowsL3LightCurve
    swapi_l3a_alpha_sw_data: SwapiL3AlphaSolarWindData
    ancillary_files: dict[str, Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        glows_l3a_dependency = next(dep
                                    for dep in dependencies if dep.descriptor.startswith(GLOWS_L3A_DESCRIPTOR))
        swapi_l3a_dependency = next(dep
                                    for dep in dependencies if dep.descriptor.startswith(SWAPI_L3A_ALPHA_SW_DESCRIPTOR))

        uv_anisotropy_factor_dependency = cls.create_ancillary_dependency("uv-anisotropy-factor")
        waw_helioion_mp_dependency = cls.create_ancillary_dependency("waw-helioion-mp")

        glows_l3a_path = download_dependency(glows_l3a_dependency)
        glows_l3a_cdf = CDF(str(glows_l3a_path))
        glows_l3a_data = read_glows_l3a_data(glows_l3a_cdf)

        swapi_l3a_path = download_dependency(swapi_l3a_dependency)
        swapi_l3a_cdf = CDF(str(swapi_l3a_path))
        swapi_l3a_alpha_sw_data = read_l3a_alpha_sw_swapi_data(swapi_l3a_cdf)

        uv_anisotropy_path = download_dependency(uv_anisotropy_factor_dependency)
        waw_helioion_mp_path = download_dependency(waw_helioion_mp_dependency)

        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL, 'f107_fluxtable.txt')
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL, 'lyman_alpha_composite.nc')
        omni2_data_path = download_external_dependency(OMNI2_URL, 'omni2_all_years.dat')

        ancillary_files = {
            UV_ANISOTROPY_KEY: uv_anisotropy_path,
            WAW_HELIOION_MP_KEY: waw_helioion_mp_path,
            F107_INDEX_KEY: f107_index_file_path,
            LYMAN_ALPHA_COMPOSITE_KEY: lyman_alpha_path,
            OMNI2_DATA_KEY: omni2_data_path
        }

        return cls(glows_l3a_data, swapi_l3a_alpha_sw_data, ancillary_files)

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
