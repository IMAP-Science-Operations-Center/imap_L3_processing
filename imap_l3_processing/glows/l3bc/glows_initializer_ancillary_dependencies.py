import json
from dataclasses import dataclass
from pathlib import Path

from astropy.time import TimeDelta
from imap_data_access import query

from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_external_dependency, download_dependency

F107_FLUX_TABLE_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
LYMAN_ALPHA_COMPOSITE_INDEX_URL = "http://lasp.colorado.edu/data/timed_see/composite_lya/lyman_alpha_composite.nc"
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"


@dataclass
class GlowsInitializerAncillaryDependencies:
    uv_anisotropy_path: str
    waw_helioion_mp_path: str
    pipeline_settings: str
    bad_days_list: str
    initializer_time_buffer: TimeDelta
    f107_index_file_path: Path
    lyman_alpha_path: Path
    omni2_data_path: Path

    @classmethod
    def fetch_dependencies(cls):
        uv_anisotropy_factor_dependency = query(instrument="glows", descriptor="uv-anisotropy-1CR",
                                                version="latest")
        waw_helioion_mp_dependency = query(instrument="glows", descriptor="WawHelioIonMP",
                                           version="latest")
        bad_day_dependency = query(instrument="glows", descriptor="bad-day-list",
                                   version="latest")
        pipeline_settings_dependency = query(instrument="glows", descriptor="pipeline-settings",
                                             version="latest")

        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL, 'f107_fluxtable.txt')
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL, 'lyman_alpha_composite.nc')
        omni2_data_path = download_external_dependency(OMNI2_URL, 'omni2_all_years.dat')

        pipeline_settings_path = download_dependency(
            UpstreamDataDependency(instrument='glows', data_level='l3b', start_date=None, end_date=None,
                                   version='latest', descriptor='pipeline-settings-L3bc'))

        with open(pipeline_settings_path) as f:
            settings = json.load(f)
            initializer_time_buffer = TimeDelta(settings["initializer_time_buffer_days"], format="jd")

        return cls(uv_anisotropy_factor_dependency[0]['file_path'], waw_helioion_mp_dependency[0]['file_path'],
                   pipeline_settings_dependency[0]['file_path'],
                   bad_day_dependency[0]['file_path'],
                   initializer_time_buffer,
                   f107_index_file_path,
                   lyman_alpha_path, omni2_data_path,
                   )
