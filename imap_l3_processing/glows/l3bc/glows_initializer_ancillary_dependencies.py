import json
from dataclasses import dataclass
from pathlib import Path

from astropy.time import TimeDelta
from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.utils import download_external_dependency

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
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        uv_anisotropy_factor_dependency = dependencies.get_file_paths(source='glows', descriptor='uv-anisotropy-1CR')
        waw_helioion_mp_dependency = dependencies.get_file_paths(source='glows', descriptor='WawHelioIonMP')
        bad_day_dependency = dependencies.get_file_paths(source='glows', descriptor='bad-days-list')
        pipeline_settings_dependency = dependencies.get_file_paths(source='glows',
                                                                   descriptor='pipeline-settings-l3bcde')

        pipeline_settings_path = download(pipeline_settings_dependency[0])

        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL,
                                                            TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')
        _comment_headers(f107_index_file_path, num_lines=2)
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL,
                                                        TEMP_CDF_FOLDER_PATH / 'lyman_alpha_composite.nc')
        omni2_data_path = download_external_dependency(OMNI2_URL, TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat')

        with open(pipeline_settings_path) as f:
            settings = json.load(f)
            initializer_time_buffer = TimeDelta(settings["initializer_time_delta_days"], format="jd")

        return cls(str(uv_anisotropy_factor_dependency[0]),
                   str(waw_helioion_mp_dependency[0]),
                   str(pipeline_settings_dependency[0]),
                   str(bad_day_dependency[0]),
                   initializer_time_buffer,
                   f107_index_file_path,
                   lyman_alpha_path, omni2_data_path,
                   )


def _comment_headers(filename: Path, num_lines=2):
    with open(filename, "r+") as file:
        lines = file.readlines()
        for i in range(num_lines):
            lines[i] = "#" + lines[i]
        file.truncate(0)
    with open(filename, "w") as file:
        file.writelines(lines)
