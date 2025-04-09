import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.utils import download_dependency_from_path


@dataclass
class GlowsL3BCDependencies:
    l3a_data: [dict]
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path]
    carrington_rotation_number: int
    start_date: datetime
    end_date: datetime
    zip_file_path: Path

    @classmethod
    def fetch_dependencies(cls, zip_file_path):
        external_files = {}
        with ZipFile(zip_file_path, 'r') as zip_file:
            file_names = ['f107_fluxtable.txt', 'omni2_all_years.dat', 'cr_to_process.json']
            for filename in file_names:
                zip_file.extract(filename, TEMP_CDF_FOLDER_PATH)
            external_files['f107_raw_data'] = TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt'
            external_files['omni_raw_data'] = TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat'
            json_with_paths_to_download = TEMP_CDF_FOLDER_PATH / 'cr_to_process.json'

        with open(json_with_paths_to_download, 'r') as json_file:
            json_string = json_file.read()
            paths_to_download = json.loads(json_string)

        ancillary_files = {
            'uv_anisotropy': download_dependency_from_path(paths_to_download['uv_anisotropy']),
            'WawHelioIonMP_parameters': download_dependency_from_path(paths_to_download['waw_helioion_mp']),
            'bad_days_list': download_dependency_from_path(paths_to_download['bad_days_list']),
            'pipeline_settings': download_dependency_from_path(paths_to_download['pipeline_settings']),
        }

        l3a_paths = paths_to_download['l3a_paths']
        l3a_data = []
        for path in l3a_paths:
            l3a_data.append(create_glows_l3a_dictionary_from_cdf(download_dependency_from_path(path)))

        return cls(l3a_data=l3a_data, external_files=external_files, ancillary_files=ancillary_files,
                   carrington_rotation_number=int(paths_to_download['cr_rotation_number']),
                   start_date=datetime.fromisoformat(paths_to_download['cr_start_date']),
                   end_date=datetime.fromisoformat(paths_to_download['cr_end_date']), zip_file_path=zip_file_path)
