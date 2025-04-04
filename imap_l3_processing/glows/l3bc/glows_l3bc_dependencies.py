import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

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

    @classmethod
    def fetch_dependencies(cls, zip_file_path):
        external_files = {}
        with ZipFile(zip_file_path, 'r') as zip_file:
            downloaded_file_dir = Path(__file__).parent / 'glows_l3b_files'
            zip_file.extract(str(downloaded_file_dir))
            external_files['f107_raw_data'] = downloaded_file_dir / 'f107_fluxtable.txt'
            external_files['omni_raw_data'] = downloaded_file_dir / 'omni2_all_years.dat'
            json_with_paths_to_download = downloaded_file_dir / 'cr_to_process.json'

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
                   start_date=datetime.fromisoformat(paths_to_download['start_date']),
                   end_date=datetime.fromisoformat(paths_to_download['end_date']))
