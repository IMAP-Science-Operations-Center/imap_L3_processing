import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency, download_dependency_from_path


@dataclass
class GlowsL3BCDependencies:
    l3a_data: [dict]
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        external_files = {}
        zip_file_path = download_dependency(dependencies[0])
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
            'waw_helioion_mp': download_dependency_from_path(paths_to_download['waw_helioion_mp'])
        }

        return cls(l3a_data=[], external_files=external_files, ancillary_files=ancillary_files)
