from dataclasses import dataclass
from pathlib import Path

import imap_data_access

from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.glows.l3bc.models import CRToProcess, ExternalDependencies


@dataclass
class GlowsL3BCDependencies:
    carrington_rotation_number: int
    l3a_data: list[dict]
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path]

    @classmethod
    def from_cr_to_process(cls, cr_to_process: CRToProcess, external_dependencies: ExternalDependencies):
        external_files = {
            'f107_raw_data': external_dependencies.f107_index_file_path,
            'omni_raw_data': external_dependencies.omni2_data_path
        }

        ancillary_files = {
            'uv_anisotropy': imap_data_access.download(cr_to_process.uv_anisotropy_file_name),
            'WawHelioIonMP_parameters': imap_data_access.download(cr_to_process.waw_helio_ion_mp_file_name),
            'bad_days_list': imap_data_access.download(cr_to_process.bad_days_list_file_name),
            'pipeline_settings': imap_data_access.download(cr_to_process.pipeline_settings_file_name),
        }

        l3a_data = []
        for l3a_file in sorted(list(cr_to_process.l3a_file_names)):
            downloaded_file_path = imap_data_access.download(l3a_file)
            l3a_data.append(create_glows_l3a_dictionary_from_cdf(downloaded_file_path))

        return cls(
            carrington_rotation_number=cr_to_process.cr_rotation_number,
            l3a_data=l3a_data,
            external_files=external_files,
            ancillary_files=ancillary_files,
        )
