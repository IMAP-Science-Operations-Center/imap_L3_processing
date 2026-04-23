from dataclasses import dataclass
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.glows.descriptors import PROTON_DENSITY_DESCRIPTOR, PLASMA_SPEED_DESCRIPTOR, \
    UV_ANISOTROPY_DESCRIPTOR, PHOTOION_DESCRIPTOR, LYA_DESCRIPTOR, ELECTRON_DENSITY_DESCRIPTOR, \
    PIPELINE_SETTINGS_L3BCDE_DESCRIPTOR, GLOWS_L3B_DESCRIPTOR, GLOWS_L3C_DESCRIPTOR
from imap_l3_processing.glows.l3bc.models import ExternalDependencies


@dataclass
class GlowsL3DDependencies:
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path | dict[str, Path]]
    l3b_file_paths: list[Path]
    l3c_file_paths: list[Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection, external_dependencies: ExternalDependencies):
        plasma_speed_legendre_path = dependencies.get_file_paths(source='glows', descriptor=PLASMA_SPEED_DESCRIPTOR)
        proton_density_legendre_path = dependencies.get_file_paths(source='glows', descriptor=PROTON_DENSITY_DESCRIPTOR)
        uv_anisotropy_path = dependencies.get_file_paths(source='glows', descriptor=UV_ANISOTROPY_DESCRIPTOR)
        photoion_path = dependencies.get_file_paths(source='glows', descriptor=PHOTOION_DESCRIPTOR)
        lya_path = dependencies.get_file_paths(source='glows', descriptor=LYA_DESCRIPTOR)
        electron_density_path = dependencies.get_file_paths(source='glows', descriptor=ELECTRON_DENSITY_DESCRIPTOR)
        pipeline_settings_l3bc_path = dependencies.get_file_paths(source='glows', descriptor=PIPELINE_SETTINGS_L3BCDE_DESCRIPTOR)

        plasma_speed_legendre = imap_data_access.download(str(plasma_speed_legendre_path[0]))
        proton_density_legendre = imap_data_access.download(str(proton_density_legendre_path[0]))
        uv_anisotropy = imap_data_access.download(str(uv_anisotropy_path[0]))
        photoion = imap_data_access.download(str(photoion_path[0]))
        lya_2010a = imap_data_access.download(str(lya_path[0]))
        electron_density = imap_data_access.download(str(electron_density_path[0]))
        pipeline_settings_l3bc = imap_data_access.download(str(pipeline_settings_l3bc_path[0]))

        l3b_file_names = dependencies.get_file_paths(source="glows", descriptor=GLOWS_L3B_DESCRIPTOR)
        l3c_file_names = dependencies.get_file_paths(source="glows", descriptor=GLOWS_L3C_DESCRIPTOR)

        l3b_file_paths = [imap_data_access.download(l3b) for l3b in l3b_file_names]
        l3c_file_paths = [imap_data_access.download(l3c) for l3c in l3c_file_names]

        ancillary_dict = {
            'pipeline_settings': pipeline_settings_l3bc,
            'WawHelioIon': {
                'speed': plasma_speed_legendre,
                'p-dens': proton_density_legendre,
                'uv-anis': uv_anisotropy,
                'phion': photoion,
                'lya': lya_2010a,
                'e-dens': electron_density
            }}

        external_dict = {
            'lya_raw_data': external_dependencies.lyman_alpha_path
        }

        return cls(external_dict, ancillary_dict, l3b_file_paths, l3c_file_paths)
