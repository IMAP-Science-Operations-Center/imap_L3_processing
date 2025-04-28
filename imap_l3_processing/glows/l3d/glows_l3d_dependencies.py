from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3d.utils import create_glows_l3c_dictionary_from_cdf
from imap_l3_processing.utils import download_dependency_from_path


@dataclass
class GlowsL3DDependencies:
    l3c_data: [dict]
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        plasma_speed_legendre_path = dependencies.get_file_paths(source='glows', descriptor='plasma-speed-legendre')
        proton_density_legendre_path = dependencies.get_file_paths(source='glows', descriptor='proton-density-legendre')
        uv_anisotropy_path = dependencies.get_file_paths(source='glows', descriptor='uv-anisotropy')
        photoion_path = dependencies.get_file_paths(source='glows', descriptor='photoion')
        lya_path = dependencies.get_file_paths(source='glows', descriptor='lya')
        electron_density_path = dependencies.get_file_paths(source='glows', descriptor='electron-density')
        pipeline_settings_l3bc_path = dependencies.get_file_paths(source='glows', descriptor='pipeline-settings-l3bc')
        external_archive = dependencies.get_file_paths(source='glows', descriptor='l3b-archive')
        l3c_solar_params_path = dependencies.get_file_paths(source='glows', descriptor='solar-params')

        plasma_speed_legendre = download_dependency_from_path(str(plasma_speed_legendre_path[0]))
        proton_density_legendre = download_dependency_from_path(str(proton_density_legendre_path[0]))
        uv_anisotropy = download_dependency_from_path(str(uv_anisotropy_path[0]))
        photoion = download_dependency_from_path(str(photoion_path[0]))
        lya_2010a = download_dependency_from_path(str(lya_path[0]))
        electron_density = download_dependency_from_path(str(electron_density_path[0]))
        pipeline_settings_l3bc = download_dependency_from_path(str(pipeline_settings_l3bc_path[0]))
        archive_dependency = download_dependency_from_path(str(external_archive[0]))
        l3c_cdf = download_dependency_from_path(str(l3c_solar_params_path[0]))

        l3_c_dict = create_glows_l3c_dictionary_from_cdf(l3c_cdf)

        with ZipFile(archive_dependency, 'r') as archive:
            archive.extract('lyman_alpha_composite.nc', TEMP_CDF_FOLDER_PATH)

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
            'lya_raw_data': TEMP_CDF_FOLDER_PATH / 'lyman_alpha_composite.nc'
        }

        return cls(l3_c_dict, external_dict, ancillary_dict)
