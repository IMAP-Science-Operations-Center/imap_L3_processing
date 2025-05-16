from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from imap_data_access import download, query
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH


@dataclass
class GlowsL3DDependencies:
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path | dict[str, Path]]
    l3b_file_paths: list[Path]
    l3c_file_paths: list[Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        plasma_speed_legendre_path = dependencies.get_file_paths(source='glows',
                                                                 descriptor='plasma-speed-2010a')
        proton_density_legendre_path = dependencies.get_file_paths(source='glows',
                                                                   descriptor='proton-density-2010a')
        uv_anisotropy_path = dependencies.get_file_paths(source='glows', descriptor='uv-anisotropy-2010a')
        photoion_path = dependencies.get_file_paths(source='glows', descriptor='photoion-2010a')
        lya_path = dependencies.get_file_paths(source='glows', descriptor='lya-2010a')
        electron_density_path = dependencies.get_file_paths(source='glows', descriptor='electron-density-2010a')
        pipeline_settings_l3bc_path = dependencies.get_file_paths(source='glows', descriptor='pipeline-settings-l3bcd')
        external_archive = dependencies.get_file_paths(source='glows', descriptor='l3b-archive')

        plasma_speed_legendre = download(str(plasma_speed_legendre_path[0]))
        proton_density_legendre = download(str(proton_density_legendre_path[0]))
        uv_anisotropy = download(str(uv_anisotropy_path[0]))
        photoion = download(str(photoion_path[0]))
        lya_2010a = download(str(lya_path[0]))
        electron_density = download(str(electron_density_path[0]))
        pipeline_settings_l3bc = download(str(pipeline_settings_l3bc_path[0]))

        archive_dependency = download(str(external_archive[0]))

        l3b_file_names = query(instrument='glows', descriptor='ion-rate-profile', version='latest')
        l3c_file_names = query(instrument='glows', descriptor='sw-profile', version='latest')

        l3b_file_paths = [download(l3b['file_path']) for l3b in l3b_file_names]
        l3c_file_paths = [download(l3c['file_path']) for l3c in l3c_file_names]

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

        return cls(external_dict, ancillary_dict, l3b_file_paths, l3c_file_paths)
