import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from astropy.time import TimeDelta
from imap_data_access import ProcessingInputCollection, RepointInput, download

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import _comment_headers, \
    F107_FLUX_TABLE_URL, LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from imap_l3_processing.glows.l3bc.models import CRToProcess
from imap_l3_processing.utils import download_external_dependency, download_dependency_from_path


@dataclass
class GlowsL3BCAncillary:
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path | dict[str, Path]]
    repointing_file: Path
    lyman_alpha_path: Path


@dataclass
class GlowsL3DAncillary:
    external_files: dict[str, Path]
    ancillary_files: dict[str, Path | dict[str, Path]]

@dataclass
class GlowsL3EDependencies:
    energy_grid_lo: Path | None
    energy_grid_hi: Path | None
    energy_grid_ultra: Path | None
    tess_xyz_8: Path | None
    tess_ang16: Path | None
    lya_series: Path
    solar_uv_anisotropy: Path
    speed_3d_sw: Path
    density_3d_sw: Path
    phion_hydrogen: Path
    sw_eqtr_electrons: Path
    ionization_files: Path
    pipeline_settings: dict
    elongation: dict
    repointing_file: Path

@dataclass
class GlowsL3BCDEDependencies:
    glows_l3b_ancillary: GlowsL3BCAncillary
    glows_l3d_ancillary: GlowsL3DAncillary


    # external_files: dict[str, Path]
    # ancillary_files: dict[str, Path | dict[str, Path]]
    # l3b_file_paths: list[Path]
    # l3c_file_paths: list[Path]
    #
    # energy_grid_lo: Path | None
    # energy_grid_hi: Path | None
    # energy_grid_ultra: Path | None
    # tess_xyz_8: Path | None
    # tess_ang16: Path | None
    # lya_series: Path
    # solar_uv_anisotropy: Path
    # speed_3d_sw: Path
    # density_3d_sw: Path
    # phion_hydrogen: Path
    # sw_eqtr_electrons: Path
    # ionization_files: Path
    # pipeline_settings: dict
    # elongation: dict
    # repointing_file: Path

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        uv_anisotropy_factor_dependency = dependencies.get_file_paths(source='glows', descriptor='uv-anisotropy-1CR')
        waw_helioion_mp_dependency = dependencies.get_file_paths(source='glows', descriptor='WawHelioIonMP')
        bad_day_dependency = dependencies.get_file_paths(source='glows', descriptor='bad-days-list')
        pipeline_settings_dependency = dependencies.get_file_paths(source='glows',
                                                                   descriptor='pipeline-settings-l3bcde')
        repointing_dependency = dependencies.get_file_paths(data_type=RepointInput.data_type)
        repointing_path = download(repointing_dependency[0].name)

        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL,
                                                            TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')
        _comment_headers(f107_index_file_path, num_lines=2)
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL,
                                                        TEMP_CDF_FOLDER_PATH / 'lyman_alpha_composite.nc')
        omni2_data_path = download_external_dependency(OMNI2_URL, TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat')

        external_files = {}
        external_files['f107_raw_data'] = f107_index_file_path
        external_files['omni_raw_data'] = omni2_data_path

        uv_anisotropy_factor_path = str(uv_anisotropy_factor_dependency[0])
        waw_helioion_mp_path = str(waw_helioion_mp_dependency[0])
        pipeline_settings_path = str(pipeline_settings_dependency[0])
        bad_day_path = str(bad_day_dependency[0])

        ancillary_files = {
            'uv_anisotropy': download_dependency_from_path(uv_anisotropy_factor_path),
            'WawHelioIonMP_parameters': download_dependency_from_path(waw_helioion_mp_path),
            'bad_days_list': download_dependency_from_path(bad_day_path),
            'pipeline_settings': download_dependency_from_path(pipeline_settings_path),
        }

        plasma_speed_legendre_path = dependencies.get_file_paths(source='glows',
                                                                 descriptor='plasma-speed-2010a')
        proton_density_legendre_path = dependencies.get_file_paths(source='glows',
                                                                   descriptor='proton-density-2010a')
        uv_anisotropy_path = dependencies.get_file_paths(source='glows', descriptor='uv-anisotropy-2010a')
        photoion_path = dependencies.get_file_paths(source='glows', descriptor='photoion-2010a')
        lya_path = dependencies.get_file_paths(source='glows', descriptor='lya-2010a')
        electron_density_path = dependencies.get_file_paths(source='glows', descriptor='electron-density-2010a')

        plasma_speed_legendre = download(str(plasma_speed_legendre_path[0]))
        proton_density_legendre = download(str(proton_density_legendre_path[0]))
        uv_anisotropy = download(str(uv_anisotropy_path[0]))
        photoion = download(str(photoion_path[0]))
        lya_2010a = download(str(lya_path[0]))
        electron_density = download(str(electron_density_path[0]))

        l3d_ancillary_dict = {
            'pipeline_settings': pipeline_settings_path,
            'WawHelioIon': {
                'speed': plasma_speed_legendre,
                'p-dens': proton_density_legendre,
                'uv-anis': uv_anisotropy,
                'phion': photoion,
                'lya': lya_2010a,
                'e-dens': electron_density
            }}

        external_dict = {
            'lya_raw_data': lyman_alpha_path
        }

        bc_ancillary = GlowsL3BCAncillary(
            external_files=external_files,
            ancillary_files=ancillary_files,
            repointing_file=repointing_path,
            lyman_alpha_path=lyman_alpha_path
        )

        d_ancillary = GlowsL3DAncillary(external_dict, l3d_ancillary_dict)

        return cls(bc_ancillary, d_ancillary)
