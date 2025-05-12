import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

import imap_l3_processing
from imap_l3_processing.glows.l3d.models import GlowsL3DSolarParamsHistory
from imap_l3_processing.models import InputMetadata

PATH_TO_L3D_TOOLKIT = Path(imap_l3_processing.__file__).parent / 'glows' / 'l3d' / 'science'


def create_glows_l3c_json_file_from_cdf(cdf_file_path: Path):
    with CDF(str(cdf_file_path)) as cdf:
        json_dict = {
            'header': {
                'filename': cdf_file_path.name
            },
            'solar_wind_profile': {
                'proton_density': cdf["proton_density_profile"][0].tolist(),
                'plasma_speed': cdf["plasma_speed_profile"][0].tolist(),
            },
            'solar_wind_ecliptic': {
                'proton_density': float(cdf["proton_density_ecliptic"][0]),
                'alpha_abundance': float(cdf["alpha_abundance_ecliptic"][0]),
            }
        }
        cr_number = cdf['cr'][...][0]
        version = cdf_file_path.name.split('_')[-1].split('.')[0]
        json_file_name = f'imap_glows_l3c_cr_{cr_number}_{version}.json'

        os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3c', exist_ok=True)

        with open(PATH_TO_L3D_TOOLKIT / 'data_l3c' / json_file_name, 'w') as fp:
            json.dump(json_dict, fp)


def create_glows_l3b_json_file_from_cdf(cdf_file_path: Path):
    with CDF(str(cdf_file_path)) as cdf:
        cr_number = int(cdf['cr'][...][0])
        json_dict = {
            'header': {
                'filename': cdf_file_path.name,
                'l3a_input_files_name': [file for file in cdf.attrs['Parents'] if 'l3a' in file]
            },
            'CR': cr_number,
            'uv_anisotropy_factor': cdf['uv_anisotropy_factor'][0].tolist(),
            'ion_rate_profile': {
                'lat_grid': cdf['lat_grid'][...].tolist(),
                'ph_rate': cdf['ph_rate'][0].tolist()
            }
        }

        version = cdf_file_path.name.split('_')[-1].split('.')[0]
        json_file_name = f'imap_glows_l3b_cr_{cr_number}_{version}.json'

        os.makedirs(PATH_TO_L3D_TOOLKIT / 'data_l3b', exist_ok=True)
        with open(PATH_TO_L3D_TOOLKIT / 'data_l3b' / json_file_name, 'w') as fp:
            json.dump(json_dict, fp)


def convert_json_to_l3d_data_product(json_file_path: Path, input_metadata: InputMetadata,
                                     parent_file_names: list[str]) -> GlowsL3DSolarParamsHistory:
    with open(json_file_path, 'r') as json_file:
        l3d_json_dict = json.load(json_file)

    return GlowsL3DSolarParamsHistory(
        input_metadata=input_metadata,
        parent_file_names=parent_file_names,
        latitude=l3d_json_dict['lat_grid'],
        cr=l3d_json_dict['cr_grid'],
        epoch=np.array([datetime.fromisoformat(time) for time in l3d_json_dict['time_grid']]),
        speed=l3d_json_dict['solar_params']['speed'],
        proton_density=l3d_json_dict['solar_params']['p-dens'],
        ultraviolet_anisotropy=l3d_json_dict['solar_params']['uv-anis'],
        phion=l3d_json_dict['solar_params']['phion'],
        lyman_alpha=l3d_json_dict['solar_params']['lya'],
        electron_density=l3d_json_dict['solar_params']['e-dens'],
    )


def get_parent_file_names_from_l3d_json(l3d_folder: Path) -> list[str]:
    parent_file_names = set()
    l3d_file_paths = sorted(os.listdir(l3d_folder))

    if len(l3d_file_paths) == 0:
        return []

    with open(l3d_folder / l3d_file_paths[0]) as l3d:
        l3d_data = json.load(l3d)
        ancillary_files = [Path(file).name for file in l3d_data['header']['ancillary_data_files']]
        external_dependencies = Path(l3d_data['header']['external_dependeciens']).name
        parent_file_names.update(ancillary_files)
        parent_file_names.add(external_dependencies)

    for file_path in l3d_file_paths:
        with open(l3d_folder / file_path, 'r') as l3d:
            l3d_data = json.load(l3d)
            l3b_parents = [Path(file).name for file in l3d_data['header']['l3b_input_filename']]
            l3c_parents = [Path(file).name for file in l3d_data['header']['l3c_input_filename']]
            parent_file_names.update(l3b_parents)
            parent_file_names.update(l3c_parents)

    return list(parent_file_names)
