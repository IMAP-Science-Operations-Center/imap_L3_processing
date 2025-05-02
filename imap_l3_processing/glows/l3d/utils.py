import json
import os
from pathlib import Path
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
        lat_grid=l3d_json_dict['lat_grid'],
        cr_grid=l3d_json_dict['cr_grid'],
        time_grid=l3d_json_dict['time_grid'],
        speed=l3d_json_dict['solar_params']['speed'],
        p_dens=l3d_json_dict['solar_params']['p-dens'],
        uv_anis=l3d_json_dict['solar_params']['uv-anis'],
        phion=l3d_json_dict['solar_params']['phion'],
        lya=l3d_json_dict['solar_params']['lya'],
        e_dens=l3d_json_dict['solar_params']['e-dens'],
    )
