import json
from datetime import datetime, timedelta
from pathlib import Path
from spacepy.pycdf import CDF

import imap_l3_processing

PATH_TO_L3D_TOOLKIT = Path(imap_l3_processing.__file__) / 'glows' / 'l3d' / 'toolkit'


def create_glows_l3c_json_file_from_cdf(cdf_file_path: Path):
    with CDF(str(cdf_file_path)) as cdf:
        json_dict = {
            'header': {
                'filename': cdf_file_path.name
            },
            'solar_wind_profile': {
                'proton_density': cdf["proton_density_profile"][0],
                'plasma_speed': cdf["plasma_speed_profile"][0],
            },
            'solar_wind_ecliptic': {
                'proton_density': cdf["proton_density_ecliptic"][0],
                'alpha_abundance': cdf["alpha_abundance_ecliptic"][0],
            }
        }
        cr_number = cdf['cr'][...][0]
        version = cdf_file_path.name.split('_')[-1].split('.')[0]
        json_file_name = f'imap_glows_l3c_cr_{cr_number}_{version}.json'

        with open(PATH_TO_L3D_TOOLKIT / 'data_l3c' / json_file_name) as fp:
            json.dump(json_dict, fp)


def create_glows_l3b_dictionary_from_cdf(cdf_file_path: Path):
    with CDF(str(cdf_file_path)) as cdf:
        json_dict = {
            'header': {
                'filename': cdf_file_path.name,
                'l3a_input_files_name': [file for file in cdf.attrs['Parents'] if 'l3a' in file]
            },
            'uv_anisotropy_factor': cdf['uv_anisotropy_factor'][0],
            'ion_rate_profile': {
                'lat_grid': cdf['lat_grid'][...],
                'ph_rate': cdf['ph_rate'][0]
            }
        }

        cr_number = cdf['cr'][...][0]
        version = cdf_file_path.name.split('_')[-1].split('.')[0]
        json_file_name = f'imap_glows_l3b_cr_{cr_number}_{version}.json'

        with open(PATH_TO_L3D_TOOLKIT / 'data_l3b' / json_file_name) as fp:
            json.dump(json_dict, fp)


def convert_json_l3d_to_cdf(json_file_path: Path, path_to_write_cdf_to: Path) -> Path:
    with open(json_file_path, 'r') as json_file:
        l3d_json_dict = json.load(json_file)

    start_date = (datetime.fromisoformat(l3d_json_dict['time_grid'][-1]) - (timedelta(days=27.25) / 2)).strftime(
        '%Y%m%d')

    with CDF(str(path_to_write_cdf_to / f'imap_glows_l3d_solar-params-history_{start_date}_v000.cdf'),
             create=True) as cdf:
        cdf['lat_grid'] = l3d_json_dict['lat_grid']
        cdf['cr_grid'] = l3d_json_dict['cr_grid']
        cdf['time_grid'] = l3d_json_dict['time_grid']
        cdf['speed'] = l3d_json_dict['solar_params']['speed']
        cdf['p_dens'] = l3d_json_dict['solar_params']['p-dens']
        cdf['uv_anis'] = l3d_json_dict['solar_params']['uv-anis']
        cdf['phion'] = l3d_json_dict['solar_params']['phion']
        cdf['lya'] = l3d_json_dict['solar_params']['lya']
        cdf['e_dens'] = l3d_json_dict['solar_params']['e-dens']

    return path_to_write_cdf_to / f'imap_glows_l3d_solar-params-history_{start_date}_v000.cdf'
