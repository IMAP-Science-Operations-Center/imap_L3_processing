from pathlib import Path
from spacepy.pycdf import CDF


def create_glows_l3c_dictionary_from_cdf(cdf_file_path: Path) -> dict:
    with CDF(str(cdf_file_path)) as cdf:
        return {
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


def create_glows_l3b_dictionary_from_cdf(cdf_file_path: Path) -> dict:
    with CDF(str(cdf_file_path)) as cdf:
        return {
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
