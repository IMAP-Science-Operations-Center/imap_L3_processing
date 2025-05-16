"""
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Constants used in the pipeline
"""

import astropy.units as u

VERSION = "3.0"  # version of ground processing software

# dictionary with names of ancillary input files expected from IMAP/GLOWS Instrument Team
ANC_INPUT_FROM_INSTRUMENT_TEAM = {
    'WawHelioIonMP_parameters': 'data_ancillary/imap_glows_WawHelioIonMP_v002.json',
    'bad_day_list': 'data_ancillary/imap_glows_bad-days-list_v001.dat',
    'pipeline_settings': 'data_ancillary/imap_glows_pipeline-settings-L3bcd_20250514_v004.json',
    'uv_anisotropy': 'data_ancillary/imap_glows_uv-anisotropy-1CR_v002.json',
    'WawHelioIon': {
        'speed': 'data_ancillary/imap_glows_plasma-speed-2010a_v003.dat',
        'p-dens': 'data_ancillary/imap_glows_proton-density-2010a_v003.dat',
        'uv-anis': 'data_ancillary/imap_glows_uv-anisotropy-2010a_v003.dat',
        'phion': 'data_ancillary/imap_glows_photoion-2010a_v003.dat',
        'lya': 'data_ancillary/imap_glows_lya-2010a_v003.dat',
        'e-dens': 'data_ancillary/imap_glows_electron-density-2010a_v003.dat'
    }}

EXT_DEPENDENCIES = {
    'f107_raw_data': 'external_dependencies/f107_fluxtable.txt',
    'omni_raw_data': 'external_dependencies/omni2_all_years.dat',
    'lya_raw_data': 'external_dependencies/lyman_alpha_composite.nc'
}

PHISICAL_CONSTANTS = {
    'm_alpha': 6.6446573357e-27 * u.kg,
    'jd_carrington_first': 2.398167329 * 10 ** 6,
    'carrington_time': 27.2753
}
