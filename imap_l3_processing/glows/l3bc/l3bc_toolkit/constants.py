"""
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Constants used in the pipeline
"""

import astropy.units as u

VERSION = "1.0" # version of ground processing software

# dictionary with names of ancillary input files expected from IMAP/GLOWS Instrument Team
ANC_INPUT_FROM_INSTRUMENT_TEAM = {
    'WawHelioIonMP_parameters': 'data_ancillary/imap_glows_WawHelioIonMP_v002.json',
    'bad_day_list': 'data_ancillary/imap_glows_bad-days-list_v001.dat',
    'pipeline_settings': 'data_ancillary/imap_glows_pipeline-settings-L3bc_v001.json',
    'uv_anisotropy' : 'data_ancillary/imap_glows_uv-anisotropy-1CR_v001.json'
}

EXT_DEPENDENCIES={
    'f107_raw_data': 'external_dependencies/f107_fluxtable.txt',
    'omni_raw_data': 'external_dependencies/omni_2010.dat'
}

PHISICAL_CONSTANTS={
    'm_alpha' : 6.6446573357e-27*u.kg,
    'jd_carrington_first' : 2.398167329*10**6,
    'carrington_time' : 27.2753
}
