'''
Author: Izabela Kowalska-Leszczynska, ikowalska@cbk.waw.pl
Main procedure to generate GLOWS L3d data product
'''

import glob
import sys
import warnings

import numpy as np

import toolkit.funcs as fun
import toolkit.l3d_SolarParamHistory as sph
from toolkit.constants import ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES

warnings.filterwarnings("ignore")

# current CR to process (should it be determine by the available L3bc? Or other way?)
CR_current = int(sys.argv[1])

# Create list of L3b/c files and read them
l3b_fn_list = np.array(sorted(glob.glob('data_l3b/imap_glows_l3b*.json')))
l3c_fn_list = np.array(sorted(glob.glob('data_l3c/imap_glows_l3c*.json')))
# l3d_fn_list = np.array(sorted(glob.glob('data_l3d/imap_glows_l3d*.json')))
data_l3b = [fun.read_json(fn) for fn in l3b_fn_list]
data_l3c = [fun.read_json(fn) for fn in l3c_fn_list]

solar_param_hist = sph.SolarParamsHistory(ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES)
fn_initial = ANC_INPUT_FROM_INSTRUMENT_TEAM['WawHelioIon']

# Start generating from the initial files (not using the last entry from L3d)

solar_param_hist.header['ancillary_data_files'] = list(fn_initial.values())
solar_param_hist.generate_initial_history(fn_initial)

solar_param_hist.update_solar_params_hist(EXT_DEPENDENCIES, data_l3b, data_l3c, CR_current)

# filename of the data products (main json file and the text version)
start_date = solar_param_hist.time_grid[0]
start_date_str = ''.join((start_date.iso[:10]).split('-'))
l3d_fn = 'data_l3d/imap_glows_l3d_solar-params-history_' + start_date_str + '-cr0' + str(CR_current) + '_v00.json'
solar_param_hist.header['filename'] = l3d_fn
l3d_txt_fn = {}
for k in solar_param_hist.ini_data['label']: l3d_txt_fn[
    k] = 'data_l3d_txt/imap_glows_l3d_' + k + '_' + start_date_str + '-cr0' + str(CR_current) + '_v00.dat'

hdr = solar_param_hist.generate_hdr_txt(l3d_txt_fn)

solar_param_hist.save_to_file(l3d_fn)
solar_param_hist.save_to_txt(l3d_txt_fn, hdr)

print('Processing CR=', CR_current)

