'''
Author: Izabela Kowalska-Leszczynska, ikowalska@cbk.waw.pl
Main procedure to generate GLOWS L3d data product
'''

import numpy as np
import glob
import toolkit.funcs as fun

import toolkit.l3d_SolarParamHistory as sph
from toolkit.constants import ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES


import warnings
warnings.filterwarnings("ignore")

# Create list of L3b/c files and read them
l3b_fn_list = np.array(sorted(glob.glob('data_l3b/imap_glows_l3b*.json')))
l3c_fn_list = np.array(sorted(glob.glob('data_l3c/imap_glows_l3c*.json')))
l3d_fn_list = np.array(sorted(glob.glob('data_l3d/imap_glows_l3d*.json')))
data_l3b=[fun.read_json(fn) for fn in l3b_fn_list]
data_l3c=[fun.read_json(fn) for fn in l3c_fn_list]

solar_param_hist=sph.SolarParamsHistory(ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES)


# if there is no previous L3d_dat files, read initial files prepared by the instrument team
if len(l3d_fn_list) == 0:
    # check ancillary files
    fn_initial=ANC_INPUT_FROM_INSTRUMENT_TEAM['WawHelioIon']
    ini_flag=True
else:
    # when there is at least one L3d file, we will use the lates one as a initial data and then update it using next CR's L3bc
    # in the final code there should be some kind of version checking, not just the lates CR, but also the highest version
    # list of all L3d text files
    fn_initial=solar_param_hist.find_fn_initial()
    ini_flag=False


solar_param_hist.header['ancillary_data_files']=list(fn_initial.values())
solar_param_hist.generate_initial_history(fn_initial)

# current CR for L3d data products file names
CR=int(np.floor(solar_param_hist.ini_data['CR_last']+1))

solar_param_hist.update_solar_params_hist(EXT_DEPENDENCIES,data_l3b,data_l3c)
    
# filename of the data products (main file and the text version)
l3d_fn = 'data_l3d/imap_glows_l3d_cr_'+str(CR)+'_v00.json'
solar_param_hist.header['filename']=l3d_fn
l3d_txt_fn={}
for k in solar_param_hist.ini_data['label']: l3d_txt_fn[k]='data_l3d_txt/imap_glows_l3d_cr_'+str(CR)+'_'+k+'_v00.dat'

if ini_flag: hdr=solar_param_hist.generate_hdr_txt_ini(l3d_txt_fn)
else: hdr=solar_param_hist.generate_hdr_txt(l3d_txt_fn)

solar_param_hist.save_to_file(l3d_fn)
solar_param_hist.save_to_txt(l3d_txt_fn,hdr)

print('Processed CR=', CR)

