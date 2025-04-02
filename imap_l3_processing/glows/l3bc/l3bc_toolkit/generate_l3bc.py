'''
Author: Izabela Kowalska-Leszczynska, ikowalska@cbk.waw.pl
Main procedure to generate GLOWS L3b and L3c data products
'''
import glob
import warnings

import numpy as np
from astropy.time import Time

import funcs as fun
import l3b_CarringtonIonRate as cir
import l3b_DailyIonRate as dir
import l3c_CarringtonSolarWind as csw
import l3c_EclipticSolarWind as esw
from constants import ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES

warnings.filterwarnings("ignore")

# Create a list of L3a files within the same Carrington number
l3a_fn_list = np.array(sorted(glob.glob('data_l3a/imap_glows_l3a*.json')))
l3a_date_list = Time([fun.time_string_from_l3a_fn(fn) for fn in l3a_fn_list])
l3a_CR_list = np.array([fun.carrington(t.jd) for t in l3a_date_list])

# Select files for each of CR number
cr, idx_CR_start = np.unique(np.array([int(i) for i in l3a_CR_list]), return_index=True)

# bad-day-list
bad_day_fn = ANC_INPUT_FROM_INSTRUMENT_TEAM['bad_day_list']
bad_days_date = Time(np.genfromtxt(bad_day_fn, dtype=str)[:, 1])

# Loop for all available CR for testing (in production code it will be executed for a current CR and that number will have to be passed to the code)
for cr_idx in range(len(cr) - 1):
    CR = cr[cr_idx]
    print('Processing CR=' + str(CR))
    # output files name
    l3b_fn = 'data_l3b/imap_glows_l3b_cr_' + str(CR) + '_v00.json'
    l3c_fn = 'data_l3c/imap_glows_l3c_cr_' + str(CR) + '_v00.json'

    # list of L3a files within 1 Carrington number
    l3a_1CR_fn_list = l3a_fn_list[idx_CR_start[cr_idx]:idx_CR_start[cr_idx + 1]]
    l3a_1CR_date_list = l3a_date_list[idx_CR_start[cr_idx]:idx_CR_start[cr_idx + 1]]

    # bad days removed
    idx_good_l3a = [np.where(l3a_date_list == d)[0][0] for d in l3a_1CR_date_list if d not in bad_days_date]

    # if all days in CR are bad, then there will be no L3bc data products
    if len(idx_good_l3a) == 0: continue

    # Daily ionization rate from the daily light curves
    daily_ion_rate = []
    l3a_used_fn = []

    for l3a_fn in l3a_fn_list[idx_good_l3a]:
        l3b_daily_fn = 'data_l3b_daily/imap_glows_l3b_daily_' + '_'.join(
            l3a_fn.split('_')[-5:])  # TEMP solution for testing
        ion_rate_daily = dir.DailyIonizationRate(
            ANC_INPUT_FROM_INSTRUMENT_TEAM)  # daily ionization rate object initialization
        ion_rate_daily.read_l3a(l3a_fn)  # readinig L3a data (masked lightcurve)
        flag = ion_rate_daily.read_model_params(
            ANC_INPUT_FROM_INSTRUMENT_TEAM)  # reading WawHelioIon-MP parameters for lightcurve to ionization rate conversion
        if flag == ion_rate_daily.settings['no_vg_point_flag']: continue
        l3a_used_fn.append(l3a_fn)
        ion_rate_daily.calculate_daily_ionization_rate()  # conversion to the daily ionization rate profile
        ion_rate_daily.save_to_file(l3b_daily_fn)  # TEMP solution for testing
        daily_ion_rate.append(ion_rate_daily.ion['ion_rate'])

    if len(daily_ion_rate) == 0: continue

    # Carrington averaged ionization rate (L3b) and Carrington averaged solar wind parameters  objects initialization
    ion_rate_Carr = cir.CarringtonIonizationRate(l3a_used_fn, ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES)
    solar_wind_Carr = csw.CarringtonSolarWind(ANC_INPUT_FROM_INSTRUMENT_TEAM)

    # File name of the L3b and L3c passed to the object structure #TODO: update documentation!
    ion_rate_Carr.header['filename'] = l3b_fn
    solar_wind_Carr.header['filename'] = l3c_fn

    # Save daily ionization rates into CarringtonIonizationRate structure
    ion_rate_Carr.daily_ion_rate['date'] = l3a_date_list[idx_good_l3a]
    ion_rate_Carr.daily_ion_rate['ion_rate'] = daily_ion_rate

    ion_rate_Carr.carr_ion_rate['date'] = Time(fun.jd_fm_Carrington(CR + 0.5), format='jd')
    ion_rate_Carr.carr_ion_rate['CR'] = CR  # Carrington rotation number

    # Solar wind in the ecliptic plane processing
    sw_ecliptic = esw.EclipticSolarWind(ANC_INPUT_FROM_INSTRUMENT_TEAM, CR)
    sw_ecliptic.calculate_invariant(EXT_DEPENDENCIES)
    solar_wind_Carr.sw_ecliptic['invariant'] = sw_ecliptic.invariant
    solar_wind_Carr.header['external_dependeciens'] = sw_ecliptic.external_dependeciens

    # Solar wind parameters passed to the CarringtonSolarWin structure for L3c processing
    idx_current_CR = np.abs(sw_ecliptic.CR_grid - ion_rate_Carr.carr_ion_rate['CR']).argmin()
    solar_wind_Carr.sw_ecliptic['mean_speed'] = sw_ecliptic.mean_speed[idx_current_CR]
    solar_wind_Carr.sw_ecliptic['mean_proton_density'] = sw_ecliptic.mean_proton_density[idx_current_CR]
    solar_wind_Carr.sw_ecliptic['mean_alpha_abundance'] = sw_ecliptic.mean_alpha_abundance[idx_current_CR]

    # Solar wind parameters passed to the CarringtonIonizationRate structure for L3c processing
    ion_rate_Carr.sw_ecliptic['mean_speed'] = sw_ecliptic.mean_speed[idx_current_CR]
    ion_rate_Carr.sw_ecliptic['mean_proton_density'] = sw_ecliptic.mean_proton_density[idx_current_CR]

    # Calculate averaged ionization rate profile and save it to the file
    ion_rate_Carr.calculate_photoion(ANC_INPUT_FROM_INSTRUMENT_TEAM, EXT_DEPENDENCIES)

    ion_rate_Carr.calculate_averaged_profile()
    ion_rate_Carr.calculate_charge_exchange()
    ion_rate_Carr.save_to_file(l3b_fn)

    # L3c Processing  
    solar_wind_Carr.read_l3b(l3b_fn)
    solar_wind_Carr.calculate_sw_profile()
    solar_wind_Carr.save_to_file(l3c_fn)
