'''
Author: Izabela Kowalska-Leszczynska, ikowalska@cbk.waw.pl
Modified by Menlo
Main procedure to generate GLOWS L3b and L3c data products
'''
import warnings

import numpy as np
from astropy.time import Time

from . import funcs as fun
from . import l3b_CarringtonIonRate as cir
from . import l3b_DailyIonRate as dir
from . import l3c_CarringtonSolarWind as csw
from . import l3c_EclipticSolarWind as esw
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies


def generate_l3bc(dependencies: GlowsL3BCDependencies):
    CR = dependencies.carrington_rotation_number
    print('Processing CR=' + str(CR))

    # Daily ionization rate from the daily light curves
    daily_ion_rate = []
    l3a_used_fn = []

    for l3a in dependencies.l3a_data:
        ion_rate_daily = dir.DailyIonizationRate(
            dependencies.ancillary_files)  # daily ionization rate object initialization
        ion_rate_daily.read_l3a(l3a)  # reading L3a data (masked lightcurve)
        flag = ion_rate_daily.read_model_params(
            dependencies.ancillary_files)  # reading WawHelioIon-MP parameters for lightcurve to ionization rate conversion
        if flag == ion_rate_daily.settings['no_vg_point_flag']: continue
        l3a_used_fn.append(l3a['filename'])
        ion_rate_daily.calculate_daily_ionization_rate()  # conversion to the daily ionization rate profile
        daily_ion_rate.append(ion_rate_daily.ion['ion_rate'])

    # if len(daily_ion_rate) == 0: continue  # TODO: Menlo figure out how and where in the code we want to handle all days are in a bad season

    # Carrington averaged ionization rate (L3b) and Carrington averaged solar wind parameters  objects initialization
    ion_rate_Carr = cir.CarringtonIonizationRate(l3a_used_fn, dependencies.ancillary_files, dependencies.external_files)
    solar_wind_Carr = csw.CarringtonSolarWind(dependencies.ancillary_files)

    l3a_date_list = Time([l3a['start_time'] for l3a in
                          dependencies.l3a_data])  # TODO: Menlo - determine which dates, start, center, middle, are needed

    # Save daily ionization rates into CarringtonIonizationRate structure
    ion_rate_Carr.daily_ion_rate['date'] = l3a_date_list
    ion_rate_Carr.daily_ion_rate['ion_rate'] = daily_ion_rate

    ion_rate_Carr.carr_ion_rate['date'] = Time(fun.jd_fm_Carrington(CR + 0.5), format='jd')
    ion_rate_Carr.carr_ion_rate['CR'] = CR  # Carrington rotation number

    # Solar wind in the ecliptic plane processing
    sw_ecliptic = esw.EclipticSolarWind(dependencies.ancillary_files, CR)
    sw_ecliptic.calculate_invariant(dependencies.external_files)
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
    ion_rate_Carr.calculate_photoion(dependencies.ancillary_files, dependencies.external_files)

    ion_rate_Carr.calculate_averaged_profile()
    ion_rate_Carr.calculate_charge_exchange()
    l3b_data = ion_rate_Carr.generate_l3b_output()

    # L3c Processing
    solar_wind_Carr.read_l3b(l3b_data)
    solar_wind_Carr.calculate_sw_profile()

    return ion_rate_Carr, solar_wind_Carr
