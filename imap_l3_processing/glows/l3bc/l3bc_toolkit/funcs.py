'''
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Functions used in L3a->L3b processing that are not class methods
'''

import json
import logging

import astropy.constants as const
import astropy.units as u
import netCDF4
import numpy as np
from astropy.coordinates import get_sun
from astropy.time import Time

from .constants import PHISICAL_CONSTANTS

logging.basicConfig(level=logging.ERROR)


##########################
# Exceptions 
##########################
class ArrayShapeError(Exception):
    pass


class OutOfLimitsError(Exception):
    pass


####### Functions ##########
def base_funcs_def(i, sin_or_cos):
    """
    a wrapper using closure (a kind of function constructor)
    needed for definition of base function for ApproxFunction class

    Parameters:
    ----------------
    i : int
        number of the sin or cos
    sin_or_cos : str
        selecting sin or cos
    """

    if i == 0:
        def fun(angle):
            return 1.0 / (2.0 * np.sqrt(np.pi)) * np.ones_like(angle)

        return fun
    if sin_or_cos == 's':
        def fun_s(angle):
            return np.sin(i * angle) / np.sqrt(np.pi)

        return fun_s
    if sin_or_cos == 'c':
        def fun_c(angle):
            return np.cos(i * angle) / np.sqrt(np.pi)

        return fun_c
    # TODO: maybe we need some error if sin_or_cos is anything beside 'c' or 's'
    return 'None'


def calculate_lya_monthly(t, flux):
    '''
    Calculates data that are averaged over each Carrington rotation period. Time is mean value of decimal year in each period.
    '''
    t_carr, flux_carr = resize_by_carr(t, flux)
    t_mean = Time(np.array([np.mean(t.decimalyear) for t in t_carr]), format='decimalyear')
    flux_mean = np.array([f.mean() for f in flux_carr])

    return t_mean, flux_mean


def calculate_sw_energy_flux(n, v, p):
    '''
    Solar wind energy flux that is latitudinal invariant
    Le Chat et al. 2012a
    Parameters must be given as a object with unit

    Parameters:
    --------------
    n - protons number density
    v - plasma speed
    p - concentration of the alpha particles wrt to the protons

    Output:
    ------------
    energy_flux: energy flux at 1au in the ecliptic plane (invariant)
    '''
    energy_flux = n * (const.m_p + p * PHISICAL_CONSTANTS['m_alpha']) * v * (0.5 * v ** 2 + const.GM_sun / const.R_sun)

    return energy_flux.to('erg.s-1.cm-2')


def carrington(jd):
    '''
    Converts time in Julian Day format into Carrington Rotation number
    '''
    ans = (jd - PHISICAL_CONSTANTS['jd_carrington_first']) / PHISICAL_CONSTANTS['carrington_time'] + 1
    return ans


def cr_from_l3b_fn(fn):
    '''
    Extracts CR from the L3b filename. This may change if the naming convention will change
    '''
    cr = fn.split('_')[5]

    return cr


def cross_section_cx_LS(E):
    '''
    Lindsay&Stebbings 2005
    Returns cross_section as a astro.unit object
    
    Parameters:
    -------------
    E: energy as an astropy.unit object

    Output:
    ------------
    sigma: cross-section as a astropy.unit object
    
    '''
    EkeV = E.to('keV').value
    out = 1e-16 * (1 - np.exp(-67.3 / EkeV)) ** 4.5 * (4.15 - 0.531 * np.log(EkeV)) ** 2
    sigma = out * u.cm * u.cm
    return sigma


def cross_section_cx(E):
    '''
    Swaczyna et al. 2025
    https://doi.org/10.3847/1538-4365/adaf17
    Returns cross_section as a astro.unit object
    
    Parameters:
    -------------
    E: energy as an astropy.unit object

    Output:
    ------------
    sigma: cross-section as a astropy.unit object
    
    '''

    # Parameters
    A1 = 4.8569
    A2 = 21.906
    A3 = 31.487
    A4 = 0.12018
    A5 = 4.1402E-6
    A6 = 3.7524
    A7 = 8.8476E-12
    A8 = 6.1091

    EkeV = E.to('keV').value
    out = (1e-16) * (A1 * np.log(A2 / EkeV + A3)) / (1 + A4 * EkeV + A5 * EkeV ** A6 + A7 * EkeV ** A8)
    sigma = out * u.cm * u.cm

    return sigma


def derive_sin_cos(order):
    """
    equivalent of Mathematica's deriveSinCos function
    needed for definition of base function for ApproxFunction class
    input variable order is the order (number of sin/cos) of the approximator

    Parameters:
    ----------------
    order : str
        order of the decomposition
    """

    funcs = []
    for i in range(order + 1):
        if i == 0:
            funcs.append(base_funcs_def(i, 'None'))
        else:
            funcs.append(base_funcs_def(i, 's'))
            funcs.append(base_funcs_def(i, 'c'))
    return funcs


def f107_daily_data(t, flux):
    '''
    Returns adjusted F10.7 flux with daily cadence. In the case of multiple measurements during a day,
    the closest to the noon is chosen

    Parameters:
    ------------
    t:    time array of raw data (astropy.time)
    flux: F10.7 flux adjusted to 1au (10^(-22) W/m^2/Hz)

    Output:
    ------------
    t_out: daily time array (astropy.time)
    flux_out: daily F10.7 flux adjusted to 1au (10^(-22) W/m^2/Hz)
    '''
    idx_noon, idx_dup = select_noon(t)  # select one measurment per day
    idx_no0 = np.where(flux[idx_noon] > 0)[0]
    t_out = t[idx_noon[idx_no0]]
    flux_out = flux[idx_noon[idx_no0]]
    return t_out, flux_out


def find_CR_idx(CR, CR_list):
    '''
    finds index of the value in the CR_list that is closest to the CR
    '''
    idx_CR = np.abs(np.array(CR_list) - int(CR)).argmin()
    idx_read = [idx_CR - 1, idx_CR, idx_CR + 1]
    return idx_read


def generate_cr_lya(CR, data_lya):
    '''
    calculates Lyman-alpha composite irradiance interpolated on the center of a given CR
    '''

    t_CR = Time(jd_fm_Carrington(CR), format='jd')

    CR_list = [int(carrington(t.jd)) for t in data_lya[0]]
    idx_read = find_CR_idx(CR, CR_list)

    t_lya = np.array([data_lya[0][i].jd for i in idx_read])
    lya = np.array([data_lya[1][i] for i in idx_read])
    lya_CR = np.interp(t_CR.jd, t_lya, lya)

    return lya_CR


def generate_cr_solar_params(CR, data_l3b, data_l3c):
    '''
    calculates solar parameters (plasma speed, proton density, photoionization, electron density, and uv anisotropy)
    interpolated on the center of a given CR
    '''
    # find indexes of the previous and the next CR

    t_CR = Time(jd_fm_Carrington(CR), format='jd')

    CR_list = [data['CR'] for data in data_l3b]
    idx_read = find_CR_idx(CR, CR_list)

    t_b = Time(np.array([data_l3b[i]['date'] for i in idx_read]))
    t_c = Time(np.array([data_l3c[i]['date'] for i in idx_read]))  # t_c should be the same as t_b

    # UV-anisotropy
    anisotropy = np.array([data_l3b[i]['uv_anisotropy_factor'] for i in idx_read])
    anisotropy_CR = np.array([np.interp(t_CR.jd, t_b.jd, anisotropy[:, i]) for i in np.arange(len(anisotropy[0]))])

    # photoion in the ecliptic plane
    Nmid = int(len(data_l3b[0]['ion_rate_profile']['lat_grid']) / 2)
    ph_b = np.array([data_l3b[i]['ion_rate_profile']['ph_rate'][Nmid] for i in idx_read])
    ph_ion_CR = np.interp(t_CR.jd, t_b.jd, ph_b)

    # solar wind parameters (plasma speed, proton density)
    p_dens = np.array([data_l3c[i]['solar_wind_profile']['proton_density'] for i in idx_read])
    sw_speed = np.array([data_l3c[i]['solar_wind_profile']['plasma_speed'] for i in idx_read])
    p_dens_CR = np.array([np.interp(t_CR.jd, t_c.jd, p_dens[:, i]) for i in np.arange(len(p_dens[0]))])
    sw_speed_CR = np.array([np.interp(t_CR.jd, t_c.jd, sw_speed[:, i]) for i in np.arange(len(sw_speed[0]))])

    # electron density in the ecliptic plane calculated from p-dens and alpha-abundance
    p_dens_ecl = np.array([data_l3c[i]['solar_wind_ecliptic']['proton_density'] for i in idx_read])
    a_abundance_ecl = np.array([data_l3c[i]['solar_wind_ecliptic']['alpha_abundance'] for i in idx_read])
    e_dens_ecl = p_dens_ecl * (1 + 2 * a_abundance_ecl)
    e_dens_CR = np.interp(t_CR.jd, t_c.jd, e_dens_ecl)

    return anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read


def get_spher_coord_rad(vec):
    """
    Compute radius r, longitude l and latitude b for a vector vec

    Parameters
    ----------
      vec: 1D or 2D numpy array
        input vector(s)

    Returns
    -------
      if vec is an 1D numpy array, return 1D numpy array rlb (see below)
      if vec is a 2D numpy array, assume that its rows represent vectors and return 2D array
        rlb (see below), where every row of rlb corresponds to every row of vec

      rlb: 1D or 2D numpy array
        if 1D then rlb=[r,l,b] where: r - magnitude of vector vec, l - longitude of the direction
          pointed by vec in radians, b - latitude of the direction pointed by vec in radians
        if 2D then columns of the array contain r,l,b

    Examples
    --------
      >>> import geom_fun_libr_box as gflb
      >>> import numpy as np
      >>> rlb=gflb.get_spher_coord_rad(np.array([0,1,1]))
      >>> print(rlb[0],np.degrees(rlb[1]),np.degrees(rlb[2]))
      1.41421356237 90.0 45.0
      >>> n=10
      >>> phi=np.radians(np.linspace(0,360,n))
      >>> tht=np.radians(np.linspace(-90,90,n))
      >>> vecs=np.column_stack([np.cos(phi)*np.cos(tht),np.sin(phi)*np.cos(tht),np.sin(tht)])
      >>> print(gflb.get_spher_coord_rad(vecs))
      [[ 1.          0.         -1.57079633]
       [ 1.          0.6981317  -1.22173048]
       [ 1.          1.3962634  -0.87266463]
       [ 1.          2.0943951  -0.52359878]
       [ 1.          2.7925268  -0.17453293]
       [ 1.          3.4906585   0.17453293]
       [ 1.          4.1887902   0.52359878]
       [ 1.          4.88692191  0.87266463]
       [ 1.          5.58505361  1.22173048]
       [ 1.          6.28318531  1.57079633]]
    """
    if vec.ndim == 1:
        if np.shape(vec)[0] != 3:
            raise "get_spher_coord_rad() error: np.shape(vec)[0]!=3"
        r = np.sqrt(np.dot(vec, vec))
        if r == 0.0:
            b = 0.0;
        else:
            b = np.arcsin(vec[2] / r)  # compute latitude
        # compute longitude
        if vec[0] == 0.0 and vec[1] == 0.0:
            l = 0.0
        else:
            l = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
        return np.array([r, l, b])  # return one array similarly as in Mathematica
    else:
        if np.shape(vec)[1] != 3:
            raise "get_spher_coord_rad() error: np.shape(vec)[1]!=3"
        r = np.sqrt((vec * vec).sum(axis=1))
        idxs = np.nonzero(r != 0.0)
        b = np.zeros((len(r)), np.float64)
        b[idxs[0]] = np.arcsin(vec[idxs[0], 2] / r[idxs[0]])  # compute latitude
        l = np.arctan2(vec[:, 1], vec[:, 0]) % (2 * np.pi)  # compute longitude
        idxs = np.nonzero(np.logical_and(vec[:, 0] == 0.0, vec[:, 1] == 0.0))
        l[idxs[0]] = 0.0  # set l=0 if vec[:,0]==0.0 and vec[:,1]==0.0
        return np.column_stack([r, l, b])  # return one array similarly as in Mathematica


def jd_fm_Carrington(carr):
    '''
    calculates time in format of Julian Days for a given Carrington Rotation number
    '''
    ans = (carr - 1) * PHISICAL_CONSTANTS['carrington_time'] + PHISICAL_CONSTANTS['jd_carrington_first']
    return ans


def make_lsq_matrix(psi, base_funcs):
    """
    make data matrix and vector for least-squares method in the matrix form
    needed for ApproxFunction class
    psi is the angle in radians

    Parameters:
    ----------------
    psi : 
    base_funs: 
    """

    lsq_mtrx = np.zeros((len(psi), len(base_funcs)), float)
    i = 0
    for fun in base_funcs:
        lsq_mtrx[:, i] = fun(psi)
        i = i + 1
    return lsq_mtrx


def process_omni_param(omni_raw, cr_grid, param_settings):
    '''
    Processes OMNI2 raw data to obtain averaged solar wind speed, density and alpha-particles abundance
    '''

    # select from the raw data columns relevant for a given parameter
    param_raw = omni_raw[:, param_settings['column_numbers']]

    # remove gaps where there was no data available based on coded value
    param = param_raw[param_raw[:, 3] < param_settings['gap_marker']]

    date = time_from_yday(param)
    date_cr = carrington(date.jd)

    # Scale parameter to 1au. OMNI Observations are done at different distance and we need to scale proton density
    if param_settings['scale']: param[:, 3] = scale_density(date, param[:, 3])

    # split array into 1-Carrington chunks
    param_cr, idx_param = np.unique(np.floor(date_cr), return_index=True, axis=0)
    param_s = np.split(param[:, 3], idx_param)[1:]

    # calculate averaged over 1 Carrington time and parameter value
    param_value = np.array([p.mean() for p in param_s])

    # interpolate into cr_grid
    if np.logical_and(param_cr[0] <= cr_grid[0], param_cr[-1] >= cr_grid[-1]):
        param_value_interp = np.interp(cr_grid, param_cr, param_value)
    else:
        logging.info('There are gaps in the OMNI2 data that cannot be filled by the interpolation')
        raise Exception('OMNI Error: not enough data for interpolation')

    return param_value_interp


def read_f107_raw_data(fn):
    '''
    Read raw F10.7 data from file

    Parameters:
    ------------
    fn: name of the file with daily F10.7 measurements (str)

    Output:
    ------------
    t0:  time array (astropy.time)
    flux_adjusted: F10.7 flux adjusted to 1au (10^(-22) W/m^2/Hz)    
    '''

    data = np.loadtxt(fn, skiprows=2)

    jd = data[:, 2]  # time of observation in JD
    flux_adjusted = data[:,
                    5]  # adjusted do 1AU flux in sfu 1sfu=10^(-22) W/m^2/Hz (taking into accont that Sun-Earth distance is changing)
    t0 = Time(jd, format='jd')  # time converted to astropy.time format

    return t0, flux_adjusted


def read_hdr_txt(fn, N):
    '''
    Reads header from the text file and preapers it to the further analysis
    '''

    # open file and read header
    with open(fn) as file:
        hdr = [next(file) for _ in range(N)]

    # remove hash 
    hdr_str = remove_hash(hdr)
    return hdr_str


def read_lya_data(fn):
    '''
    reads daily Lyman-alpha composite irradiance and calculates monthly averages
    '''
    t_lya, flux_lya, _ = read_lya_raw_data(fn)
    t_lya_m, flux_lya_m = calculate_lya_monthly(t_lya, flux_lya)
    return [t_lya_m, flux_lya_m]


def read_lya_raw_data(filename):
    '''
    Reads *.nc file with composite Lyman-alpha solar irradiance (from LAPS)
    Returns daily product adjusted to 1au. Flux is in ph/(cm^2 s^2)
    '''

    data = netCDF4.Dataset(filename)

    convert_to_ph = data.variables['convert_to_photons'][
        0].data  # Multiplicative factor to convert irr_121 from W/m^2 to ph/(cm^2 s^2) [ph / (cm^2 s^2)] / (W / m^2)
    t_d1947 = data.variables['time'][:].data  # time of observation [days since 1947-01-01 12:00:00 UTC]
    flux0 = data.variables['irr_121'][
            :].data  # Averaged daily irradiance in band from 121.0 to 122.0 nm at 1 AU [W/m^2]
    flux = flux0 * convert_to_ph  # Flux in ph/(cm^2 s^2)
    flux_uncert = data.variables['irr_121_uncertainty'][
                  :].data  # Uncertainty (1 standard deviation) of average daily irradiance [W/m^2]

    t0 = Time('1947-01-01 12:00')

    # convertion to jd and adding days since 1947-01-01 - it is more accurate than simple adding days to date   
    t = Time(t0.jd + t_d1947, format='jd')

    return t, flux, flux_uncert


def read_raw_OMNI_data(ext_dependencies, t_window=None):
    '''
    Reads 3 types of data from OMNI2 database: plasma speed, proton density, and alpha-particles abundance in the ecliptic plane
     
    Parameters:
    ------------
    ext_dependencies : dict
        list of file names with external dependencies
    t_window: list [astropy.time, astropy.time]
        time window of the data that are needed
    '''
    # Read OMNI2 data (hour resolution)
    # year, DOY, hour, S/C Id, proton_dens, plasma_speed, alpha_abundance, sigma proton_dens, sigma plasma_speed, sigma alpha_abundance
    omni_raw = np.loadtxt(ext_dependencies['omni_raw_data'], usecols=(0, 1, 2, 5, 23, 24, 27, 30, 31, 34))
    omni_raw_date = time_from_yday(omni_raw[:, (0, 1, 2)])

    if t_window == None:
        out = omni_raw
    else:
        idx_ini = np.abs(omni_raw_date - t_window[0]).argmin()
        idx_fin = np.abs(omni_raw_date - t_window[1]).argmin()

        out = omni_raw[idx_ini:idx_fin]

    return out


def read_json(fn):
    '''
    Opens, reads and returns the content of the json file
    '''
    fp = open(fn, 'r')
    data = json.load(fp)
    fp.close()
    return data


def remove_hash(hdr):
    '''
    Removes # from the header that is read from the text files and returns string that will 
    be used to generate another header
    '''
    hdr_str = [l[2:] for l in hdr]
    return hdr_str


def resize_by_carr(t, flux):
    '''
    Reshaping flux array in such a way that each element is a list of fluxes from the same carrington rotation number
    '''
    t_carr = carrington(t.jd)
    n_carr = np.array([int(t_carr[i]) for i in np.arange(len(t_carr))])
    flux_carr_sort = np.split(flux, np.unique(n_carr, return_index=True)[1][1:])
    t_carr_sort = np.split(t, np.unique(n_carr, return_index=True)[1][1:])
    return t_carr_sort, flux_carr_sort


def scale_density(date, dens):
    '''
    Scaling density measurements that are taken at different distance than 1au.
    We need solar wind parameters ajusted to 1au.
    Proton density decreases as 1/r^2

    Parameters:
    --------------
    date : astropy.time
        time of the measurement
    dens : 1D float vector
        measured density [#/cm^3]
    '''

    # Sun-Earth distance at given moment
    d_sun = get_sun(date).distance

    # Sun-L1 distance
    r_obs = d_sun * (1 - (const.M_earth / (3 * const.M_sun)) ** (1. / 3))

    # density ajustment
    dens_1au = dens * (r_obs / u.au) ** 2

    return dens_1au


def select_data_window(data, t_window):
    '''
    Selects data whitin a given time window

    Parameters:
    ----------------
    data: [time, value]
        list of two 1D vectors 
    t_ini: astropy.time
    t_end: astropy.time

    '''
    idx = np.where(np.logical_and(data[0] >= t_window[0], data[0] <= t_window[1]))[0]
    t = data[0][idx]
    data = data[1][idx]

    return [t, data]


def select_noon(t_pen):
    '''
    Select the dates that are closest to the local noon

     Parameters:
     ------------
     t_pen:  time array of all measurements (astropy.time)
   
    Output:
    ------------
        1. index list of all data taken at noon (20 UT) or the closest one if there is no data at noon
        2. a list of indexes of data taken on the same day (if there is only one measurement then this 
            element is just 1 element list np. [[1],[2,3,4],[6]]
    '''
    # TODO it should be done in a dynamical way - it should determine how many measurements are in a given day
    index = np.array([0])
    index_duplicate = []
    i = 0
    while i < (len(t_pen) - 3):

        if (t_pen[i].yday[:8] == t_pen[i + 1].yday[:8]):

            if (t_pen[i].yday[:8] == t_pen[i + 2].yday[:8]):
                if (t_pen[i].yday[:8] == t_pen[i + 3].yday[:8]):
                    h = np.array([int(t_pen[i].yday[9:11]), int(t_pen[i + 1].yday[9:11]), int(t_pen[i + 2].yday[9:11]),
                                  int(t_pen[i + 3].yday[9:11])])
                    k = i + np.abs(h - 20).argmin()  # index of the hour that is near 20UT (local noon)
                    d = [i, i + 1, i + 2, i + 3]
                    i = i + 4
                else:
                    h = np.array([int(t_pen[i].yday[9:11]), int(t_pen[i + 1].yday[9:11]), int(t_pen[i + 2].yday[9:11])])
                    k = i + np.abs(h - 20).argmin()  # index of the hour that is near 20UT (local noon)
                    d = [i, i + 1, i + 2]
                    i = i + 3
            else:
                h = np.array([int(t_pen[i].yday[9:11]), int(t_pen[i + 1].yday[9:11])])
                k = i + np.abs(h - 20).argmin()  # index of the hour that is near 20UT (local noon)
                d = [i, i + 1]
            i = i + 2
        else:
            k = i
            d = [i]
            i = i + 1
        index = np.append(index, k)
        index_duplicate.append(d)

    return index[1:], index_duplicate


def test_WawHelioInMP_file(model):
    '''
    test if json file has what it should inside
    '''

    vg_points_number = model['header']['vg_points_number']
    hion_filtration = model['params_hion']['hion_filtration']
    hion_order = model['params_hion']['hion_order']

    hion_const_0 = np.array(model['params_hion']['hion_const']).shape[0]
    hion_base = np.array(model['params_hion']['hion_base'])
    hion_base_0 = np.array(model['params_hion']['hion_base']).shape[0]
    hion_base_1 = np.array(model['params_hion']['hion_base']).shape[1]

    if vg_points_number != len(model['vg_points']):
        logging.error('vg_points_number is incorrect.\nIt is ' + str(vg_points_number)
                      + '\nIt should be ' + str(len(model['vg_points'])))
        raise ArrayShapeError('Array shape error: Array shape is incorrect')

    elif hion_const_0 != (hion_order - 1 - hion_filtration):
        logging.error('hion_const shape is incorrect.\nIt is ' + str(hion_const_0)
                      + '\nIt should be ' + str(hion_order - 1 - hion_filtration))
        raise ArrayShapeError('Array shape error: Array shape is incorect')

    elif np.logical_or(hion_base_0 != hion_order - 1, hion_base_1 != hion_order + 1):
        logging.error('hion_base shape is incorrect.\nIt is ' + str(hion_base.shape)
                      + '\nIt should be (' + str(hion_order - 1) + ',' + str(hion_order + 1) + ')')
        raise ArrayShapeError('Array shape error: Array shape is incorect')

    for k in np.arange(model['header']['vg_points_number']):

        lcrv_filtration = model['vg_points'][k]['parameters']['lcrv_filtration']
        lcrv_order = model['vg_points'][k]['parameters']['lcrv_order']

        model_const_0 = np.array(model['vg_points'][k]['parameters']['model_const']).shape[0]
        model_matrix = np.array(model['vg_points'][k]['parameters']['model_matrix'])
        model_matrix_0 = np.array(model['vg_points'][k]['parameters']['model_matrix']).shape[0]
        model_matrix_1 = np.array(model['vg_points'][k]['parameters']['model_matrix']).shape[1]

        lcrv_Trmatrix = np.array(model['vg_points'][k]['parameters']['lcrv_Trmatrix'])
        lcrv_Trmatrix_0 = np.array(model['vg_points'][k]['parameters']['lcrv_Trmatrix']).shape[0]
        lcrv_Trmatrix_1 = np.array(model['vg_points'][k]['parameters']['lcrv_Trmatrix']).shape[1]

        if model_const_0 != hion_filtration:
            logging.error('model_const shape is incorrect.\nIt is ' + str(model_const_0)
                          + '\nIt should be ' + str(hion_filtration))
            raise ArrayShapeError('Array shape error: Array shape is incorrect')

        elif np.logical_or(model_matrix_0 != hion_filtration, model_matrix_1 != lcrv_filtration):
            logging.error('model_matrix shape is incorrect.\nIt is ' + str(model_matrix.shape)
                          + '\nIt should be (' + str(hion_filtration) + ',' + str(lcrv_filtration) + ')')
            raise ArrayShapeError('Array shape error: Array shape is incorect')

        elif np.logical_or(lcrv_Trmatrix_0 != 2 * lcrv_order + 1, lcrv_Trmatrix_1 != 2 * lcrv_order + 1):
            logging.error('lcrv_Trmatrix shape is incorrect.\nIt is ' + str(lcrv_Trmatrix.shape)
                          + '\nIt should be (' + str(2 * lcrv_order + 1) + ',' + str(2 * lcrv_order + 1) + ')')
            raise ArrayShapeError('Array shape error: Array shape is incorect')


def time_string_from_l3a_fn(fn):
    '''
    Extracts date from the L3a filename. This may change if the naming convention will change
    '''
    fn_string = fn.split('_')[4]
    time_string = fn_string[0:4] + '-' + fn_string[4:6] + '-' + fn_string[6:8] + ' ' + fn_string[
                                                                                       8:10] + ':' + fn_string[
                                                                                                     10:12] + ':' + fn_string[
                                                                                                                    12:14]
    return time_string


def time_from_yday(yday_list):
    string = [str(int(yday_list[i, 0])) + ':' + str(int(yday_list[i, 1])) + ':' + str(int(yday_list[i, 2])) + ':0' for i
              in range(len(yday_list))]
    date = Time(string, format='yday')
    return date


def v2E(v):
    '''
    Speed to kinetic energy transformation for hydrogen

    Parameters:
    -------------
    v: speed as an astropy.unit object

    Output:
    ------------
    E: energy as an astropy.unit object
    '''

    m_H = 1.6733e-24 * u.g
    E = (0.5 * m_H * v ** 2).to('eV')
    return E
