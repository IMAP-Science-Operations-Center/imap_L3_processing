'''
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Daily ionization rate class
'''

import json
import logging
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from . import funcs as fun
from .constants import VERSION

logging.basicConfig(level=logging.ERROR)


##########################
# Exceptions 
##########################
class ArrayShapeError(Exception):
    pass


class OutOfLimitsError(Exception):
    pass


class WawHelioIonMPError(Exception):
    pass


@dataclass
class DailyIonizationRate():
    '''
    Class for a GLOWS daily ionization rate
    Contains original light curve, parameters used to convert light curve into 
    ionization profile and ionization rate profile

    Attributes
    ----------
    header: dict
        software_version : float
            L3b-c software version
        ancillary_files : str
            list of ancillary files used in calculations
        l3a_input_filename: str
            Input L3a lightcurve filename
    settings: dict
        pipeline settings
    spacecraft_location: list
        spacecraft average location x,y,z in ecliptic heliocentric reference frame [km]
    lcrv: dict
        L3a data
        date : astropy.time
            date of the light curve as a astropy.time object
        ecl_lon : float [deg]
            ecliptic longitude of the spacecraft
        sc_coord : astropy.SkyCoord
            scanning circle centre coordinates as a astropy.SkyCoord object
        spin_angle : float 1D vector
            spin angle from the N, anty clock-wise [deg], not regular
        photon_flux : float 1D vector
            light curve intensity measured by GLOWS [R]
        flux_uncertainties: float 1D vector
            measurement uncertainties of the flux [R]
        bkg_galaxy : float 1D vector
            galaxy background estimation [R]
        bkg_time : float 1D vector
            time-dependent component of the signal background [R]
        glow_flux : float 1D vector
            photon flux with subtracted background [R]
        glow_flux_swan : float 1D vector
            glow flux rescaled to the SOHO/SWAN units [SWAN R]
        decomposition_coeff: np.array
            list of coeffiecents in sin cos decomposition of the light curve
    model: dict
        WawHelioIon-MP parameters to convert light curve into ionization rate
        params_hion: dict
            hion_order: int
                order of the ionization profile decomposition into Legendre’s polynomials]
            hion_filtration: int 
                number of significant terms in ionization expansion
            hion_const: float 1D vector [size: hion_order-1-hion_filtration]
                constant vector that completes ionization coefficients vector
                in PCA base to the size defined by hion_order
            hion_base: float 2D array [size: (hion_order+1-2, hion_order+1)]
                array of coefficients of base polynomials to calculate
                total ionization rate profile as a function of x=sin(latitude).
                Number of polynomials is hion_order + 1(0-order polynomial -> constant) -2
                (requirement that the 1st derivative should be smooth at both poles)
        vg_points
            pos_name: str
                cardinal position label.
                ['upw','cwF','dnw','cwS'] correspond to ecliptic latitude [252.4,342.4,72.4,162.4]
                digits correspond to degree from the cardinal point
                (e.g. cwS30 corresponds to ecliptic latitude 162.4+30=192.4 deg)
            ecl_lon: float
                ecliptic longitude of the bin center
            ecl_lon_ini: float
                initial ecliptic longitude of the valid bin
            ecl_lon_fin: float
                final ecliptic longitude of the valid bin
            parameters
                lcrv_order: int
                    order of the light curve decomposition into sin/cos series
                lcrv_filtration: int
                    number of significant terms in light curve expansion after PCA transformation
                lcrv_Tmatrix: float 2D array [size: (2*lcrv_order+1,2*lcrv_order+1)]
                    matrix for PCA transformation of decomposed light curve
                lcrv_const: float 1D vector [size: (hion_filtration)]
                    constant vector added to the transformed (by PCA) light curve coefficients
                lcrv_matrix: float 2D array [size: (hion_filtration, lcrv_filtration)]
                    the main matrix that converts light curve coefficients in PCA base into ionization coefficients in PCA base
    ion: dict
        ionization rate profile
        ion_grid: float 1D vector
            latitudinal grid 
        ion_rate: float 1D vector
            ionization rate value [#/s]
        decomposition_coeff: 1D vector
            list of coeffiecents in Legendre approximation of the ionization rate profile


    Methods
    -------
    calculate_daily_ionization_rate()
        main method that calculates daily ionization rate
    _calculate_ion_profile(ion_profile_fun)
        calculates ionization rate profile on the final grid
    _decompose_light_curve()
        decomposes rescaled do SWAN lightcurve into the base functions using regression methods and returns vector of coeffieciens
    _find_vg_ecl(model)
        finds the closest vantage point in our model to the spacecraft current position
    read_l3a(filename)
        reads original L3a lightcurve
    read_model_params(anc_input_from_instr_team)
        reads parameters of the WawHelioIon-MP model needed to conversion lightcurve-> ionization rate
    _rescale_to_SWAN()
        rescales estimated glow flux measured by GLOWS to the SOHO/SWAN units
    save_to_file(fn)
        saves daily ionization rate profiles to file [TEMP]
    _subtract_background()
        subtracts galaxy and time dependent background from the measured photon flux
    _total_ion_from_lcrv_params()
        calculates daily ionization rate profile from the lightcurve decomposed coeffiecients
    _total_ion_params_from_lcrv_params()
        calculates ionization rate Legendre decomposition coeffiecients from the the lightcurve decomposed coeffiecients

    '''

    def __init__(self, anc_input_from_instr_team):
        """
        Class constructor
        """

        self.header = {
            'software_version': VERSION,
            'ancillary_data_files': anc_input_from_instr_team['WawHelioIonMP_parameters'],
            'l3a_input_filename': None
        }

        # read pipeline settings to be used for L3b
        self.settings = fun.read_json(anc_input_from_instr_team['pipeline_settings'])

        # All parameters will be read from L3a file or computed later on
        self.spacecraft_location = None

        self.lcrv = {}
        self.lcrv['date'] = None
        self.lcrv['ecl_lon'] = None
        self.lcrv['sc_coord'] = None
        self.lcrv['spin_angle'] = None
        self.lcrv['photon_flux'] = None
        self.lcrv['flux_uncertainties'] = None
        self.lcrv['bkg_galaxy'] = None
        self.lcrv['bkg_time'] = None
        self.lcrv['glow_flux'] = None
        self.lcrv['glow_flux_swan'] = None
        self.lcrv['decomposition_coeff'] = None

        self.model = {}
        self.model['params_hion'] = {}
        self.model['params_hion']['hion_order'] = None
        self.model['params_hion']['hion_filtration'] = None
        self.model['params_hion']['hion_const'] = None
        self.model['params_hion']['hion_base'] = None
        self.model['vg_points'] = {}
        self.model['vg_points']['pos_name'] = None
        self.model['vg_points']['ecl_lon'] = None
        self.model['vg_points']['ecl_lon_ini'] = None
        self.model['vg_points']['ecl_lon_fin'] = None
        self.model['vg_points']['parameters'] = {}
        self.model['vg_points']['parameters']['lcrv_order'] = None
        self.model['vg_points']['parameters']['lcrv_filtration'] = None
        self.model['vg_points']['parameters']['lcrv_Tmatrix'] = None
        self.model['vg_points']['parameters']['lcrv_const'] = None
        self.model['vg_points']['parameters']['lcrv_matrix'] = None

        self.ion = {}
        self.ion['ion_grid'] = self.settings['ion_rate_grid']
        self.ion['ion_rate'] = None
        self.ion['decomposition_coeff'] = None

    def calculate_daily_ionization_rate(self):
        '''
        Main method that calculates daily ionization rate profile step-by-step
        '''

        # subtract background
        self._subtract_background()
        # rescale to SWAN
        self._rescale_to_SWAN(self.settings['swan_scaling_factor'])
        # decompose  lightcurve into coeffiecents vector
        self._decompose_light_curve()
        # calculate function that describes ionization rate profile
        ion_profile_fun = self._total_ion_from_lcrv_params()
        # calculate ionization rate profile on the ionization grid
        self._calculate_ion_profile(ion_profile_fun)

    def _calculate_ion_profile(self, ion_profile_fun):
        '''
        Generating tabular profile on ionization grid defined in pipeline settings

        Parameters
        -----------
        ion_profile_fun 
    
        '''

        bins = np.arange(-90, 100, 10)  # latitude in deg
        # sin, not cos at it is in Legendre's because in astronomy latitude is meaured
        # from the equator, not from z axis as it is in mathematics
        ion = ion_profile_fun(np.sin(np.radians(bins))) / 1e6
        self.ion['ion_grid'] = bins
        self.ion['ion_rate'] = ion

    def _decompose_light_curve(self):
        '''
        Decomposition GLOWS light curve rescaled to SWAN units into sin, cos series
        As a result, the list of coefficients is calculated
        '''
        order = self.model['vg_points']['parameters']['lcrv_order']
        uncert = self.lcrv['flux_uncertainties']
        lightcurve = self.lcrv['glow_flux_swan']
        method = self.settings['lcrv_approximation_method']
        alpha = self.settings['lcrv_approximation_alpha']

        # define basis functions
        base_funcs = fun.derive_sin_cos(order)
        # compute matrix for the least-squares method
        lsq_mtrx = fun.make_lsq_matrix(np.radians(self.lcrv['spin_angle']), base_funcs)
        # fit the linear regression model

        if method == 'LinearRegression':
            lin_model = LinearRegression(fit_intercept=False).fit(lsq_mtrx, lightcurve,
                                                                  sample_weight=1.0 / (uncert * uncert))
        elif method == 'Lasso':
            lin_model = Lasso(alpha=alpha, fit_intercept=False).fit(lsq_mtrx, lightcurve,
                                                                    sample_weight=1.0 / (uncert * uncert))
        elif method == 'Ridge':
            lin_model = Ridge(alpha=alpha, fit_intercept=False).fit(lsq_mtrx, lightcurve,
                                                                    sample_weight=1.0 / (uncert * uncert))
        else:
            raise Exception('Unknown regression method. Use LinearRegression, Lasso or Ridge')

        self.lcrv['decomposition_coeff'] = lin_model.coef_

    def _find_vg_ecl(self, model):
        '''
        find the index of the vg_point that is closest to the lightcurve, but whitin defined boundaries.

        Parameters
        -----------
        model : dict
            Structure read from the WawHelioIon-MP json file
        '''

        model_ecl_bins = np.array([[model['vg_points'][i]['ecl_longIni'], model['vg_points'][i]['ecl_longFin']] for i in
                                   range(len(model['vg_points']))])
        k = self.settings['no_vg_point_flag']  # if there is no bin in vg_points main loop should be interupted
        for i, ecl_bin in enumerate(model_ecl_bins):
            ini, fin = ecl_bin
            # Case when we are not crossing 0-360
            if ini <= fin:
                if ini <= self.lcrv['ecl_lon'] <= fin: k = i
            # Case when we are crossing 0-360
            else:
                if np.logical_or(self.lcrv['ecl_lon'] >= ini, self.lcrv['ecl_lon'] <= fin): k = i
        return k

    def read_l3a(self, data_l3a: dict):
        '''
        Read light curve from the L3a json file

        Parameters
        -----------
        filename : str
            Path to the L3a data product file
        '''
        # Open and load json file with L3a data

        self.spacecraft_location = list(data_l3a['spacecraft_location_average'].values())
        self.lcrv['ecl_lon'] = np.degrees(fun.get_spher_coord_rad(np.array(self.spacecraft_location))[1])
        self.lcrv['date'] = Time(data_l3a['start_time']) + 0.5 * (
                Time(data_l3a['end_time']) - Time(data_l3a['start_time']))
        self.lcrv['sc_coords'] = SkyCoord(lon=data_l3a['spin_axis_orientation_average']['lon'],
                                          lat=data_l3a['spin_axis_orientation_average']['lat'],
                                          unit=u.deg,
                                          frame='heliocentrictrueecliptic')  # scaning circle centre coordinates
        self.lcrv['spin_angle'] = np.array(
            data_l3a['daily_lightcurve']['spin_angle'])  # spin angle grid [masked bins are missing]
        self.lcrv['photon_flux'] = np.array(data_l3a['daily_lightcurve']['photon_flux'])  # signal in physical units
        self.lcrv['flux_uncertainties'] = np.array(
            data_l3a['daily_lightcurve']['flux_uncertainties'])  # measurements uncertaintie
        self.lcrv['bkg_galaxy'] = np.array(
            data_l3a['daily_lightcurve']['extra_heliospheric_bckgrd'])  # constant extra-galactic background
        self.lcrv['bkg_time'] = np.array(
            data_l3a['daily_lightcurve']['time_dependent_bckgrd'])  # time-dependend background
        self.lcrv['glow_flux'] = self.lcrv['photon_flux'] - self.lcrv['bkg_galaxy'] - self.lcrv[
            'bkg_time']  # glow flux in physical units

    def read_model_params(self, anc_input_from_instr_team):
        '''
        Read WawHelioIon-MP parameters

        Parameters:
        -----------
        anc_input_from_instr_team : dict
            ANC_INPUT_FROM_INSTRUMENT_TEAM from the toolkit.constants 
            Filenames of the anillary files

        Output:
        returns flag that is non 0 if light curve can't be converted to the ionization rate due to the gap in 
        WawHelioIon-MP
        '''

        # read file
        model = fun.read_json(anc_input_from_instr_team['WawHelioIonMP_parameters'])

        # find vantage point in the WawHelioIon-MP file that isvalid to analyse current light curve
        vg_idx = self._find_vg_ecl(model)
        if vg_idx == self.settings['no_vg_point_flag']:
            # Light curve can not be converted to the ionization rate profile - WawHelioIon-MP could not be train in that part of the orbit
            return self.settings['no_vg_point_flag']

        self.model['params_hion'] = model['params_hion']
        self.model['vg_points'] = model['vg_points'][vg_idx]
        return 0

    def _rescale_to_SWAN(self, sf):
        '''
        GLOWS ligth curve without background needs to be rescaled to the SWAN units, because WaWHelioIon-MP was tuned on SWAN data

        Parameters
        -----------
        sf : vector
            coefficients of the linear scaling function
        '''
        self.lcrv['glow_flux_swan'] = self.lcrv['glow_flux'] * sf[0] + sf[1]

    def save_to_file(self, fn):
        '''
        Daily ionization rate files are not official GLOWS data product.
        They are needed to calculate average profile, so they will be calculated but not saved to a files.
        This method is for testing purposes

        Parameters
        -----------
        fn : str
            Path to the daily L3b file
        '''

        # Dictionary with the daily ionization rate profile that will be saved in json file
        output = {}
        output['header'] = self.header

        output['data'] = {}
        output['data']['date'] = self.lcrv['date'].iso
        output['data']['ion_grid'] = self.ion['ion_grid'].tolist()
        output['data']['ion_rate'] = self.ion['ion_rate'].tolist()

        json_content = json.dumps(output, indent=3)

        output_fp = open(fn, 'w')
        print(json_content, file=output_fp)
        output_fp.close()

    def _subtract_background(self):
        '''
        Subtracts galactic and time-dependent background from the measured photon flux
        '''
        self.lcrv['glow_flux'] = self.lcrv['photon_flux'] - (self.lcrv['bkg_galaxy'] + self.lcrv['bkg_time'])

    def _total_ion_from_lcrv_params(self):
        '''
        Calculates total ionization rate profile
        for a set of light curve decomposition parameters.
        '''
        hion_base = np.array(self.model['params_hion']['hion_base'])
        total_ion_params = self._total_ion_params_from_lcrv_params()

        # Reconstruction of the total ionization rate profile using base functions and total_ion_params
        # Base functions are polynomials and hion_base are coefficients c0, c1, c2,...

        # creating base functions
        total_ion_base_functions = np.array([np.polynomial.Polynomial(matrix)
                                             for matrix in hion_base])

        # creating a function that describes total ion profile
        total_ion_profile_fun = np.dot(total_ion_params, total_ion_base_functions)
        return total_ion_profile_fun

    def _total_ion_params_from_lcrv_params(self):
        '''
        Calculates ionization rate coeffiecents that allows to reconstruct ionization rate profile in Legandre base
        '''
        # Read model data from json file

        lcrv_Trmatrix = np.array(self.model['vg_points']['parameters']['lcrv_Trmatrix'])
        model_const = np.array(self.model['vg_points']['parameters']['model_const'])
        model_matrix = np.array(self.model['vg_points']['parameters']['model_matrix'])
        hion_const = np.array(self.model['params_hion']['hion_const'])
        lcrv_filtration = np.array(self.model['vg_points']['parameters']['lcrv_filtration'])

        # Transformation PCA (we are taking only the first few components of the light curve decomposition)
        lcrv_params_PCA = np.dot(self.lcrv['decomposition_coeff'], lcrv_Trmatrix)[:lcrv_filtration]
        # Transformation from light curve parameters into total ion parameters
        total_ion_params = np.concatenate((model_const + np.dot(model_matrix, lcrv_params_PCA)
                                           , hion_const))
        self.ion['decomposition_coeff'] = total_ion_params
        return total_ion_params
