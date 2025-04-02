'''
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Solar wind parameters data in the ecliptic plane
'''

from dataclasses import dataclass
import numpy as np
from astropy.time import Time
from astropy import units as u
import toolkit.funcs as fun

import logging
logging.basicConfig(level=logging.ERROR)

@dataclass
class EclipticSolarWind():
    '''
    Class for a GLOWS Carrington averaged solar wind speed and density

    Attributes
    ----------
    settings : dict
        pipeline settings
    external_dependeciens : list
        list of external files
    CR_grid : float 1D vector
        list of available CR numbers
    CR: int
        currently analized Carrington Rotation period
    mean_speed : float 1D vector
        mean value of the solar wind speed in the ecliptic plane for each CR [km/s]
    mean_proton_density :  float 1D vector
        mean value of the proton density in the ecliptic plane for each CR [#/cm^3]
    alpha_abundance :  float 1D vector
        mean value of the alpha particle to proton ratio in the ecliptic plane for each CR [#/cm^3]
        
    Methods
    -------
    calculate_invariant(self,ext_dependencies)
        calculates latitudinal invariant as a long term average of the solar wind energy flux
    _read_OMNI_data(self,ext_dependencies)
        reads solar wind parameters from OMNI2 database
    _read_SWAPI_data(self, ext_dependencies)
        reads solar wind parameters from SWAPI
    read_sw_ecliptic(self,ext_dependencies)
        reads solar wind parameters in the ecliptic plane (source of the data is set in pipeline settings)

    '''

    def __init__(self, anc_input_from_instr_team, CR):
        """
        Class constructor
        """

        # read pipeline settings to be used
        self.settings = fun.read_json(anc_input_from_instr_team['pipeline_settings'])

        self.external_dependeciens=None
        # All parameters will be read from L3b file or computed later on
        self.CR=CR        # current CR number
        self.CR_grid=None
        self.mean_speed=None
        self.mean_proton_density=None
        self.mean_alpha_abundance=None
        self.invariant=None

    def calculate_invariant(self,ext_dependencies):
        '''
        Solar wind energy flux was found to be latitudinal invariant (by Ulysses mission).
        Using OMNI, then SWAPI data daily invariant will be calculated and then averaged over Carrington rotation period
        We need long averages (~13 Carrington rotation periods)

        Parameters:
        ------------
        ext_dependencies : dict
            list of file names with external dependencies set in constants.py

        Output:
        ------------
        invariant : astropy.units
            averaged solar wind energy flux as a object with unists [erg/s/cm^2]
        '''

        #TODO: SWAPI if we have full 13 Carrington rotation periods of data

        # Read solar wind parameters (Carrington averaged parameters on common carrington number grid)
        grid_cr, plasma_speed, proton_dens, p_alpha = self.read_sw_ecliptic(ext_dependencies)

        # Calculate solar wind energy flux for each CR
        sw_energy_flux=fun.calculate_sw_energy_flux(proton_dens/(u.cm)**3,plasma_speed*u.km/u.s,p_alpha)

        N_Carr=self.settings['invariant_average_Carr_number']  # number of Carringtons that we want to average over
        
        self.invariant=sw_energy_flux[-N_Carr:].mean()


    def _read_OMNI_data(self,ext_dependencies):
        '''
        Reads 3 types of data from OMNI2 database: solar wind speed, density and alpha-particles abundance in the ecliptic plane
        
        Parameters:
        ------------
        ext_dependencies : dict
            list of file names with external dependencies
        '''
        # Read OMNI2 data (hour resolution) from last few CR (exact number is specified in pipeline settings)
        cr_n=self.settings['invariant_average_Carr_number']
        cr_ini=self.CR-cr_n
        t_window=[Time(fun.jd_fm_Carrington(cr_ini),format='jd'),Time(fun.jd_fm_Carrington(self.CR+1),format='jd')]

        omni_raw=fun.read_raw_OMNI_data(ext_dependencies,t_window)
        
        param_settings={'density': {'column_numbers': (0,1,2,4,7), 'gap_marker': 999.9, 'scale': True},
                        'speed': {'column_numbers': (0,1,2,5,8), 'gap_marker': 9999, 'scale': False},
                        'alpha': {'column_numbers': (0,1,2,6,9), 'gap_marker': 9.999, 'scale': False}
        }
        
        proton_dens_cr, proton_dens_carr=fun.process_omni_param(omni_raw,param_settings['density'])
        plasma_speed_cr, plasma_speed_carr=fun.process_omni_param(omni_raw,param_settings['speed'])
        p_alpha_cr, p_alpha_carr=fun.process_omni_param(omni_raw,param_settings['alpha'])
    
        self.external_dependeciens=[ext_dependencies['omni_raw_data']]
        self.CR_grid=proton_dens_cr
        self.mean_speed=plasma_speed_carr
        self.mean_proton_density=proton_dens_carr
        self.mean_alpha_abundance=p_alpha_carr

        return proton_dens_cr, proton_dens_carr, plasma_speed_carr, p_alpha_carr


    def _read_SWAPI_data(self, ext_dependencies):
        '''
        Placeholder

        Parameters:
        ---------------------
        ext_dependencies : dict
            list of file names with external dependencies

        Output:
        ----------------------
        proton_dens_cr : float 1D vector
            list of available CR 
        proton_dens_carr : float 1D vector
            Carrington-period averaged proton densites adjusted to 1au
        plasma_speed_carr : float 1D vector
            Carrington-period averaged plasma speed
        p_alpha_carr
            Carrington-period averaged alpha-particles abundance
        '''


    def read_sw_ecliptic(self,ext_dependencies):
        '''
        Calls procedure to read solar wind data in the ecliptic plane.
        It can be either from OMNI2 database or IMAP/SWAPI
        The source will be given in the pipeline settings file provided by the instrument team


        Parameters:
        ------------
        ext_dependencies : dict
            list of file names with external dependencies

        Output:
        ------------
        grid : float 1D vector
            CR grid common for all solar wind parameters [Carrington rotation period resolution]
        plasma_speed : float 1D vector
            plasma  speed data in the ecliptic plane averaged over Carrington rotation period [km/s]
        proton_dens : float 1D vector
            proton density data in the ecliptic plane averaged over Carrington rotation period [#/cm^3]
        p_alpha : float 1D vector
            alpha particle abundance wrt protons data in the ecliptic plane averaged over Carrington rotation period
        
        '''
        sw_ecliptic_source = self.settings['sw_ecliptic_source']
        
        if sw_ecliptic_source == 'OMNI': grid_cr, plasma_speed, proton_dens, p_alpha = self._read_OMNI_data(ext_dependencies)
        elif sw_ecliptic_source == 'SWAPI': grid_cr, plasma_speed, proton_dens, p_alpha = self._read_SWAPI_data(ext_dependencies)
        else: raise Exception('Unknown solar wind source. Select OMNI or SWAPI')

        return  grid_cr, plasma_speed, proton_dens, p_alpha

