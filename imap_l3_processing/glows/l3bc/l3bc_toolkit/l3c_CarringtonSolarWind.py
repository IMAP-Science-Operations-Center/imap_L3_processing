'''
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Carrington solar wind class
'''

import json
from dataclasses import dataclass
import numpy as np
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
import toolkit.funcs as fun
from .constants import VERSION, PHISICAL_CONSTANTS

import logging
logging.basicConfig(level=logging.ERROR)

@dataclass
class CarringtonSolarWind():
    '''
    Class for a GLOWS Carrington averaged solar wind speed and density

    Attributes
    ----------
    header : dict
        software_version : float
            L3bc software version
        filename : str
            current file name
        ancillary_files : str
            list of ancillary files used in calculations
        external_dependeciens : str
            list of external dependeciens files
        l3b_input_filename: str
            Input L3b ionization rate profiles file
    settings : dict
        pipeline settings
    cx_profile : dict
        rate : float 1D vector
            L3b charge exchange rate profile value [#/s]
        uncert : float 1D vector
            L3b charge exchange rate uncertainty [#/s]
    sw_ecliptic : dict
        mean_speed : float
            mean value of the solar wind speed in the ecliptic plane [km/s]
        mean_proton_density : float
            mean value of the proton density in the ecliptic plane [#/cm^3]
        mean_alpha_abundance : float
            mean value of the alpha particle to proton ratio in the ecliptic plane [#/cm^3]
        invariant : float
            solar wind energy flux (it is used as a latitudnial invariant)
    sw_profile : dict
        date : astropy.time
            date of the profile as a astropy.time object
        CR : float
            Carrington rotation number
        grid : float 1D vector
            latitudinal grid [deg]
        plasma_speed : float 1D vector
            solar wind speed latitudinal profile [km/s]
        proton_density : float 1D vector 
            solar wind density latitudinal profile [#/cm^3]
        
    Methods
    -------   
    calculate_sw_profile(self,ext_dependencies)
        main procedure to calculate solar wwind speed and density by solving set of 2 equations
    read_l3b(self,filename)
        reads L3b data product file
    save_to_file(self,fn)
        saves L3c data product to the file
    '''

    def __init__(self, anc_input_from_instr_team):
        """
        Class constructor
        """

        self.header={
            'software_version': VERSION,
            'filename' : None,
            'ancillary_data_files': anc_input_from_instr_team,
            'external_dependeciens': None,
            'l3b_input_filename': None
        }

        # read pipeline settings to be used for L3b
        self.settings = fun.read_json(anc_input_from_instr_team['pipeline_settings'])
    
        # All parameters will be read from L3b file or computed later on
        self.cx_profile={}
        self.cx_profile['rate']=None
        self.cx_profile['uncert']=None

        self.sw_ecliptic={}
        self.sw_ecliptic['mean_speed']=None
        self.sw_ecliptic['mean_proton_density']=None
        self.sw_ecliptic['mean_alpha_abundance']=None
        self.sw_ecliptic['invariant']=None

        self.sw_profile={}
        self.sw_profile['date']=None
        self.sw_profile['CR']=None
        self.sw_profile['grid']=self.settings['solar_wind_grid']
        self.sw_profile['plasma_speed']=None
        self.sw_profile['proton_density']=None
        

    def calculate_sw_profile(self):
        '''
        Main procedure to calculate solar wind parameters profiles.
        Sets of 2 equations is numerically solved.
        For detailed disscussion see Documentation Section 13.7

        
        '''

        # initialization of the vectors that will be calculated later on
        # solar wind speed latitudinal profile   
        plasma_speed_profile=np.empty(len(self.settings['solar_wind_grid']))
        # solar wind density latitudinal profile   
        proton_dens_profile=np.empty(len(self.settings['solar_wind_grid']))
        
        
        invariant=self.sw_ecliptic['invariant']
        p_alpha=self.sw_ecliptic['mean_alpha_abundance']

        v_grid=np.array(self.settings['plasma_speed_numerical_grid'])*u.km/u.s   # numerical grid to solve equations [km/s]
        cs_grid=fun.cross_section_cx(fun.v2E(v_grid))             # cross section for charge exchange for speed grid [cm^2]
        rho_grid=invariant/(const.m_p+p_alpha*PHISICAL_CONSTANTS['m_alpha'])/(0.5*v_grid**2+const.GM_sun/const.R_sun)

        # calculation of the charge exchange rate for each speed in speed grid
        cx_grid=(rho_grid*cs_grid).to('1/s')
    
        for i in np.arange(len(plasma_speed_profile)):
            # charge exchane rate obtained from lcrv
            cx_obs=self.cx_profile['rate'][i]

            # Searching for a measured charge exchange in the cx_grid. In that way we will find sw speed
            idx=np.abs(cx_grid.value-cx_obs).argmin()
            plasma_speed_profile[i]=v_grid[idx].to('km.s-1').value
            proton_dens_profile[i]=(rho_grid[idx]/v_grid[idx]).to('cm-3').value
        self.sw_profile['plasma_speed']=plasma_speed_profile
        self.sw_profile['proton_density']=proton_dens_profile



    def read_l3b(self,filename):
        '''
        Read charge exchange rate profile from the L3b data product file

        Parameters
        -----------
        filename : str
            Path to the L3b data product file
        '''
        # Open and load json file with L3b data
        fp=open(filename)
        data_l3b=json.load(fp)
        fp.close()
    
        self.sw_profile['date']=Time(data_l3b['date'])    # do I really need it in astropy.time format?
        self.sw_profile['CR']=data_l3b['CR']
        self.cx_profile['rate']=data_l3b['ion_rate_profile']['cx_rate']
        self.cx_profile['uncert']=data_l3b['ion_rate_profile']['cx_uncert']

        self.header['l3b_input_filename']=filename

   
    def save_to_file(self,fn):
        '''
        Carrington solar wind plasma speed and proton densite profiles file is an official GLOWS L3c data product
        Temporarly it is in json format, but in production code it should be in CDF

        Parameters:
        ------------
        fn: str
            name of the output file 
        '''

        # Dictionary with the solar wind speed and density profiles  that will be saved in json file
        output={}
        output['header']=self.header
        output['date']=self.sw_profile['date'].iso
        output['CR']=self.sw_profile['CR']
        output['solar_wind_ecliptic']={}
        output['solar_wind_ecliptic']['plasma_speed']=self.sw_ecliptic['mean_speed']
        output['solar_wind_ecliptic']['proton_density']=self.sw_ecliptic['mean_proton_density']
        output['solar_wind_ecliptic']['alpha_abundance']=self.sw_ecliptic['mean_alpha_abundance']
        output['solar_wind_profile']={}
        output['solar_wind_profile']['lat_grid']=self.sw_profile['grid']
        output['solar_wind_profile']['plasma_speed']=self.sw_profile['plasma_speed'].tolist()
        output['solar_wind_profile']['proton_density']=self.sw_profile['proton_density'].tolist()

        json_content = json.dumps(output, indent=3)
        
        output_fp = open(fn, 'w')
        print(json_content, file=output_fp)
        output_fp.close()
