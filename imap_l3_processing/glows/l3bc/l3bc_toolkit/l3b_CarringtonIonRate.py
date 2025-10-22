'''
Author: Izabela Kowalska-Leszczynska (ikowalska@cbk.waw.pl)
Carrington averaged ionization rate class
'''

import json
import numpy as np
from dataclasses import dataclass
from astropy.time import Time
from astropy import units as u
from .constants import VERSION
from . import funcs as fun

import logging
logging.basicConfig(level=logging.ERROR)

@dataclass
class CarringtonIonizationRate():
    '''
    Class for a GLOWS Carrington averaged ionization rate profiles
    Contains original daily ionization rate profiles, averaged ionization rate profile
    charge exchange and photoionization profiles

    Attributes
    ----------
    header: dict
        software_version : float
            L3b-d software version
        filename : str
            current file name
        ancillary_files : str
            list of ancillary files used in calculations
        external_dependeciens : str
            list of external dependeciens files
        l3a_input_filename: str
            Input L3a lightcurve filename
    settings: dict
        pipeline settings
    sw_ecliptic: dict
        averaged solar wind parameters in the ecliptic plane
        mean_speed: avearaged solar wind plasma speed [km/s]
        mean_proton_density: averaged solar wind proton density [#/cm^3]
    daily_ion_rate: dict
        daily ionization rate profiles (only those not excluded by the bad-days-list)
        date : astropy.time vector
            list of dates of the daily ionization rates profiles
        ion_rate: float 1D vector
            ionization rate value [#/s]
    carr_ion_rate: dict
        ionization rate profile averaged over Carrington rotation period
        date: astropy.time
            date set at the middle of Carrington rotation period
        CR: float
            number of the Carrington rotation
        ion_grid: float 1D vector
            latitudinal grid
        ion_rate: float 1D vector
            ionization rate value [#/s]
        cx_rate: float 1D vector
            charge exchange rate value [#/s]
        ph_rate: float 1D vector
            photoionization rate value [#/s]
        ion_rate_uncert: float 1D vector
            ionization rate uncertainty [#/s]
        cx_rate_uncert: float 1D vector
            charge exchange rate uncertainty [#/s]
        ph_rate_uncert: float 1D vector
            photoionization rate uncertainty [#/s]
    uv_anisotropy: float 1D vector
        latitudinal profile of the UV anisotropy

    Methods
    -------
    _anchor_to_ecliptic(self,carr_ion_rate)
        normalizes the profile to match solar wind parameters measured in the ecliptic plane
    calculate_averaged_profile(self)
        calculates average value from the daily ionization profiles (bin-by-bin)
    calculate_charge_exchange(self)
        calculates averaged charge exchane rate by subtracting photoionization profile from the ionization rate profile obrain from the lightcurve
    calculate_photoion(self)
        calculates averaged photoionization rate profile using daily photoionization rate based on F10.7
    _calculate_photoion_from_f107(self,flux)
        calculates daily photoionization rate in the ecliptic plane from F10.7 radio flux measurements
    _read_uv_anisotrpy(self,fn)
        Read file with UV latitudinal anisotrpy factor averaged over 1 Carrington rotation period
    save_to_file(self,fn)
        saves L3b structure to file
    '''

    def __init__(self, anc_input_from_instr_team,ext_dependencies):
        """
        Class constructor
        """
        self.header={
            'software_version': VERSION,
            'filename': None,
            'ancillary_data_files': anc_input_from_instr_team,
            'external_dependeciens': ext_dependencies['f107_raw_data'],
        }

        # read pipeline settings to be used for L3b
        self.settings = fun.read_json(anc_input_from_instr_team['pipeline_settings'])

        self.sw_ecliptic={}
        self.sw_ecliptic['mean_speed']=None
        self.sw_ecliptic['mean_proton_density']=None

        # All parameters will be read from L3a file or computed later on
        self.daily_ion_rate={}
        self.daily_ion_rate['date']=None
        self.daily_ion_rate['ion_rate']=None

        self.carr_ion_rate={}
        self.carr_ion_rate['date']=None
        self.carr_ion_rate['CR']=None
        self.carr_ion_rate['ion_grid']=self.settings['ion_rate_grid']
        self.carr_ion_rate['ion_rate']=None
        self.carr_ion_rate['cx_rate']=None
        self.carr_ion_rate['ph_rate']=None
        self.carr_ion_rate['ion_rate_uncert']=None
        self.carr_ion_rate['cx_rate_uncert']=None
        self.carr_ion_rate['ph_rate_uncert']=None

        self.uv_anisotropy=None
        self.uv_anisotropy_flag=None

    def _anchor_to_ecliptic(self,carr_ion_rate):
        '''
        normalizes the profile in such a way that the value in its central bin
        matches the average ionization rate calculated based on the measured parameters
        of the solar wind in the ecliptic plane

        Parameters:
        ---------------
        carr_ion_rate: mean ionization rate profile calculated by taking average value of all daily ionization rate profiles within considered CR

        '''
        idx_middle=int(len(self.carr_ion_rate['ion_grid'])/2)    # middle bin (equator) in the latitudinal grid
        carr_ion_rate0=carr_ion_rate[idx_middle]

        ph_rate0=self.carr_ion_rate['ph_rate'][idx_middle]    # ph_rate at latitude=0
        v0=self.sw_ecliptic['mean_speed']*u.km/u.s
        n0=self.sw_ecliptic['mean_proton_density']/(u.cm)**3
        sigma0=fun.cross_section_cx(fun.v2E(v0))

        cx_rate0=(v0*sigma0*n0).to('1/s').value

        ion_rate0=ph_rate0+cx_rate0
        self.carr_ion_rate['ion_rate']=carr_ion_rate/carr_ion_rate0*ion_rate0


    def calculate_averaged_profile(self):
        '''
        Mean value of the ionization profile and time. Average taken bin-by-bin on the latitudinal grid.
        '''

        carr_ion_rate=np.mean(self.daily_ion_rate['ion_rate'],axis=0)

        # Anchore averaged ionization rate profile to the solar wind measurements in the ecliptic plane
        self._anchor_to_ecliptic(carr_ion_rate)
        self.carr_ion_rate['date']=np.mean(self.daily_ion_rate['date'])

        # Statistical uncertinity
        if len(self.daily_ion_rate['ion_rate'])>1:
            self.carr_ion_rate['ion_rate_uncert']=np.std(self.daily_ion_rate['ion_rate'],axis=0,ddof=1)
        else: self.carr_ion_rate['ion_rate_uncert']=1.0E31*np.ones_like(self.carr_ion_rate['ion_rate'])



    def calculate_charge_exchange(self):
        '''
        Charge exchange rate for a stationary atom aproximation assuming that total ionization rate is a sum
        of photoionization and charge exchange rate.
        '''

        # ionization rate obtained from the light curve is a sum of charge exchange rate and photoionization rate
        self.carr_ion_rate['cx_rate']=self.carr_ion_rate['ion_rate']-self.carr_ion_rate['ph_rate']

        # statistical uncertainty
        if len(self.carr_ion_rate['cx_rate'])>1:
            self.carr_ion_rate['cx_rate_uncert']=self.carr_ion_rate['ion_rate_uncert']+self.carr_ion_rate['ph_rate_uncert']
        else:
            self.carr_ion_rate['cx_rate_uncert']=1.0E31*np.ones_like(self.carr_ion_rate['cx_rate'])

    def calculate_photoion(self, anc_input_from_instr_team,ext_dependencies):
        '''
        Photoionization rate at 1au averaged over a time window
        Based on daily photoionization calculated from F10.7 using WawHelioIon method
        '''

        # raw data from the same period as available daily profiles
        t_ini=self.daily_ion_rate['date'][0]
        t_end=self.daily_ion_rate['date'][-1]
        t_mid=t_ini+(t_end-t_ini)/2

        # read F10.7 file (the raw data from LASP)
        t_raw,flux_raw=fun.read_f107_raw_data(ext_dependencies['f107_raw_data'])
        t_daily, flux_daily0=fun.f107_daily_data(t_raw,flux_raw) # zmiana

        # select data from the window
        #data_window=fun.select_data_window([t_raw,flux_raw],[t_ini,t_end])  zmiana
        data_window=fun.select_data_window([t_daily,flux_daily0],[t_ini,t_end])

        # check if there are at least 2 data points in the window
        photoion_daily=[]

        if len(data_window[0])>2:
            flux_daily=data_window[1]
            # apply correlation between photoion and F10.7
            photoion_daily=self._calculate_photoion_from_f107(flux_daily)

            # average daily photoionization
            photoion_mean=photoion_daily.mean()
            photoion_std=np.std(photoion_daily,ddof=1)

        else:
            # if there are less than 2 data points in the time window, we want to interpolate
            # between mean values of the previous and next CR
            t_ini_next=Time(fun.jd_fm_Carrington(self.carr_ion_rate['CR']+1),format='jd')
            t_end_next=Time(fun.jd_fm_Carrington(self.carr_ion_rate['CR']+2),format='jd')
            data_window_next=fun.select_data_window([t_raw,flux_raw],[t_ini_next,t_end_next])
            if len(data_window_next[0])>2:
                # if there are at least data points in the next CR, then we can interpolate
                t_ini_prev=Time(fun.jd_fm_Carrington(self.carr_ion_rate['CR']-1),format='jd')
                t_end_prev=Time(fun.jd_fm_Carrington(self.carr_ion_rate['CR']),format='jd')
                data_window_prev=fun.select_data_window([t_raw,flux_raw],[t_ini_prev,t_end_prev])

                data_next_mean=[data_window_next[0].mean(),data_window_next[1].mean()]
                data_prev_mean=[data_window_prev[0].mean(),data_window_prev[1].mean()]

                data_mean=[t_mid,np.interp(t_mid.jd,np.array([data_next_mean[0].jd,data_prev_mean[0].jd]),np.array([data_next_mean[1],data_prev_mean[1]]))]

                photoion_mean=self._calculate_photoion_from_f107(data_mean[1])
            else:
                # if there are no data in the next CR, then we need to wait for another CR
                raise Exception("There are no data to interpolate F10.7 for CR=%d",self.carr_ion_rate['CR'])


        # including latitudinal anisotropy factor
        self._read_uv_anisotrpy(anc_input_from_instr_team['uv_anisotropy'])
        self.carr_ion_rate['ph_rate']=photoion_mean*self.uv_anisotropy

        if len(photoion_daily)>1:
            self.carr_ion_rate['ph_rate_uncert']=photoion_std*np.ones_like(self.carr_ion_rate['ph_rate'])
        else:
            self.carr_ion_rate['ph_rate_uncert']=1.0E31*np.ones_like(self.carr_ion_rate['ph_rate'])

    def _calculate_photoion_from_f107(self,flux):
        '''
        Correlation between photoionization rate and F10.7 (sokol_bzowski:14a, sokol_etal:20a)

        Parameters:
        -----------
        flux: F10.7 adjusted flux in sfu 1sfu=10^(-22) W/m^2/Hz

        Output:
        ------------
        photoion: the value of the photoionization at 1au on ecliptic calculated from F10.7

        '''
        photoion=-2.9819e-8+2.416e-8*flux**0.4017
        return photoion

    def _read_uv_anisotrpy(self,fn):
        '''
        Read file with UV latitudinal anisotrpy factor averaged over 1 Carrington rotation period.
        File will be delivered by the instrument team

        Parameters:
        --------------
        fn:    file name (str)
        '''

        anisotropy = fun.read_json(fn)

        self.uv_anisotropy = np.array(anisotropy['anisotropy_factor'])
        self.uv_anisotropy_flag = anisotropy['flag']

    def get_dict(self):

        # Dictionary with the ionization rate profiles (total, photoionization and charge exchange) that will be saved in json file
        output = {}
        output['header'] = self.header

        output['date'] = self.carr_ion_rate['date'].iso
        output['CR'] = int(self.carr_ion_rate['CR'])

        output['ion_rate_profile'] = {}
        output['ion_rate_profile']['lat_grid'] = self.carr_ion_rate['ion_grid']
        output['ion_rate_profile']['sum_rate'] = self.carr_ion_rate['ion_rate'].tolist()
        output['ion_rate_profile']['ph_rate'] = self.carr_ion_rate['ph_rate'].tolist()
        output['ion_rate_profile']['cx_rate'] = self.carr_ion_rate['cx_rate'].tolist()
        output['ion_rate_profile']['sum_uncert'] = self.carr_ion_rate['ion_rate_uncert'].tolist()
        output['ion_rate_profile']['ph_uncert'] = self.carr_ion_rate['ph_rate_uncert'].tolist()
        output['ion_rate_profile']['cx_uncert'] = self.carr_ion_rate['cx_rate_uncert'].tolist()

        output['uv_anisotropy_factor'] = self.uv_anisotropy.tolist()
        output['uv_anisotropy_flag'] = self.uv_anisotropy_flag

        return output

    def save_to_file(self,fn):
        '''
        Carrington ionization rate file is an official GLOWS L3b data product
        Temporarly it is in json format, but in production code it should be in CDF

        Parameters:
        ------------
        fn: str
            name of the output file

        '''

        # Dictionary with the ionization rate profiles (total, photoionization and charge exchange) that will be saved in json file
        output={}
        output['header']=self.header

        output['date']=self.carr_ion_rate['date'].iso
        output['CR']=int(self.carr_ion_rate['CR'])

        output['ion_rate_profile']={}
        output['ion_rate_profile']['lat_grid']=self.carr_ion_rate['ion_grid']
        output['ion_rate_profile']['sum_rate']=self.carr_ion_rate['ion_rate'].tolist()
        output['ion_rate_profile']['ph_rate']=self.carr_ion_rate['ph_rate'].tolist()
        output['ion_rate_profile']['cx_rate']=self.carr_ion_rate['cx_rate'].tolist()
        output['ion_rate_profile']['sum_uncert']=self.carr_ion_rate['ion_rate_uncert'].tolist()
        output['ion_rate_profile']['ph_uncert']=self.carr_ion_rate['ph_rate_uncert'].tolist()
        output['ion_rate_profile']['cx_uncert']=self.carr_ion_rate['cx_rate_uncert'].tolist()

        output['uv_anisotropy_factor']=self.uv_anisotropy.tolist()
        output['uv_anisotropy_flag'] = self.uv_anisotropy_flag

        # Check validity of the data
        fun.check_nan(output['ion_rate_profile'])

        json_content = json.dumps(output, indent=3)

        output_fp = open(fn, 'w')
        print(json_content, file=output_fp)
        output_fp.close()

