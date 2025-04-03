import numpy as np
from astropy.time import Time
import json
import netCDF4
import toolkit.funcs as fun
from .constants import VERSION
from dataclasses import dataclass


@dataclass
class SolarParamsHistory():
    '''
    Class for a Ionization files generator

    Methods
    -------
    
    '''

    def __init__(self,anc_input_from_instr_team,ext_dependencies):
        """
        Class constructor
        """
        # read pipeline settings to be used for L3d
        self.settings = fun.read_json(anc_input_from_instr_team['pipeline_settings'])

        self.header={
            'software_version': VERSION,
            'ancillary_data_files': anc_input_from_instr_team['WawHelioIon'],
            'external_dependeciens': ext_dependencies['lya_raw_data'],
            'l3b_input_filename': None,
            'l3c_input_filename': None
        }

        label_list=self.settings['initial_solar_params_label_list']
        self.ini_data={}
        self.ini_data['label']=label_list
        self.ini_data['time']=len(label_list)*[None]
        self.ini_data['data']=len(label_list)*[None]
        self.ini_data['txt_header']=len(label_list)*[None]

        self.lat_grid=self.settings['solar_wind_grid']      # latitudinal grid [deg]
        self.CR_grid=None                                   # regular grid with Carrington rotation period resolution. Bin centered on Carr halves
        self.time_grid=None                                 # decimal year corresponding to the regular grid with bins on Carrington halves
        self.CR_last=None

        self.solar_params={}                                                # ionization parameters interplolated to the Carrington rotation halves
        
        self.solar_params['speed']=None                                     # plasma speed history. 2D array CR_grid vs lat_grid [km/s]
        self.solar_params['density']=None                                   # proton density history. 2D array CR_grid vs lat_grid [#/cm^3]
        self.solar_params['uv_anisotropy']=None                             # UV anisotropy factor history. 2D array
        self.solar_params['phion']=None                                     # photoionization rate history in the ecliptic plane  [1/s]
        self.solar_params['lya']=None                                       # total solar irradiance in Lyman-alpha line history in the ecliptic plane [ph/cm^2/s]
        self.solar_params['electrons']=None                                 # electron density history in the ecliptic plane [#/cm^3]
    
    
    def generate_initial_history(self,anc_input_from_instr_team):
        gap_marker=-1

        # poprawic check, bo siatka z dziurami przeszla
        self._read_files(anc_input_from_instr_team)
        self._check_ini_data()

        # Find file that has the longest history
        N_list=[len(t) for t in self.ini_data['time']]
        k=np.array(N_list).argmax()
        
        # Time grid is taken from the file with the logest history
        self.time_grid=self.ini_data['time'][k]
        self.CR_grid=[np.round(fun.carrington(t.jd),2) for t in self.time_grid]
        self.CR_last=self.CR_grid[-1]
        # Dimensions of the ionization parameters on common grid
        Nt=len(self.time_grid)
        Nl=len(self.lat_grid)

        # Initialization of the ionization parameters on common grid.
        # At first all values are set at -1 
        self.solar_params['speed']=np.ones((Nt,Nl))*gap_marker
        self.solar_params['density']=np.ones((Nt,Nl))*gap_marker
        self.solar_params['uv_anisotropy']=np.ones((Nt,Nl))*gap_marker
        self.solar_params['phion']=np.ones(Nt)*gap_marker
        self.solar_params['lya']=np.ones(Nt)*gap_marker
        self.solar_params['electrons']=np.ones(Nt)*gap_marker

        # Indexes where actual data starts
       
        idx=len(self.ini_data['label'])*['']
        for i in np.arange(len(self.ini_data['label'])):
            idx[i]=np.abs(self.time_grid-self.ini_data['time'][i][0]).argmin()
        
        cr=fun.carrington(self.ini_data['time'][5].jd)
        diff=np.array([cr[i+1]-cr[i] for i in np.arange(len(cr)-1)])
        idx_zle=np.where(diff>1.5)

        # print(self.ini_data['time'][0])
        self.solar_params['speed'][idx[0]:]=self.ini_data['data'][0]
        self.solar_params['density'][idx[1]:]=self.ini_data['data'][1]
        self.solar_params['uv_anisotropy'][idx[2]:]=self.ini_data['data'][2]
        self.solar_params['phion'][idx[3]:]=self.ini_data['data'][3]
        self.solar_params['lya'][idx[4]:]=self.ini_data['data'][4]
        self.solar_params['electrons'][idx[5]:]=self.ini_data['data'][5]  

    def _check_ini_data(self):
        '''
        Check if initial data ends at the same CR
        '''
        # Resolution of the file should be Carrington rotation


        # Last CR in every file should be the same
        eps=0.001
        diff=np.array([ini_time[-1]-self.ini_data['time'][0][-1] for ini_time in self.ini_data['time']])
        if (diff>eps).any(): 
            raise Exception('Incorect format of the input files. Last CR is not the same in all files')
        else: return True
    
    def _read_ecliptic(self,fn):
        data=np.loadtxt(fn)
        # time, value

        t=Time(data[:,0],format='decimalyear')

        return t, data[:,1]
    
    def _read_txt_header(self,filename):

        with open(path_to_file) as input_file:
            head = [next(input_file) for _ in range(lines_number)]



    def _read_files(self,anc_input_from_instr_team):
        # Read ionization files
        # Profiles (speed, dens and anisotropy)
        fn_list=list(anc_input_from_instr_team['WawHelioIon'].values())
        for i in np.arange(3):
            self.ini_data['time'][i], self.ini_data['data'][i]=self._read_profile(fn_list[i])

        # Ecliptic values (lya, electrons)
        for i in np.arange(3,len(fn_list)):
            self.ini_data['time'][i], self.ini_data['data'][i]=self._read_ecliptic(fn_list[i])

    def _read_profile(self,fn):
        #TODO: add last column as CR (it needs to be consulted with MK)
        data=np.loadtxt(fn)
        # time, value(-90), value(-80), ... value(0), ..., value(90), CR

        t=Time(data[:,0],format='decimalyear')

        return t, data[:,1:-1]


    def save_to_file(self,fn):
        '''
        Solar params history file is an official GLOWS L3d data product
        Temporarly it is in json format, but in production code it should be in CDF

        Parameters:
        ------------
        fn: str
            name of the output file
        '''

        
        output={}
        output['header']=self.header
        output['lat_grid']=self.lat_grid
        output['cr_grid']=self.CR_grid.tolist()
        output['time_grid']=[t.iso for t in self.time_grid]
        output['solar_params']={}
        output['solar_params']['plasma_speed']=self.solar_params['speed'].tolist()
        output['solar_params']['proton_density']=self.solar_params['density'].tolist()
        output['solar_params']['uv_anisotropy']=self.solar_params['uv_anisotropy'].tolist()
        output['solar_params']['photoion']=self.solar_params['phion'].tolist()
        output['solar_params']['lya']=self.solar_params['lya'].tolist()
        output['solar_params']['electron_abundance']=self.solar_params['electrons'].tolist()


        json_content = json.dumps(output, indent=3)
        
        output_fp = open(fn, 'w')
        print(json_content, file=output_fp)
        output_fp.close()

    def update_solar_params_hist(self,ext_dependencies,data_l3b,data_l3c):
        CR=self.CR_last+1

        # update CR grid
        self.CR_grid = np.append(self.CR_grid, CR)

        # update time grid
        t_CR = Time(fun.jd_fm_Carrington(CR),format='jd')
        self.time_grid = np.append(self.time_grid, t_CR)

        # update Lyman-alpha data
        self._update_lya_data(ext_dependencies,CR)

        # upade rest of the parameters
        self._update_l3bc_data(data_l3b,data_l3c,CR)

    def _update_lya_data(self,ext_dependencies,CR):

        data_lya = fun.read_lya_data(ext_dependencies['lya_raw_data'])
        lya_CR = fun.generate_cr_lya(CR, data_lya)
        self.solar_params['lya'] = np.append(self.solar_params['lya'], lya_CR)

    def _update_l3bc_data(self,data_l3b,data_l3c,CR):
        anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read = fun.generate_cr_solar_params(CR, data_l3b, data_l3c)

        self.header['l3b_input_filename']=[data_l3b[i]['header']['filename'] for i in idx_read]
        self.header['l3c_input_filename']=[data_l3c[i]['header']['filename'] for i in idx_read]

        self.solar_params['phion'] = np.append(self.solar_params['phion'], ph_ion_CR)
        self.solar_params['electrons'] = np.append(self.solar_params['electrons'], e_dens_CR)

        self.solar_params['speed'] = np.r_[self.solar_params['speed'],[sw_speed_CR]]
        self.solar_params['density'] = np.r_[self.solar_params['density'],[p_dens_CR]]
        self.solar_params['uv_anisotropy'] = np.r_[self.solar_params['uv_anisotropy'],[anisotropy_CR]]