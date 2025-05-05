import numpy as np
from astropy.time import Time
import json
import glob
import toolkit.funcs as fun
from .constants import VERSION
from dataclasses import dataclass


@dataclass
class SolarParamsHistory():
    '''
    Class for an Ionization files generator
    Text files (solar wind speed, density, uv-anisotropy, Lyman-alpha irradiance, photoionization, and electron density) 
    are requaried as a input files for L3e processing
    Methods
    -------
    _check_ini_data(self) 
    find_fn_initial(self)
    _generate_cr_solar_params(self,CR, data_l3b, data_l3c)  
    generate_initial_history(self,fn_list)
    generate_hdr_txt_ini(self,fn_dict)
    generate_hdr_txt(self,fn_dict)
    _generate_hdr_txt_glows(self,hdr_temp,hdr_ini_str,fn_out)
    _generate_hdr_e_dens(self,hdr_temp,hdr_ini_str,fn_out)
    _generate_hdr_lya(self,hdr_temp,hdr_ini_str,fn_out)
    _generate_hdr_phion(self,hdr_temp,hdr_ini_str,fn_out)
    _generate_hdr_speed_dens(self,hdr_temp,hdr_ini_str,fn_out)
    _generate_hdr_uv_anis(self,hdr_temp,hdr_ini_str,fn_out)  
    _generate_hdr_template(self)
    _read_ecliptic(self,fn)
    _read_files(self,fn_list)
    _read_profile(self,fn)
    save_to_file(self,fn)
    save_to_txt(self,fn_out,hdr)
    _save_to_txt_ecliptic(self,fn,k,hdr)
    _save_to_txt_profile(self,fn,k,hdr)
    _update_l3bc_data(self,data_l3b,data_l3c,CR)
    _update_lya_data(self,ext_dependencies,CR)
    update_solar_params_hist(self,ext_dependencies,data_l3b,data_l3c)

    '''

    def __init__(self,anc_input_from_instr_team,ext_dependencies):
        """
        Class constructor

        Parameters
        -----------
        anc_input_from_instr_team: dict 
            Dictionary containing input from the instrument team.
        ext_dependencies: dict 
            Dictionary containing external dependencies (e.g., Lya raw data path).

        Attributes
        ----------
        header : dict
            software_version : float
                L3d software version
            filename : str
                current file name
            ancillary_files : str
                list of ancillary files used in calculations
            external_dependeciens : str
                list of external dependeciens files
            l3b_input_filename: str
                Input L3b ionization rate profiles file
            l3c_input_filename: str
                Input L3c solar wind speed and density profiles file
        settings : dict
            pipeline settings
        ini_data: dict
            initial historical data up to the current CR
            CR_last: float
                last CR point in the history (it will be halve of the CR)
            label: list
                list of keys to identify solar parameters (defined in const file)
            time: dict
                list of astropy.time objects with time grid of the solar parameters history (lenght can be different for different parameters)
            data: dict
                values of the solar parameters on the time grid
            hdr_txt: dict
                headers from the text files (30 lines)
        lat_grid: 1D float vector
            latitudinal grid from -90 to 90 deg
        CR_grid:
            Carrington rotation regular grid (halves of the integers)
        time_grid:
            UTC time corresponding to the CR_grid
        CR_last:
            last CR in the updated solar parameters history
        solar_params: dict
            values of the solar parameters for a current CR appended to the initial history

        Methods
        ------------



        """
        # read pipeline settings to be used for L3d
        self.settings = fun.read_json(anc_input_from_instr_team['pipeline_settings'])

        self.header={
            'software_version': VERSION,
            'filename': None,
            'ancillary_data_files': None,
            'external_dependeciens': ext_dependencies['lya_raw_data'],
            'l3b_input_filename': None,
            'l3c_input_filename': None
        }

        self.ini_data={}
        self.ini_data['CR_last']=None
        self.ini_data['label']=list(anc_input_from_instr_team['WawHelioIon'].keys())
        N=len(self.ini_data['label'])
        self.ini_data['time']={}
        for k in anc_input_from_instr_team['WawHelioIon']: self.ini_data['time'][k]=None
        self.ini_data['data']={}
        for k in anc_input_from_instr_team['WawHelioIon']: self.ini_data['data'][k]=None
        self.ini_data['hdr_txt']={}
        for k in anc_input_from_instr_team['WawHelioIon']: self.ini_data['hdr_txt'][k]=None

        self.lat_grid=self.settings['solar_wind_grid']      # latitudinal grid [deg]
        self.CR_grid=None                                   # regular grid with Carrington rotation period resolution. Bin centered on Carr halves
        self.time_grid=None                                 # decimal year corresponding to the regular grid with bins on Carrington halves
        self.CR_last=None

        self.solar_params={}                                                # ionization parameters interplolated to the Carrington rotation halves
        
        self.solar_params['speed']=None                                     # plasma speed history. 2D array CR_grid vs lat_grid [km/s]
        self.solar_params['p-dens']=None                                    # proton density history. 2D array CR_grid vs lat_grid [#/cm^3]
        self.solar_params['uv-anis']=None                                   # UV anisotropy factor history. 2D array
        self.solar_params['phion']=None                                     # photoionization rate history in the ecliptic plane  [1/s]
        self.solar_params['lya']=None                                       # total solar irradiance in Lyman-alpha line history in the ecliptic plane [ph/cm^2/s]
        self.solar_params['e-dens']=None                                    # electron density history in the ecliptic plane [#/cm^3]
    
    def _check_ini_data(self):
        '''
        Check if initial data ends at the same CR
        '''
        # Resolution of the file should be Carrington rotation

        # Last CR in every file should be the same
        eps=0.001
        diff=np.array([self.ini_data['time'][k][-1]-self.ini_data['time']['lya'][-1] for k in self.ini_data['time']])
        if (diff>eps).any(): 
            raise Exception('Incorrect format of the input files. Last CR is not the same in all files')
        else: return True
    
    def find_fn_initial(self):
        '''
        Finds text files from the previous CR that now are initial files for the current CR.
        For each type of the text file (speed, p-dens, etc.) a file with the highest CR is selected
        In the final version there should be additional option for version checking
        '''
        l3d_txt_fn_list={}
        for k in self.ini_data['label']: l3d_txt_fn_list[k]=np.array(sorted(glob.glob('data_l3d_txt/imap_glows_l3d*'+k+'*.dat')))
    
        # list of the L3d text files from the latest CR
        fn_initial={}
        for k in l3d_txt_fn_list:
            idx_latest=np.array([f.split('cr_')[1].split('_')[0] for f in l3d_txt_fn_list[k]]).argmax()
            fn_initial[k]=l3d_txt_fn_list[k][idx_latest]
        return fn_initial


    def _generate_cr_solar_params(self,CR, data_l3b, data_l3c):
        '''
        Calculates solar parameters (plasma speed, proton density, photoionization, electron density, and uv anisotropy)
        interpolated on the center of a given CR
        '''
        # find indexes of the closest to the current CR and the next CR where L3bc are available

        t_CR = Time(fun.jd_fm_Carrington(CR),format='jd')
    
        CR_list=[data['CR'] for data in data_l3b]
        idx_read = fun.find_CR_idx(CR, CR_list)

        # if there is only current CR, but not the next one available, then we can't interpolate
        if idx_read[-1]>=len(CR_list): raise Exception('L3d not generated: there is not enough L3bc data to interpolate')

        # mean time based on the light curves that were actually used during L3b processing (after removing bad-days and bad-seasons)
        t_b=Time(np.array([fun.calculate_mean_date(data_l3b[i]['header']['l3a_input_files_name']) for i in idx_read]))
        
        # time nodes that will be used for interpolation. The first one is the last time stamp from the previous CR, the second and the third are next available L3bc dates
        t_nods=np.concatenate([[fun.jd_fm_Carrington(self.ini_data['CR_last'])],t_b.jd])

        # UV-anisotropy
        anisotropy=np.concatenate([[self.ini_data['data']['uv-anis'][-1]],np.array([data_l3b[i]['uv_anisotropy_factor'] for i in idx_read])])
        anisotropy_CR=np.array([np.interp(t_CR.jd,t_nods,anisotropy[:,i]) for i in np.arange(len(anisotropy[0]))])

        # photoion in the ecliptic plane
        Nmid=int(len(data_l3b[0]['ion_rate_profile']['lat_grid'])/2)
        ph_b=np.concatenate([[self.ini_data['data']['phion'][-1]],np.array([data_l3b[i]['ion_rate_profile']['ph_rate'][Nmid] for i in idx_read])])
        ph_ion_CR=np.interp(t_CR.jd,t_nods,ph_b)

        # solar wind parameters (plasma speed, proton density)
        p_dens=np.concatenate([[self.ini_data['data']['p-dens'][-1]],np.array([data_l3c[i]['solar_wind_profile']['proton_density'] for i in idx_read])])
        sw_speed=np.concatenate([[self.ini_data['data']['speed'][-1]],np.array([data_l3c[i]['solar_wind_profile']['plasma_speed'] for i in idx_read])])
        p_dens_CR=np.array([np.interp(t_CR.jd,t_nods,p_dens[:,i]) for i in np.arange(len(p_dens[0]))])
        sw_speed_CR=np.array([np.interp(t_CR.jd,t_nods,sw_speed[:,i]) for i in np.arange(len(sw_speed[0]))])

        # electron density in the ecliptic plane calculated from p-dens and alpha-abundance
        p_dens_ecl=np.array([data_l3c[i]['solar_wind_ecliptic']['proton_density'] for i in idx_read])
        a_abundance_ecl=np.array([data_l3c[i]['solar_wind_ecliptic']['alpha_abundance'] for i in idx_read])
        e_dens_ecl=np.concatenate([[self.ini_data['data']['e-dens'][-1]],p_dens_ecl*(1+2*a_abundance_ecl)])
        e_dens_CR=np.interp(t_CR.jd,t_nods,e_dens_ecl)

        return anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read

    def generate_initial_history(self,fn_list):
        '''
        Reads text files with a time history. Usually it is the previous CR file
        Function fills ini_data structure that will be passed to the next text file
        '''
        fill_value=self.settings['WawHelioIonGlows_fill_value']
        
        self._read_files(fn_list)
        self._check_ini_data()

        # Find file that has the longest history
        k_max=max(self.ini_data['time'],key=lambda k: len(self.ini_data['time'][k]))
        
        # Time grid is taken from the file with the logest history
        self.time_grid=self.ini_data['time'][k_max]
        self.CR_grid=[np.round(fun.carrington(t.jd),2) for t in self.time_grid]
        self.ini_data['CR_last']=self.CR_grid[-1]
        
        # Dimensions of the ionization parameters on common grid
        Nt=len(self.time_grid)
        Nl=len(self.lat_grid)

        # Initialization of the ionization parameters on common grid.
        # At first all values are set at fill value defined in pipeline settings
        
        for f in fn_list:
            if np.logical_or(np.logical_or(f=='speed',f=='p-dens'),f=='uv-anis'):
                self.solar_params[f]=np.ones((Nt,Nl))*fill_value
            elif np.logical_or(np.logical_or(f=='lya',f=='phion'),f=='e-dens'):
                self.solar_params[f]=np.ones(Nt)*fill_value

        # Indexes where actual data starts
        idx={}
        for k,v in self.ini_data['time'].items(): idx[k]=np.abs(self.time_grid-v[0]).argmin()

        # solar_params structure is filled by values from the ini_data on the uniform time grid
        for f in fn_list: self.solar_params[f][idx[f]:]=self.ini_data['data'][f] 

    def generate_hdr_txt_ini(self,fn_dict):
        '''
        Generates text header for L3d text files from initial ancilliary files (this function will be called just once when there is no L3d data yet)
        '''
        hdr_dict={}
        for f in fn_dict:
            hdr=self._generate_hdr_template()
            if f=='phion': hdr_dict[f]=self._generate_hdr_phion(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
            elif f=='lya': hdr_dict[f]=self._generate_hdr_lya(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
            elif f=='e-dens': hdr_dict[f]=self._generate_hdr_e_dens(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
            elif f=='uv-anis': hdr_dict[f]=self._generate_hdr_uv_anis(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
            elif f=='speed': hdr_dict[f]=self._generate_hdr_speed_dens(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
            elif f=='p-dens': hdr_dict[f]=self._generate_hdr_speed_dens(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
        return hdr_dict
    
    def generate_hdr_txt(self,fn_dict):
        '''
        Generates text header for a text L3d files from prievious L3d text files
        '''
        hdr_dict={}
    
        for f in fn_dict:
            hdr=self._generate_hdr_template()
            hdr_dict[f]=self._generate_hdr_txt_glows(hdr, self.ini_data['hdr_txt'][f],fn_dict[f])
        return hdr_dict

    def _generate_hdr_txt_glows(self,hdr_temp,hdr_ini_str,fn_out):
        '''
        Copy some lines from the input heder to the output header structure.
        As a input there is unfied header structure
        '''

        hdr_temp[1]='filename: ' + fn_out+'\n'
        hdr_temp[2:4]=hdr_ini_str[2:4]
        hdr_temp[16:29]=hdr_ini_str[16:29]
        
        return hdr_temp

    def _generate_hdr_e_dens(self,hdr_temp,hdr_ini_str,fn_out):
        '''
        Reads and aranges lines from the initial electron density file into uniform structure of the header that will be used in L3d text files
        '''

        hdr_temp[1]='filename: ' + fn_out+'\n'
        hdr_temp[2]='initial file: '+ hdr_ini_str[2].split(' ')[-1][:-1] + ' / '+ ' '.join(hdr_ini_str[4].split(' ')[2:4])+' ('+ hdr_ini_str[5].split(':')[1][:-1]+') / '+hdr_ini_str[8].split(' ')[-1]
        hdr_temp[3]='description: '+hdr_ini_str[3]
        
        hdr_temp[23]='initial creation date:'+ ':'.join(hdr_ini_str[7].split(':')[1:])
        hdr_temp[24]='initial software: '+hdr_ini_str[6].split(' ')[2] +'\n'
        hdr_temp[25]='initial source files: '+hdr_ini_str[10].split(':')[-1]
        hdr_temp[28]=hdr_ini_str[28][:-1] + ' CR\n'
        
        
        return hdr_temp

    def _generate_hdr_lya(self,hdr_temp,hdr_ini_str,fn_out):
        '''
        Reads and aranges lines from the initial Lyman-alpha irradiance file into uniform structure of the header that will be used in L3d text files
        '''
        
        hdr_temp[1]='filename: ' + fn_out+'\n'
        hdr_temp[2]='initial file: '+ hdr_ini_str[1].split(' ')[1] + ' / '+ hdr_ini_str[2].split('/')[0].split(':')[1]+' ('+ hdr_ini_str[2].split('/')[2].split(' ')[1]+') / '+hdr_ini_str[5].split(' ')[2]+'\n'
        hdr_temp[3]='description: '+hdr_ini_str[8].split(':')[1]
        
        hdr_temp[16]=hdr_ini_str[7]
        hdr_temp[17]=hdr_ini_str[9]
        
    
        hdr_temp[23]='initial creation date:'+ hdr_ini_str[4].split(':')[1]
        hdr_temp[24]='initial software: '+hdr_ini_str[3].split(':')[1]
        hdr_temp[25]='initial source files: '+hdr_ini_str[6].split(':')[-1]
        hdr_temp[28]=hdr_ini_str[29][:-1] + ' CR\n'
        
        return hdr_temp

    def _generate_hdr_phion(self,hdr_temp,hdr_ini_str,fn_out):
        '''
        Reads and aranges lines from the initial photoionization rate in the ecliptic plane file into uniform structure of the header that will be used in L3d text files
        '''
        
        hdr_temp[1]='filename: ' + fn_out+'\n'
        hdr_temp[2]='initial file: '+ hdr_ini_str[1].split(' ')[1][:-1] + ' / '+ hdr_ini_str[5].split(',')[0].split(':')[1]+' ('+ hdr_ini_str[5].split(',')[2][1:-1]+') / '+hdr_ini_str[8].split(';')[0].split(':')[1]+'\n'
        hdr_temp[3]='description: '+(' ').join(hdr_ini_str[2].split(' ')[2:-1]) + '  ['+hdr_ini_str[11].split(' ')[1][:-1]+']\n'
        
        hdr_temp[16]=hdr_ini_str[3]
        hdr_temp[17]=hdr_ini_str[4]
        hdr_temp[18]=hdr_ini_str[10]
    
        hdr_temp[23]='initial creation date:'+ hdr_ini_str[7].split(':')[1]
        hdr_temp[24]='initial software: '+(' ').join(hdr_ini_str[6].split(' ')[1:4])
        hdr_temp[25]='initial source files: '+hdr_ini_str[13].split(':')[-1]
        hdr_temp[28]=hdr_ini_str[29][:-1] + ' CR\n'
        
        return hdr_temp

    def _generate_hdr_speed_dens(self,hdr_temp,hdr_ini_str,fn_out):
        '''
        Reads and aranges lines from the initial solar wind speed and density profile files into uniform structure of the header that will be used in L3d text files
        '''
        
        hdr_temp[1]='filename: ' + fn_out+'\n'
        hdr_temp[2]='initial file: '+ hdr_ini_str[2].split(':')[1][:-1] + ' / '+ hdr_ini_str[4].split(':')[1].split(',')[0]+' ('+ hdr_ini_str[5].split(':')[1][:-1]+') / '+hdr_ini_str[6].split(':')[1]
        hdr_temp[3]='description: '+ hdr_ini_str[12].split(':')[1]
        
        hdr_temp[16]=hdr_ini_str[7]
        hdr_temp[17]=hdr_ini_str[9]
        hdr_temp[18]=hdr_ini_str[11]
        hdr_temp[19]=hdr_ini_str[13]
        hdr_temp[20]=hdr_ini_str[20]
    
        hdr_temp[23]='initial creation date:'+ ':'.join(hdr_ini_str[1].split(':')[1:])
        hdr_temp[24]='initial software: '+ hdr_ini_str[21].split(' ')[-1][:-1] + ', ' + hdr_ini_str[22].split(' ')[-1]
        hdr_temp[25]='initial source files: '+ hdr_ini_str[10].split(' ')[-1][:-1]+ ', ' + hdr_ini_str[23].split(' ')[-1]

        hdr_temp[27]='latitudinal grid deg: '+ str(self.settings['solar_wind_grid']) + '\n'
        hdr_temp[28]=hdr_ini_str[29]
        
        return hdr_temp

    def _generate_hdr_uv_anis(self,hdr_temp,hdr_ini_str,fn_out):
        '''
        Reads and aranges lines from the initial uv anisotrpy profile file into uniform structure of the header that will be used in L3d text files
        '''
        
        hdr_temp[1]='filename: ' + fn_out+'\n'
        hdr_temp[2]='initial file: '+ hdr_ini_str[1].split(':')[1][:-1] + ' / '+ hdr_ini_str[2].split('/')[0].split(':')[1]+' ('+ hdr_ini_str[2].split('/')[-1][2:-1]+') / '+hdr_ini_str[6].split(':')[1]
        
        hdr_temp[16]=hdr_ini_str[16]
    
        hdr_temp[23]='initial creation date:'+ hdr_ini_str[3].split(':')[1]
        hdr_temp[24]='initial software: '+ hdr_ini_str[4].split(':')[1]
        hdr_temp[25]='initial source files: '+ hdr_ini_str[5].split(':')[1]
        hdr_temp[27]='latitudinal grid deg: '+ str(self.settings['solar_wind_grid']) + '\n'
        hdr_temp[28]=hdr_ini_str[28]
        
        return hdr_temp
    
    def _generate_hdr_template(self):
        '''
        Generates unified header for the L3d text files
        '''
        
        # Elements in the header that are common for all files
        N=self.settings['hdr_txt_lines_number']
        # initialize uniform header structure
        hdr_temp=N*['\n']
        hdr_temp[0]='lines in header: '+str(N)+'\n'
        hdr_temp[4]='ground software version: '+VERSION+'\n'
        hdr_temp[5]='creation date: '+ Time.now().iso +'\n'
        hdr_temp[6]='input files: ' + ', '.join(self.header['l3b_input_filename']) + ', ' + ', '.join(self.header['l3c_input_filename']) +'\n'
        hdr_temp[7]='external dependencies: ' + self.header['external_dependeciens'] + '\n'
        hdr_temp[8]='last CR: '+ str(self.CR_last) + '\n'
        hdr_temp[9]='fill value: '+ str(self.settings['WawHelioIonGlows_fill_value']) + '\n'
        hdr_temp[15]='Remarks:\n'
        hdr_temp[29]='###############################################'
        return hdr_temp

    
    def _read_ecliptic(self,fn):
        '''
        Reads L3d text files that are 1D value in the ecliptic plane
        '''
        data=np.loadtxt(fn)
        # time, value

        t=Time(data[:,0],format='decimalyear')

        return t, data[:,1]


    def _read_files(self,fn_list):
        '''
        Reads L3d text files
        '''
        # Read ionization files
        N=self.settings['hdr_txt_lines_number']
        for f in fn_list:
            self.ini_data['hdr_txt'][f]=fun.read_hdr_txt(fn_list[f],N)
            if np.logical_or(np.logical_or(f=='speed',f=='p-dens'),f=='uv-anis'):
                self.ini_data['time'][f], self.ini_data['data'][f]=self._read_profile(fn_list[f])
            elif np.logical_or(np.logical_or(f=='lya',f=='phion'),f=='e-dens'):
                self.ini_data['time'][f], self.ini_data['data'][f]=self._read_ecliptic(fn_list[f])
        
    def _read_profile(self,fn):
        '''
        Reads L3d text files that are 2D profiles (on the time, latitude grid)
        '''
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
        for k in self.ini_data['label']: output['solar_params'][k]=self.solar_params[k].tolist()
        json_content = json.dumps(output, indent=3)
        
        output_fp = open(fn, 'w')
        print(json_content, file=output_fp)
        output_fp.close()

    def save_to_txt(self,fn_out,hdr):
        '''
        Write to the text file
        '''
        
        for f in fn_out:
            if np.logical_or(np.logical_or(f=='speed',f=='p-dens'),f=='uv-anis'):
                self._save_to_txt_profile(fn_out[f],f,hdr[f])
            elif np.logical_or(np.logical_or(f=='lya',f=='phion'),f=='e-dens'):
                self._save_to_txt_ecliptic(fn_out[f],f,hdr[f])

    def _save_to_txt_ecliptic(self,fn,k,hdr):
        '''
        Write to the text file parameters in the ecliptic plane (1D)
        '''
        Ncol= 3   # number of columns in eliptic-type files (time, value, CR)
        Nrow=len(self.solar_params[k])
        output = np.zeros((Nrow,Ncol))

        output[:,0]=np.array([t.decimalyear for t in self.time_grid])
        output[:,1]=self.solar_params[k]
        output[:,-1]=np.array(self.CR_grid)

        np.savetxt(fn,output,fmt='%4.12f %.15e %4.1f',header=('').join(hdr))
        return 0
    
    def _save_to_txt_profile(self,fn,k,hdr):
        '''
        Write to the text file parameters that are profiles on the latitudinal grid
        '''
        Ncol=len(self.settings['solar_wind_grid'])+2   # number of columns in profile-type files (time, grid bins, CR)
        Nrow=len(self.solar_params[k])
        output = np.zeros((Nrow,Ncol))

        output[:,0]=np.array([t.decimalyear for t in self.time_grid])
        output[:,1:Ncol-1]=self.solar_params[k]
        output[:,-1]=np.array(self.CR_grid)

        np.savetxt(fn,output,fmt='%4.12f '+ (Ncol-2)*'%.15e ' + '%4.1f',header=('').join(hdr))
        return 0
    

    def _update_l3bc_data(self,data_l3b,data_l3c,CR):
        '''
        Generates new value for current CR of the solar parameters.
        '''
        anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read = self._generate_cr_solar_params(CR, data_l3b, data_l3c)

        self.header['l3b_input_filename']=[data_l3b[i]['header']['filename'] for i in idx_read]
        self.header['l3c_input_filename']=[data_l3c[i]['header']['filename'] for i in idx_read]

        self.solar_params['phion'] = np.append(self.solar_params['phion'], ph_ion_CR)
        self.solar_params['e-dens'] = np.append(self.solar_params['e-dens'], e_dens_CR)

        self.solar_params['speed'] = np.r_[self.solar_params['speed'],[sw_speed_CR]]
        self.solar_params['p-dens'] = np.r_[self.solar_params['p-dens'],[p_dens_CR]]
        self.solar_params['uv-anis'] = np.r_[self.solar_params['uv-anis'],[anisotropy_CR]]

    def _update_lya_data(self,ext_dependencies,CR):
        '''
        Generates next value of the Lyman-alpha irradiance based on external daily measurements.
        '''

        data_lya = fun.read_lya_data(ext_dependencies['lya_raw_data'])
        lya_CR = fun.generate_cr_lya(CR, data_lya)
        self.solar_params['lya'] = np.append(self.solar_params['lya'], lya_CR)

    def update_solar_params_hist(self,ext_dependencies,data_l3b,data_l3c):
        '''
        Updates structure by appending new values for current CR to the initial history 
        '''
        CR=self.ini_data['CR_last']+1

        # update CR grid
        self.CR_grid = np.append(self.CR_grid, CR)

        # update time grid
        t_CR = Time(fun.jd_fm_Carrington(CR),format='jd')
        self.time_grid = np.append(self.time_grid, t_CR)

        # update Lyman-alpha data
        self._update_lya_data(ext_dependencies,CR)

        # upade rest of the parameters
        self._update_l3bc_data(data_l3b,data_l3c,CR)

        # update last CR
        self.CR_last=CR
