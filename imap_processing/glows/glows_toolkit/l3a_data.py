"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""
import json
import copy
import numpy as np
import healpy as hp
from .constants import VERSION, SCAN_CIRCLE_ANGULAR_RADIUS
from .funcs import calibration_factor, check_if_contains_actual_data
from .geom_fun_libr_box import spin2lb


class L3aData():
    """
    L3aData() class for GLOWS-L3a data
    """

    def __init__(self, anc_input_from_instr_team):
        """
        Constructor for this class
        """
        # self.data is a dictionary to be saved as L2 data
        self.data = {}
        self.data['header'] = {}
        self.data['identifier'] = None
        self.data['start_time'] = None
        self.data['end_time'] = None
        self.data['daily_lightcurve'] = {}

        # self.var is for various variables needed for computations, but not to be saved in L3a output file
        self.var = {}

        # read pipeline settings to be used for L3a
        file_handler = open(anc_input_from_instr_team['settings'], 'r')
        settings = json.load(file_handler)
        file_handler.close()
        self.var['nominal_number_of_bins'] = settings['l3a_nominal_number_of_bins']

    def _compute_extra_heliospheric_bckgrd(self, anc_input_from_instr_team):
        """
        Compute extra-heliospheric background for L3a scanning circle by interpolation from
        a map provided by the IMAP/GLOWS team as an ancillary file
        Args:
            anc_input_from_instr_team: dictionary with names of ancillary files provided by the GLOWS team
                see ANC_INPUT_FROM_INSTRUMENT_TEAM in constants.py
        """

        if check_if_contains_actual_data(anc_input_from_instr_team['extra_heliospheric_bckgrd']):

            fh = open(anc_input_from_instr_team['extra_heliospheric_bckgrd'], 'r')
            for i in range(3):
                line = fh.readline()
            # sanity check
            if line.split()[4] != 'NSIDE':
                raise Exception('error _compute_extra_heliospheric_bckgrd(): NSIDE string not found')
            # read NSIDE parameter for healpix grid
            nside = int(line.split()[6])
            # read ecliptic coords and background values span on a healpix grid
            lon_lat_bckgrd = np.loadtxt(fh)
            fh.close()

            # generate Healpix grid internally here for a sanity check performed later
            ipxls = hp.query_strip(nside, 0.0, np.pi, inclusive=True)
            lon, lat = hp.pix2ang(nside, ipxls, lonlat=True)
            # sanity check for the Healpix grid read from anc_input_from_instr_team['extra_heliospheric_bckgrd']
            # it should match the Healpix grid generated internally, otherwise raise error
            if np.fabs(lon - lon_lat_bckgrd[:, 0]).max() > 1.0e-6 or \
                    np.fabs(lat - lon_lat_bckgrd[:, 1]).max() > 1.0e-6:
                raise Exception('error _compute_extra_heliospheric_bckgrd(): ' + \
                                'healpix grid mismatch (NSIDE wrong? NESTED ordering instead of RING?)')

            # interpolate the background estimate from the healpix map into the centers of the scanning circle bins
            self.data['daily_lightcurve']['extra_heliospheric_bckgrd'] = hp.pixelfunc.get_interp_val(
                lon_lat_bckgrd[:, 2],
                self.data['daily_lightcurve']['ecliptic_lon'], self.data['daily_lightcurve']['ecliptic_lat'],
                nest=False, lonlat=True)

    def _compute_time_dependent_bckgrd(self, anc_input_from_instr_team):
        """
        Compute time-dependent background corrections for L3a scanning circle by reading arrays
        from an ancillary file provided by the IMAP/GLOWS team
        Args:
            anc_input_from_instr_team: dictionary with names of ancillary files provided by the GLOWS team
                see ANC_INPUT_FROM_INSTRUMENT_TEAM in constants.py
        """

        if check_if_contains_actual_data(anc_input_from_instr_team['time_dependent_bckgrd']):

            # read the size of the background-estimate array
            fh = open(anc_input_from_instr_team['time_dependent_bckgrd'], 'r')
            for i in range(2):
                line = fh.readline()
            # sanity check
            if line.split()[5] != 'array' or line.split()[6] != 'size' or line.split()[7] != '=':
                raise Exception(
                    'error _compute_time_dependent_bckgrd(): input file header does not contain expected keywords (1)')
            # read the number of bins
            bckgrd_array_size = int(line.split()[8])
            # sanity check
            if bckgrd_array_size != 3600:
                raise Exception(
                    'error _compute_time_dependent_bckgrd(): array size in input file header is not equal to 3600')

            # read the array of spin angles for which the background was estimated
            line = fh.readline()
            # sanity check
            if line.split()[1] != 'spin' or line.split()[2] != 'angle' or line.split()[3] != 'grid:':
                raise Exception(
                    'error _compute_time_dependent_bckgrd(): input file header does not contain expected keywords (2)')
            bckgrd_array_spin_angle = np.fromstring(line[18:], dtype=float, sep=' ')
            fh.close()

            # read L3a lightcurve start dates and array of corrections
            identifiers = np.loadtxt(anc_input_from_instr_team['time_dependent_bckgrd'], usecols=[0], dtype=str)
            start_dates = np.loadtxt(anc_input_from_instr_team['time_dependent_bckgrd'], usecols=[1], dtype=str)
            bckgrd_arrays = np.loadtxt(anc_input_from_instr_team['time_dependent_bckgrd'],
                                       usecols=range(2, bckgrd_array_size + 2))
            idx = np.nonzero(self.data['start_time'].replace(' ', 'T') == start_dates)[0]
            print(' tdbckgrd idx:', idx, '  ', end='')
            if idx.size == 0:
                # corrections has not been found for the current date, so we set zeros
                self.data['daily_lightcurve']['time_dependent_bckgrd'] = \
                    np.zeros_like(self.data['daily_lightcurve']['spin_angle'], float)
            else:
                idx = idx[0]
                if idx.size != 1:
                    raise Exception(
                        'error _compute_time_dependent_bckgrd(): not-found or multiple-definition error for ' +
                        'the start date %s' % self.data['start_time'].replace(' ', 'T'))

                # interpolate the background estimate from the bckgrd_array into the centers of the scanning circle bins
                self.data['daily_lightcurve']['time_dependent_bckgrd'] = \
                    np.interp(self.data['daily_lightcurve']['spin_angle'], bckgrd_array_spin_angle,
                              bckgrd_arrays[idx, :])

    def generate_l3a_data(self, anc_input_from_instr_team):
        """
        The final part of settings of values of L3a fields
        Args:
            anc_input_from_instr_team: dictionary with names of ancillary files provided by the GLOWS team
                see ANC_INPUT_FROM_INSTRUMENT_TEAM in constants.py
        """

        # compute count rate for L3a
        self.var['count_rate'] = np.zeros_like(self.data['daily_lightcurve']['raw_histogram'], float)
        good_angle_idxs = np.nonzero(np.array(self.data['daily_lightcurve']['exposure_times']) > 0)[0]
        self.var['count_rate'][good_angle_idxs] = \
            self.data['daily_lightcurve']['raw_histogram'][good_angle_idxs] / \
            self.data['daily_lightcurve']['exposure_times'][good_angle_idxs]

        # compute calibration factor
        cps_per_R = calibration_factor(anc_input_from_instr_team['calibration_data'],
                                       self.data['start_time'].replace(' ', 'T'))
        print(' cps/R:', cps_per_R, '  ', end='')

        # convert count rate to photon flux
        self.data['daily_lightcurve']['photon_flux'] = self.var['count_rate'] / cps_per_R

        # compute uncertainties
        self.var['raw_uncertainties'] = np.sqrt(self.data['daily_lightcurve']['raw_histogram'])
        self.data['daily_lightcurve']['flux_uncertainties'] = np.zeros_like(
            self.data['daily_lightcurve']['raw_histogram'], float)
        self.data['daily_lightcurve']['flux_uncertainties'][good_angle_idxs] = \
            self.var['raw_uncertainties'][good_angle_idxs] / \
            np.array(self.data['daily_lightcurve']['exposure_times'])[good_angle_idxs]
        self.data['daily_lightcurve']['flux_uncertainties'] = \
            self.data['daily_lightcurve']['flux_uncertainties'] / cps_per_R

        # compute ecliptic longitude and latitude of histogram bin centers
        ones = np.ones_like(self.data['daily_lightcurve']['spin_angle'], float)
        histogram_bins_ecl_coords = np.degrees(spin2lb(
            np.radians(self.data['spin_axis_orientation_average']['lon']) * ones,
            np.radians(self.data['spin_axis_orientation_average']['lat']) * ones,
            np.radians(self.data['daily_lightcurve']['spin_angle']), 0.0,
            np.radians(SCAN_CIRCLE_ANGULAR_RADIUS) * ones))
        self.data['daily_lightcurve']['ecliptic_lon'] = histogram_bins_ecl_coords[:, 0]
        self.data['daily_lightcurve']['ecliptic_lat'] = histogram_bins_ecl_coords[:, 1]

        # compute an estimate of extra-heliospheric background
        self._compute_extra_heliospheric_bckgrd(anc_input_from_instr_team)

        # compute an estimate of time dependent background
        self._compute_time_dependent_bckgrd(anc_input_from_instr_team)

        # remove masked bins from the lightcurve
        self._remove_masked_bins()

        # add the number of bins as a key to be saved in L3a output file
        self.data['daily_lightcurve']['number_of_bins'] = \
            len(self.data['daily_lightcurve']['spin_angle'])

        print('...  done')

    def process_l2_data_file(self, l2_data):
        """
        Set values of L3a fields using L2 data read from a file
        Args:
            l2_data: L2 data provided in the form of data field of L2DataHistogram object
        """

        # prepare quality_flag_array for L2, where 0 means bad data and 1 means good data
        l2_quality_flag_array = np.ones_like(l2_data['daily_lightcurve']['histogram_flag_array'], dtype=int)
        bad_angle_idxs = np.nonzero(np.array(l2_data['daily_lightcurve']['histogram_flag_array']) > 0)[0]
        l2_quality_flag_array[bad_angle_idxs] = 0

        # copy ancillary date from L2 to L3a - PART 1
        for key in ['identifier', 'start_time', 'end_time', 'filter_temperature_average', 'filter_temperature_std_dev',
                    'hv_voltage_average', 'hv_voltage_std_dev', 'spin_period_average', 'spin_period_std_dev',
                    'spin_period_ground_average', 'spin_period_ground_std_dev', 'pulse_length_average',
                    'pulse_length_std_dev',
                    'position_angle_offset_average', 'position_angle_offset_std_dev']:
            self.data[key] = l2_data[key]

        # copy ancillary date from L2 to L3a - PART 2
        for key1 in ['spin_axis_orientation_average', 'spin_axis_orientation_std_dev']:
            self.data[key1] = {}
            for key2 in ['lon', 'lat']:
                self.data[key1][key2] = l2_data[key1][key2]

        # copy ancillary date from L2 to L3a - PART 3
        for key1 in ['spacecraft_location_average', 'spacecraft_location_std_dev',
                     'spacecraft_velocity_average', 'spacecraft_velocity_std_dev']:
            self.data[key1] = {}
            for key2 in ['x', 'y', 'z']:
                self.data[key1][key2] = l2_data[key1][key2]

        # generate the nominal spin angle grid for L3a
        bin_width = 360.0 / self.var['nominal_number_of_bins']
        self.data['daily_lightcurve']['spin_angle'] = np.linspace(0.5 * bin_width,
                                                                  360.0 - 0.5 * bin_width,
                                                                  self.var['nominal_number_of_bins'])

        # compute raw counts and exposure times for low-resolution L3a histogram
        self.data['daily_lightcurve']['raw_histogram'] = []
        self.data['daily_lightcurve']['exposure_times'] = []
        for i in range(self.var['nominal_number_of_bins']):
            # find indexes of l2 bins falling into i-th l3a bin
            idxs = np.nonzero(np.logical_and(
                l2_data['daily_lightcurve']['spin_angle'] > self.data['daily_lightcurve']['spin_angle'][
                    i] - 0.5 * bin_width,
                l2_data['daily_lightcurve']['spin_angle'] <= self.data['daily_lightcurve']['spin_angle'][
                    i] + 0.5 * bin_width))[0]
            # compute raw counts
            self.data['daily_lightcurve']['raw_histogram'].append(
                np.sum(np.array(l2_data['daily_lightcurve']['raw_histogram'])[idxs] * l2_quality_flag_array[idxs]))
            # compute exposure times
            self.data['daily_lightcurve']['exposure_times'].append(
                np.sum(np.array(l2_data['daily_lightcurve']['exposure_times'])[idxs] * l2_quality_flag_array[idxs]))
        self.data['daily_lightcurve']['raw_histogram'] = np.array(self.data['daily_lightcurve']['raw_histogram'])
        self.data['daily_lightcurve']['exposure_times'] = np.array(self.data['daily_lightcurve']['exposure_times'])

        self.data['header'] = {
            'flight_software_version': l2_data['header']['flight_software_version'],
            'ground_software_version': VERSION,
            'pkts_file_name': l2_data['header']['pkts_file_name'],
            'ancillary_data_files': l2_data['header']['ancillary_data_files'],
            'l2_input_file_name': l2_data['l2_file_name']
        }

    def read_from_file(self, file_name):
        """
        Read L3a data from file
        Args:
            file_name: name of of file with L3a data
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        #    modules for IMAP will be provided by SDC
        self._read_from_json_file(file_name)

    def _read_from_json_file(self, file_name):
        """
        Read L3a data from JSON file
        Args:
            file_name: name of of file with L1b data
        """
        file_handler = open(file_name, 'r')
        self.data = json.load(file_handler)
        file_handler.close()

        # additional field to keep track where data come from
        self.data['l3a_file_name'] = file_name

    def _remove_masked_bins(self):
        """
        Remove masked bins from L3a lightcurve
        """
        good_angle_idxs = np.nonzero(np.array(self.data['daily_lightcurve']['exposure_times']) > 0)[0]
        for key in self.data['daily_lightcurve'].keys():
            self.data['daily_lightcurve'][key] = self.data['daily_lightcurve'][key][good_angle_idxs]

    def save_data_to_file(self):
        """
        Save generated L3a data to a file
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        # modules for IMAP will be provided by SDC
        self._save_data_to_json_file()

    def _save_data_to_json_file(self):
        """
        Save generated L3a data to JSON file
        Output file name is set automatically here (TBC if perhaps it should be method argument)
        """
        # define formats for printing values to JSON file
        fmt = {
            'daily_lightcurve': {
                'spin_angle': '%.3f',
                'photon_flux': '%.1f',
                'raw_histogram': '%d',
                'extra_heliospheric_bckgrd': '%.2f',
                'time_dependent_bckgrd': '%.2f',
                'exposure_times': '%.3e',
                'flux_uncertainties': '%.3e',
                'ecliptic_lon': '%.3f',
                'ecliptic_lat': '%.3f'
            },
            'scalar variables': {
                'filter_temperature_average': '%.2f',
                'filter_temperature_std_dev': '%.3e',
                'hv_voltage_average': '%.1f',
                'hv_voltage_std_dev': '%.3e',
                'spin_period_average': '%.7f',
                'spin_period_std_dev': '%.3e',
                'spin_period_ground_average': '%.7f',
                'spin_period_ground_std_dev': '%.3e',
                'pulse_length_average': '%.3e',
                'pulse_length_std_dev': '%.3e',
                'position_angle_offset_average': '%.3f',
                'position_angle_offset_std_dev': '%.3e'
            },
            'lon lat variables': {
                'spin_axis_orientation_average': '%.3f',
                'spin_axis_orientation_std_dev': '%.3e'
            },
            'vector variables': {
                'spacecraft_location_average': '%.1f',
                'spacecraft_location_std_dev': '%.3e',
                'spacecraft_velocity_average': '%.3f',
                'spacecraft_velocity_std_dev': '%.3e'
            }
        }

        # create local copy for printing purposes
        data = copy.deepcopy(self.data)

        # some fields need to be temporarily converted to string arrays, because we want to force
        # them to be saved as one-liners (i.e., without \n after each element) or we want them to 
        # be printed with a specific format by json.dumps()
        # so we convert variables to strings as defined in fmt
        for key in fmt['daily_lightcurve'].keys():
            data['daily_lightcurve'][key] = \
                '[' + ', '.join(map(lambda x: fmt['daily_lightcurve'][key] % x,
                                    list(data['daily_lightcurve'][key]))) + ']'
        for key in fmt['scalar variables'].keys():
            data[key] = '|' + fmt['scalar variables'][key] % data[key] + '|'
        for key in fmt['lon lat variables'].keys():
            for sec_key in ['lon', 'lat']:
                data[key][sec_key] = '|' + fmt['lon lat variables'][key] % data[key][sec_key] + '|'
        for key in fmt['vector variables'].keys():
            for sec_key in ['x', 'y', 'z']:
                data[key][sec_key] = '|' + fmt['vector variables'][key] % data[key][sec_key] + '|'

        # generate JSON content to be saved in output file
        json_content = json.dumps(data, indent=4, default=vars)

        # some corrections to get numbers back
        json_content = json_content.replace('"[', '[').replace(']"', ']')
        json_content = json_content.replace('|"', '').replace('"|', '')
        json_content = json_content.replace('\\"', '"')

        # the output file name is constructed using a convention defined in Sec. 1.4.2 of
        # "IMAP SDC to Instrument Team ICD"
        # date_utc = ''.join(filter(str.isdigit, self.data['identifier']))
        date_utc = ''.join(filter(str.isdigit, self.data['start_time']))
        file_name = 'data_l3a/imap_glows_l3a_%s_orbX_modX_p_v00.json' % date_utc

        file_handler = open(file_name, 'w')
        print(json_content, file=file_handler)
        file_handler.close()
