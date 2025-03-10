"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""
import json
import copy
import numpy as np
import time
import sys
from .constants import VERSION, L1B_L2_HIST_FLAG_FIELDS, SCAN_CIRCLE_ANGULAR_RADIUS
from .funcs import time_sec_subsec_to_utc, calibration_factor
from .geom_fun_libr_box import spin2lb

class L2DataHistogram():
    """
    L2DataHistogram() class for GLOWS-L2-histogram data
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

        # self.var is for various variables needed for computations, but not to be saved in L2 output file
        self.var = {}

        # initialize counters
        self.var['l1b_file_counter_all'] = 0 # all l1b files processed for a given day
        self.var['l1b_file_counter_good_time'] = 0 # only good-time files for a given day

        # read pipeline settings to be used for L2
        file_handler = open(anc_input_from_instr_team['settings'], 'r')
        self.var['settings'] = json.load(file_handler)
        file_handler.close()

        # sanity check for bad-angle flags (all L1B_L2_HIST_FLAG_FIELDS flags must be included in
        # self.var['settings'] in the correct order)
        list_1 = list(self.var['settings']['active_bad_angle_flags'].keys())
        list_2 = list(L1B_L2_HIST_FLAG_FIELDS.keys())
        assert len(list_1) == len(list_2), \
            'Different numbers of keys in self.var[\'settings\'][\'active_bad_angle_flags\'] ' + \
            'and in L1B_L2_HIST_FLAG_FIELDS'
        for i, __ in enumerate(list_1):
            assert list_1[i] == list_2[i], \
                'Different keys "' + list_1[i] + '" and "' + list_2[i] + \
                '" in self.var[\'settings\'][\'active_bad_angle_flags\'] and L1B_L2_HIST_FLAG_FIELDS'

        # define a structure to keep track of occurences of bad-time flags
        self.var['bad_time_flag_occurences'] = {}
        for key in self.var['settings']['active_bad_time_flags'].keys():
            self.var['bad_time_flag_occurences'][key] = 0

        # define a bad-time mask which is used to decide which L1b files are accepted
        # as a whole during generation of L2 data product
        self.var['bad_time_mask'] = \
            ''.join(['%d' % self.var['settings']['active_bad_time_flags'][key]
                     for key in self.var['settings']['active_bad_time_flags']])
        self.var['bad_time_mask'] = '0b' + self.var['bad_time_mask'][::-1]

        # define a bad-angle mask which is used to decide which bins of L1b histograms are accepted
        # during generation of L2 data product
        self.var['bad_angle_mask'] = \
            ''.join(['%d' % self.var['settings']['active_bad_angle_flags'][key]
                     for key in self.var['settings']['active_bad_angle_flags']])
        self.var['bad_angle_mask'] = '0b' + self.var['bad_angle_mask'][::-1]

        self.var['is_night_for_previous_l1b'] = -1 # stores is-night-flag state from the previously processed L1b file
        self.var['is_night_counter'] = 0 # counter of L1b histograms for which is_night flag has been raised

    def _averages_accumulate(self, l1b_data):
        """
        Accumulate dictionary for computing L2 averages by appending for each key the average value
            from the currently processed L1b file
        Args:
            l1b_data: L1b data provided as data field of L1bDataHistogram object
        """
        for key in self.var['l1b_averages'].keys():
            if key == 'counter':
                continue
            if isinstance(self.var['l1b_averages'][key], dict):
                for second_key in self.var['l1b_averages'][key].keys():
                    self.var['l1b_averages'][key][second_key].append(l1b_data[key][second_key])
            else:
                self.var['l1b_averages'][key].append(l1b_data[key])
        self.var['l1b_averages']['counter'] += 1

    def _averages_init(self, l1b_data):
        """
        Intialize dictonary for computing L2 averages and spreads of '*_average' parameters over
            L1b files
        Args:
            l1b_data: L1b data provided as data field of L1bDataHistogram object
        """

        # generate dictionary for computing averages over GOOD TIMES for L1b-file ancillary parameters
        self.var['l1b_averages'] = {}
        for key in l1b_data.keys():
            if key.endswith('average'):
                if isinstance(l1b_data[key], dict):
                    self.var['l1b_averages'][key] = {}
                    for second_key in l1b_data[key].keys():
                        self.var['l1b_averages'][key][second_key] = []
                else:
                    self.var['l1b_averages'][key] = []
        self.var['l1b_averages']['counter'] = 0

    def _check_l1b_bad_time_flags(self, l1b_data):
        """
        Check bad-time (i.e., related to L1b histogram as a whole) flags for L1b histogram with
        respect to predefined self.var['bad_time_mask'] which determines bit fields to be taken
        into account for the checking
        Args:
            l1b_data: L1b data provided in the form of data field of L1bDataHistogram object
        """
        # sanity check, all bad-time flags in L1b file must be included in self.var['settings']
        # in the correct order, otherwise we can assume that an error occurred
        list_1 = list(self.var['settings']['active_bad_time_flags'].keys())
        list_2 = list(l1b_data['flags'].keys())
        assert len(list_1) == len(list_2), \
            'Different number of keys in self.var[\'settings\'][\'active_bad_time_flags\'] ' + \
            'and in l1b_data[\'flags\']'
        for i, __ in enumerate(list_1):
            assert list_1[i] == list_2[i], \
                'Different keys "' + list_1[i] + '" and "' + list_2[i] + \
                '" in self.var[\'settings\'][\'active_bad_time_flags\'] and l1b_data[\'flags\']'

        # set bad-time-flag bits to compare it with self.var['bad_time_mask']
        l1b_bad_time_flags = ''.join(['%d' % l1b_data['flags'][key] for key in l1b_data['flags']])
        l1b_bad_time_flags = '0b' + l1b_bad_time_flags[::-1]

        # compare bad_time_flags for L1b with mask
        cmp_result = int(l1b_bad_time_flags, 2) & int(self.var['bad_time_mask'], 2)

        # if self.var['bad_time_mask'] has inactive is_night field we have to account for
        # self.var['is_night_counter'] and correct the cmp_result
        if cmp_result == 0 and \
           self.var['is_night_counter'] >= self.var['settings']['number_of_good_histograms_at_night']:
            cmp_result = 1

        if l1b_data['flags']['is_night'] == True:
            print('(night', self.var['is_night_counter'], cmp_result, end=') ')

        if cmp_result != 0:
            print('-> bad time')
            return False

        print('-> good time')
        return True

    def _compute_anc_parms_averages_and_spreads(self):
        """
        Compute averages and spread (std dev) for accumulated L1b data. Mean value is computed over
            all accumulated L1b averages. Spread is computed as standard deviation of block averages.
        """
        if self.var['l1b_averages']['counter'] == 0: # there is nothing to compute
            return

        for key in self.var['l1b_averages'].keys():
            spread_key = '_'.join(key.split('_')[:-1]) + '_std_dev'
            if key == 'counter':
                continue
            if isinstance(self.var['l1b_averages'][key], dict):
                self.data[key] = {}
                self.data[spread_key] = {}
                for second_key in self.var['l1b_averages'][key].keys():
                    self.data[key][second_key] = np.mean(self.var['l1b_averages'][key][second_key])
                    self.data[spread_key][second_key] = \
                        np.std(self.var['l1b_averages'][key][second_key])
            else:
                self.data[key] = np.mean(self.var['l1b_averages'][key])
                self.data[spread_key] = np.std(self.var['l1b_averages'][key])

    def generate_l2_data(self, anc_input_from_instr_team):
        """
        The final part of settings of values of L2 fields
        Args:
            anc_input_from_instr_team: dictionary with names of ancillary files provided by the GLOWS team
                and MOC see ANC_INPUT_FROM_INSTRUMENT_TEAM in constants.py
        """

        self.data['header']['number_of_l1b_files_used'] = self.var['l1b_file_counter_good_time']
        self.data['header']['number_of_all_l1b_files'] = self.var['l1b_file_counter_all']

        if self.data['header']['number_of_l1b_files_used'] > 0:
            self.data['start_time'] = \
                time_sec_subsec_to_utc(int(self.var['l1b_start_time']),
                                       0).strftime('%Y-%m-%d %H:%M:%S')
            self.data['end_time'] = \
                time_sec_subsec_to_utc(int(self.var['l1b_end_time']),
                                       0).strftime('%Y-%m-%d %H:%M:%S')
        else:
            # good times have not been found
            self.data['start_time'] = \
                time_sec_subsec_to_utc(int(self.var['l1b_start_time_all']),
                                       0).strftime('%Y-%m-%d %H:%M:%S')
            self.data['end_time'] = \
                time_sec_subsec_to_utc(int(self.var['l1b_end_time_all']),
                                       0).strftime('%Y-%m-%d %H:%M:%S')
            self.data['daily_lightcurve']['exposure_times'] = \
                np.zeros(self.var['number_of_bins_per_histogram'], float)
            self.var['raw_histogram'] = \
                np.zeros(self.var['number_of_bins_per_histogram'], int)
            self.data['daily_lightcurve']['photon_flux'] = \
                np.zeros(self.var['number_of_bins_per_histogram'], float)
            self.data['daily_lightcurve']['flux_uncertainties'] = \
                np.zeros(self.var['number_of_bins_per_histogram'], float)

        # compute cps from raw histogram and exposure times
        idxs = np.nonzero(self.data['daily_lightcurve']['exposure_times'] > 0.0)[0]
        self.data['daily_lightcurve']['photon_flux'][idxs] = \
            self.var['raw_histogram'][idxs]/self.data['daily_lightcurve']['exposure_times'][idxs]

        # add raw histogram counts
        self.data['daily_lightcurve']['raw_histogram'] = self.var['raw_histogram']

        # compute uncertainties
        self.var['raw_uncertainties'] = np.sqrt(self.var['raw_histogram'])
        self.data['daily_lightcurve']['flux_uncertainties'][idxs] = \
            self.var['raw_uncertainties'][idxs]/ \
            self.data['daily_lightcurve']['exposure_times'][idxs]

        if self.data['header']['number_of_l1b_files_used'] > 0:
            # compute averages and spreads for parameters for L2
            self._compute_anc_parms_averages_and_spreads()

            # generate spin angle values (for histogram bin centers) measured from the northernmost
            # point of the scanning circle
            spin_angle_from_north = (self.data['daily_lightcurve']['spin_angle'] - \
                                     self.data['position_angle_offset_average'] + 360.0) % 360.0

            # redefine the spin angle, for L2 it is not the imap spin angle anymore but angle measured
            # from the northernmost part of the scanning circle
            self.data['daily_lightcurve']['spin_angle'] = spin_angle_from_north

            # find the index of the smallest value of spin_angle_from_north, it indicates the shift
            # value for rolling elements of self.data['daily_lightcurve'] arrays
            idx = np.argsort(spin_angle_from_north)[0]

            # we have the shift value, so we can roll all arrays in self.data['daily_lightcurve']
            for key in self.data['daily_lightcurve'].keys():
                self.data['daily_lightcurve'][key] = np.roll(self.data['daily_lightcurve'][key], -idx)

            # compute ecliptic longitude and latitude of histogram bin centers
            ones = np.ones(len(self.data['daily_lightcurve']['spin_angle']), float)
            histogram_bins_ecl_coords = np.degrees(spin2lb(
                np.radians(self.data['spin_axis_orientation_average']['lon'])*ones,
                np.radians(self.data['spin_axis_orientation_average']['lat'])*ones,
                np.radians(self.data['daily_lightcurve']['spin_angle']), 0.0,
                np.radians(SCAN_CIRCLE_ANGULAR_RADIUS)*ones))
            self.data['daily_lightcurve']['ecliptic_lon'] = histogram_bins_ecl_coords[:, 0]
            self.data['daily_lightcurve']['ecliptic_lat'] = histogram_bins_ecl_coords[:, 1]

        # apply calibration to get helioglow intensity in physical units of Rayleigh
        cps_per_R = calibration_factor(anc_input_from_instr_team['calibration_data'],
            self.data['start_time'].replace(' ', 'T'))
        self.data['daily_lightcurve']['photon_flux'] = \
            self.data['daily_lightcurve']['photon_flux'] / cps_per_R
        self.data['daily_lightcurve']['flux_uncertainties'] = \
            self.data['daily_lightcurve']['flux_uncertainties'] / cps_per_R

        # add the number of bins as a key to be saved in L2 output file
        self.data['daily_lightcurve']['number_of_bins'] = \
            len(self.data['daily_lightcurve']['spin_angle'])

        self.data['bad_time_flag_occurences'] = self.var['bad_time_flag_occurences']

    def is_new_observational_day(self, l1b_data):
        """
        Check if we have a new observational day based on is_night flag from the previous and current
        L1b histograms. PRESUMABLY, IN THE SDC IMPLEMENTATION, THIS PROCEDURE SHOULD USE THE OFFICIAL
        IMAP POINTING COUNTER INSTEAD OF THE IS_NIGHT FLAG. OTHERWISE, HANDLING GAPS IN THE TIMELINE
        COVERAGE BY HISTOGRAMS COULD BECOME PROBLEMATIC IN EMERGENCY SITUATIONS.
        Args:
            l1b_data: L1b data provided in the form of data field of L1bDataHistogram object
        """

        is_new_observ_day = False # init

        # check if is_night flag at L1b goes from True to False, if so then we have a new observational day
        if self.var['is_night_for_previous_l1b'] >= 0: # we are at the second (or later) l1b file
            if self.var['is_night_for_previous_l1b'] == 1 and l1b_data['flags']['is_night'] == False: # new observational day
                is_new_observ_day = True

        self.var['is_night_for_previous_l1b'] = l1b_data['flags']['is_night']
        return is_new_observ_day

    def process_l1b_data_files(self, l1b_data, pointing_counter):
        """
        Set values of L2 fields using L1b data read from a file
        Args:
            l1b_data: L1b data provided in the form of data field of L1bDataHistogram object
        """

        if self.var['l1b_file_counter_all'] == 0:
            # the start time of the first L1b file
            self.var['l1b_start_time_all'] = l1b_data['imap_start_time']

            # generate spin-angle values for L2 for which spin=0 is measured from the
            # northernmost point of lightcurve, however it must have the same resolution as
            # l1b histograms so we can use L1b imap_spin_angle values for angular-grid generation
            self.data['daily_lightcurve']['spin_angle'] = l1b_data['imap_spin_angle_bin_cntr']
        self.var['l1b_end_time_all'] = l1b_data['imap_start_time'] + l1b_data['imap_end_time_offset']
        self.var['number_of_bins_per_histogram'] = l1b_data['number_of_bins_per_histogram']

        # increment all-l1b-files counter
        self.var['l1b_file_counter_all'] += 1

        for key in self.var['settings']['active_bad_time_flags'].keys():
            if l1b_data['flags'][key]:
                self.var['bad_time_flag_occurences'][key] += 1

        if self._check_l1b_bad_time_flags(l1b_data):

            if self.var['l1b_file_counter_good_time'] <= 0:
                # things to do when we have the first good-time L1b data file
                self.data['header'] = {
                    'flight_software_version': l1b_data['block_header']['flight_software_version'],
                    'ground_software_version': VERSION,
                    'pkts_file_name': l1b_data['block_header']['pkts_file_name'],
                    'ancillary_data_files': l1b_data['block_header']['ancillary_data_files']
                }

                # we are here at the first good-time L1b file, so we set the current state of
                # the pointing counter as an unique identifier in L2
                self.data['identifier'] = pointing_counter

                # the start time of the first good-time L1b file is needed for start time of L2
                self.var['l1b_start_time'] = l1b_data['imap_start_time']

                # initialize dictionary for computing averages over L1b files
                self._averages_init(l1b_data)

                # initialize daily helioglow-flux-lightcurve related stuff
                self.data['daily_lightcurve']['photon_flux'] = \
                    np.zeros_like(self.data['daily_lightcurve']['spin_angle'], float)
                self.var['raw_histogram'] = \
                    np.zeros_like(self.data['daily_lightcurve']['spin_angle'], int)
                self.data['daily_lightcurve']['exposure_times'] = \
                    np.zeros_like(self.data['daily_lightcurve']['spin_angle'], float)
                self.data['daily_lightcurve']['flux_uncertainties'] = \
                    np.zeros_like(self.data['daily_lightcurve']['spin_angle'], float)
                self.data['daily_lightcurve']['histogram_flag_array'] = \
                    np.zeros_like(self.data['daily_lightcurve']['spin_angle'], np.int16)

            # keep track of the last end_time for good-time L1b files, it is needed for
            # the end time of L2 file
            self.var['l1b_end_time'] = l1b_data['imap_start_time'] + l1b_data['imap_end_time_offset']

            # update dictionary for computing averages over L1b files
            self._averages_accumulate(l1b_data)

            # generate bad-angle mask for L2, if any L1b file has bitfield set to True, L2 does too
            self.data['daily_lightcurve']['histogram_flag_array'] = \
                np.bitwise_and(int(self.var['bad_angle_mask'], 2),
                np.bitwise_or(self.data['daily_lightcurve']['histogram_flag_array'],
                              l1b_data['histogram_flag_array']))

            # compute the exposure time per bin for currently processed L1b-data file
            # we must do it here because it can vary from one block to another
            l1b_exposure_time_per_bin = l1b_data['spin_period_average'] * \
                                        l1b_data['number_of_spins_per_block'] / \
                                        l1b_data['number_of_bins_per_histogram']

            # update raw L2 histogram and exposure times, note that block-to-block variations of the
            # position angle offset are not taken into account here, but if they will turn to be
            # significant we must include them somehow here
            self.var['raw_histogram'] += np.array(l1b_data['histogram'])
            self.data['daily_lightcurve']['exposure_times'] += l1b_exposure_time_per_bin

            # increment good-time counter for L1b files
            self.var['l1b_file_counter_good_time'] += 1

        if l1b_data['flags']['is_night'] == True:
            # increment is-night-flag-raised counter for L1b files
            self.var['is_night_counter'] += 1

    def read_from_file(self, file_name):
        """
        Read L2 data from file
        Args:
            file_name: name of of file with L2 data
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        #    modules for IMAP will be provided by SDC
        self._read_from_json_file(file_name)

    def _read_from_json_file(self, file_name):
        """
        Read L2 data from JSON file
        Args:
            file_name: name of of file with L1b data
        """
        file_handler = open(file_name, 'r')
        self.data = json.load(file_handler)
        file_handler.close()

        # hexadecimal values saved as strings require special treatment to make them integers
        self.data['daily_lightcurve']['histogram_flag_array'] = \
            [int(x, base=16) for x in self.data['daily_lightcurve']['histogram_flag_array']]

        # additional field to keep track where data come from
        self.data['l2_file_name'] = file_name

    def save_data_to_file(self):
        """
        Save generated L2 data to a file
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        # modules for IMAP will be provided by SDC
        self._save_data_to_json_file()

        #sys.stdout.flush()
        #time.sleep(5)

    def _save_data_to_json_file(self):
        """
        Save generated L2 data to JSON file
        Output file name is set automatically here (TBC if perhaps it should be method argument)
        """
        # define formats for printing values to JSON file
        fmt = {
            'daily_lightcurve': {
                'spin_angle': '%.3f',
                'photon_flux': '%.1f',
                'raw_histogram': '%d',
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
            data[key] = '|'+ fmt['scalar variables'][key] % data[key] +'|'
        for key in fmt['lon lat variables'].keys():
            for sec_key in ['lon', 'lat']:
                data[key][sec_key] = '|'+ fmt['lon lat variables'][key] % data[key][sec_key] +'|'
        for key in fmt['vector variables'].keys():
            for sec_key in ['x', 'y', 'z']:
                data[key][sec_key] = '|'+ fmt['vector variables'][key] % data[key][sec_key] +'|'

        # histogram_flag_array requires special hexadecimal treatment
        data['daily_lightcurve']['histogram_flag_array'] = '[' + \
            ', '.join(map(lambda x: '"0x' + format(x, 'x').zfill(4) +'"',
            list(data['daily_lightcurve']['histogram_flag_array']))) + ']'

        # generate JSON content to be saved in output file
        json_content = json.dumps(data, indent=4, default=vars)

        # some corrections to get numbers back
        json_content = json_content.replace('"[', '[').replace(']"', ']')
        json_content = json_content.replace('|"', '').replace('"|', '')
        json_content = json_content.replace('\\"', '"')

        # the output file name is constructed using a convention defined in Sec. 1.4.2 of
        # "IMAP SDC to Instrument Team ICD"
        date_utc = ''.join(filter(str.isdigit, self.data['start_time']))
        file_name = 'data_l2_histograms/imap_glows_l2_%s_orbX_modX_p_v00.json' % date_utc

        file_handler = open(file_name, 'w')
        print(json_content, file=file_handler)
        file_handler.close()
