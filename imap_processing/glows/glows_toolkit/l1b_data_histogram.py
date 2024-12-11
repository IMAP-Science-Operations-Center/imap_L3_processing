"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""
import json
import copy
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from .constants import VERSION, SCAN_CIRCLE_ANGULAR_RADIUS, L1B_L2_HIST_FLAG_FIELDS, MAIN_TIME_FIELD
from .funcs import time_sec_subsec_to_utc, time_sec_subsec_to_float64, flags_deserialize, \
    decode_ancillary_parameters_avg_stddev, check_if_contains_actual_data
from .geom_fun_libr_box import spin2lb

class L1bDataHistogram():
    """
    L1bDataHistogram() class for GLOWS-L1b-histogram data
    """

    def __init__(self):
        """
        Constructor for this class
        """
        self.data = {}

        # aux variables that are expected to be set by create_from_l1a_data()
        self.block_start_time = None
        self.block_end_time = None

    def _block_header_add_list_of_anc_files(self, anc_input_from_sdc, anc_input_from_instr_team):
        """
        Add list of used ancillary files to block header
        Args:
            anc_input_from_sdc: ancillary input provided by IMAP SDC (general telemetry)
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
        """
        self.data['block_header']['ancillary_data_files'] = []
        for key in anc_input_from_instr_team:
            self.data['block_header']['ancillary_data_files'].append(anc_input_from_instr_team[key])
        for key in anc_input_from_sdc:
            self.data['block_header']['ancillary_data_files'].append(anc_input_from_sdc[key])

    def _compute_spin_period_ground_estimate(self, anc_data):
        """
        Compute the average and standard deviation for spin period using ancillary data
            from general IMAP telemetry available during ground processing
        TMP_SOLUTION: we use here a csv file with mock ancillary data, this procedure
            needs to be thoroughly updated when ancillary data from SDC will be available
        Args:
            anc_data: Pandas Dataframe with ancillary data from CSV file
        """
        # extract data frames: last before block start and first after block end
        df_1 = anc_data.loc[anc_data['imap_time'] <= self.block_start_time].iloc[-1, :]
        df_2 = anc_data.loc[anc_data['imap_time'] >= self.block_end_time].iloc[0, :]

        # generate synthetic series with spin-period data (assumed to arrive every 0.1 s)
        time = np.arange(self.block_start_time, self.block_end_time, 0.1)
        data = np.interp(time, [df_1['imap_time'], df_2['imap_time']],
                         [df_1['spin_period'], df_2['spin_period']])

        # compute spin-period average and standard deviation over block
        average = np.mean(data)
        std_deviation = np.std(data)
        return average, std_deviation

    def _compute_position_angle(self, anc_data):
        """
        Compute the average and standard deviation for GLOWS position angle using ancillary data
            from general IMAP telemetry available during ground processing
        TMP_SOLUTION: we use here a csv file with mock ancillary data, this procedure
            needs to be thoroughly updated when ancillary data from SDC will be available
        Args:
            anc_data: Pandas Dataframe with ancillary data from CSV file
        """
        # extract data frames: last before block start and first after block end
        df_1 = anc_data.loc[anc_data['imap_time'] <= self.block_start_time].iloc[-1, :]
        df_2 = anc_data.loc[anc_data['imap_time'] >= self.block_end_time].iloc[0, :]

        # generate synthetic series with position-angle data (assumed to arrive every 0.1 s)
        time = np.arange(self.block_start_time, self.block_end_time, 0.1)
        data = np.interp(time, [df_1['imap_time'], df_2['imap_time']],
                         [df_1['position_angle'], df_2['position_angle']])

        # compute position-angle average and standard deviation over block
        average = np.mean(data)
        std_deviation = np.std(data)
        return average, std_deviation

    def _compute_spin_axis_orientation(self, anc_data):
        """
        Compute the average and standard deviation for spin-axis orientation using ancillary data
            from general IMAP telemetry available during ground processing
        TMP_SOLUTION: we use here a csv file with mock ancillary data, this procedure
            needs to be thoroughly updated when ancillary data from SDC will be available
        Args:
            anc_data: Pandas Dataframe with ancillary data from CSV file
        """
        # extract data frames: last before block start and first after block end
        df_1 = anc_data.loc[anc_data['imap_time'] <= self.block_start_time].iloc[-1, :]
        df_2 = anc_data.loc[anc_data['imap_time'] >= self.block_end_time].iloc[0, :]

        # generate synthetic series with spin-axis-orientation data (assumed to arrive every 0.1 s)
        time = np.arange(self.block_start_time, self.block_end_time, 0.1)
        data_lon = np.interp(time, [df_1['imap_time'], df_2['imap_time']],
                             [df_1['spin_ax_lon'], df_2['spin_ax_lon']])
        data_lat = np.interp(time, [df_1['imap_time'], df_2['imap_time']],
                             [df_1['spin_ax_lat'], df_2['spin_ax_lat']])

        # compute spin-axis-orientation average and standard deviation over block
        average_lon = np.mean(data_lon)
        std_deviation_lon = np.std(data_lon)
        average_lat = np.mean(data_lat)
        std_deviation_lat = np.std(data_lat)
        return {'lon': average_lon, 'lat': average_lat}, \
               {'lon': std_deviation_lon, 'lat': std_deviation_lat}

    def _compute_spacecraft_location(self, anc_data):
        """
        Compute the average and standard deviation for spacecraft location using ancillary data
            from general IMAP telemetry available during ground processing
        TMP_SOLUTION: we use here a csv file with mock ancillary data, this procedure
            needs to be thoroughly updated when ancillary data from SDC will be available
        Args:
            anc_data: Pandas Dataframe with ancillary data from CSV file
        """
        # extract data frames: last before block start and first after block end
        df_1 = anc_data.loc[anc_data['imap_time'] <= self.block_start_time].iloc[-1, :]
        df_2 = anc_data.loc[anc_data['imap_time'] >= self.block_end_time].iloc[0, :]

        # generate synthetic series with spacecraft location data (assumed to arrive every 0.1 s)
        time = np.arange(self.block_start_time, self.block_end_time, 0.1)
        data_x = np.interp(time, [df_1['imap_time'], df_2['imap_time']], [df_1['x'], df_2['x']])
        data_y = np.interp(time, [df_1['imap_time'], df_2['imap_time']], [df_1['y'], df_2['y']])
        data_z = np.interp(time, [df_1['imap_time'], df_2['imap_time']], [df_1['z'], df_2['z']])

        # compute spacecraft-location average and standard deviation over block
        average = [np.mean(data_x), np.mean(data_y), np.mean(data_z)]
        std_deviation = [np.std(data_x), np.std(data_y), np.std(data_z)]
        return {'x': average[0], 'y': average[1], 'z': average[2]}, \
               {'x': std_deviation[0], 'y': std_deviation[1], 'z': std_deviation[2]}

    def _compute_spacecraft_velocity(self, anc_data):
        """
        Compute the average and standard deviation for spacecraft velocity using ancillary data
            from general IMAP telemetry available during ground processing
        TMP_SOLUTION: we use here a csv file with mock ancillary data, this procedure
            needs to be thoroughly updated when ancillary data from SDC will be available
        Args:
            anc_data: Pandas Dataframe with ancillary data from CSV file
        """
        # extract data frames: last before block start and first after block end
        df_1 = anc_data.loc[anc_data['imap_time'] <= self.block_start_time].iloc[-1, :]
        df_2 = anc_data.loc[anc_data['imap_time'] >= self.block_end_time].iloc[0, :]

        # generate synthetic series with spacecraft velocity data (assumed to arrive every 0.1 s)
        time = np.arange(self.block_start_time, self.block_end_time, 0.1)
        data_vx = np.interp(time, [df_1['imap_time'], df_2['imap_time']], [df_1['vx'], df_2['vx']])
        data_vy = np.interp(time, [df_1['imap_time'], df_2['imap_time']], [df_1['vy'], df_2['vy']])
        data_vz = np.interp(time, [df_1['imap_time'], df_2['imap_time']], [df_1['vz'], df_2['vz']])

        # compute spacecraft-velocity average and standard deviation over block
        average = [np.mean(data_vx), np.mean(data_vy), np.mean(data_vz)]
        std_deviation = [np.std(data_vx), np.std(data_vy), np.std(data_vz)]
        return {'x': average[0], 'y': average[1], 'z': average[2]}, \
               {'x': std_deviation[0], 'y': std_deviation[1], 'z': std_deviation[2]}

    def _compute_histogram_bins_ecl_coords(self):
        """
        Compute ecliptic longitude and latitude for histogram bin centers
        """
        n_bins = self.data['number_of_bins_per_histogram'] # just for convenience

        # generate spin angle values (for histogram bin centers) measured from the northernmost
        # point of the scanning circle
        spin_angle_from_north = (self.data['imap_spin_angle_bin_cntr'] - \
                                 self.data['position_angle_offset_average'] + \
                                 360.0) % 360.0

        # compute ecliptic longitude and latitude of histogram bin centers
        histogram_bins_ecl_coords = spin2lb(
            np.radians(self.data['spin_axis_orientation_average']['lon'])*np.ones(n_bins, float),
            np.radians(self.data['spin_axis_orientation_average']['lat'])*np.ones(n_bins, float),
            np.radians(spin_angle_from_north), 0.0,
            np.radians(SCAN_CIRCLE_ANGULAR_RADIUS)*np.ones(n_bins, float))

        return np.degrees(histogram_bins_ecl_coords) # in deg units

    def _convert_ancillary_parameters(self, l1a_data, anc_input_from_instr_team):
        """
        Convert ancillary parameters collected onboard from integer values to physcial
            units
        Args:
            l1a_data: L1a data provided in the form of data field of L1aDataHistogram object
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
        """

        # load conversion/decoding table for ancillary parameters collected onboard
        file_handler = open(anc_input_from_instr_team['conversion_table_for_anc_data'], 'r')
        conversion_table = json.load(file_handler)
        file_handler.close()

        # ancillary parameters collected onboard that need to be decoded
        self.data['filter_temperature_average'], self.data['filter_temperature_std_dev'] = \
            decode_ancillary_parameters_avg_stddev('filter_temperature', conversion_table, l1a_data)
        self.data['hv_voltage_average'], self.data['hv_voltage_std_dev'] = \
            decode_ancillary_parameters_avg_stddev('hv_voltage', conversion_table, l1a_data)
        self.data['spin_period_average'], self.data['spin_period_std_dev'] = \
            decode_ancillary_parameters_avg_stddev('spin_period', conversion_table, l1a_data)
        self.data['pulse_length_average'], self.data['pulse_length_std_dev'] = \
            decode_ancillary_parameters_avg_stddev('pulse_length', conversion_table, l1a_data)

    def create_l1b_from_l1a_data(self, l1a_data, anc_input_from_sdc, anc_input_from_instr_team):
        """
        Sets values of L1b fields using L1a data read from a file
        Args:
            l1a_data: L1a data provided in the form of data field of L1aDataHistogram object
            anc_input_from_sdc: ancillary input provided by IMAP SDC (general telemetry)
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
        """
        self.data['block_header'] = {
            'flight_software_version': l1a_data['block_header']['flight_software_version'],
            'ground_software_version': VERSION,
            'pkts_file_name': l1a_data['block_header']['pkts_file_name'],
            'seq_count_in_pkts_file': l1a_data['block_header']['seq_count_in_pkts_file'],
            'l1a_file_name': l1a_data['l1a_file_name']
        }

        # add list of used ancillary files to block_header
        self._block_header_add_list_of_anc_files(anc_input_from_sdc, anc_input_from_instr_team)

        self.data['unique_block_identifier'] = \
            time_sec_subsec_to_utc(l1a_data[MAIN_TIME_FIELD]['seconds'],
                                   l1a_data[MAIN_TIME_FIELD]['subseconds']
                                  ).strftime('%Y-%m-%dT%H:%M:%S') # TBC: cmp with l1a identifier

        # set IMAP and GLOWS time fields
        self.set_times(l1a_data)

        # set histogram-related quantities
        self.data['number_of_spins_per_block'] = l1a_data['number_of_spins_per_block']
        self.data['number_of_bins_per_histogram'] = l1a_data['number_of_bins_per_histogram']

        # set histogram
        self.data['histogram'] = l1a_data['histogram']
        self.data['number_of_events'] = l1a_data['number_of_events']

        # set spin angle values for bin centers
        spin_offset_first_bin = 0.5*360.0/self.data['number_of_bins_per_histogram']
        self.data['imap_spin_angle_bin_cntr'] = np.array( \
            np.array(range(self.data['number_of_bins_per_histogram']))*360.0/ \
            self.data['number_of_bins_per_histogram'] + spin_offset_first_bin).tolist()

        # this is to make the key 'histogram_flag_array' to be seen immediately after 'histogram'
        self.data['histogram_flag_array'] = None

        # dictionary for some additional debugging data
        self.data['debugging_data'] = {}

        # set block_start_time and block_end_time for funcs invoked in process_anc_input_from_sdc()
        self.block_start_time = self.data[MAIN_TIME_FIELD]
        if MAIN_TIME_FIELD == 'imap_start_time':
            self.block_end_time = self.block_start_time + self.data['imap_end_time_offset']
        if MAIN_TIME_FIELD == 'glows_start_time':
            self.block_end_time = self.block_start_time + self.data['glows_end_time_offset']

        # convert ancillary parameters collected onboard to physical units
        self._convert_ancillary_parameters(l1a_data, anc_input_from_instr_team)

        # invoke functions processing ancillary input from IMAP SDC
        self._process_anc_input_from_sdc(anc_input_from_sdc)

        # deserialize flags generated onboard and save them as self.data['flags'] dictionary
        self.data['flags'] = flags_deserialize(l1a_data['flags']['flags_set_onboard'])

        # add ground-processing flags to self.data['flags']
        for key in l1a_data['flags']:
            if key != 'flags_set_onboard':
                self.data['flags'][key] = l1a_data['flags'][key]

        # initialize histogram bin flags
        self.data['histogram_flag_array'] = np.zeros_like(self.data['histogram'], np.int16)

        # apply filters defined by the IMAP/GLOWS Instrument Team
        self._filters(anc_input_from_instr_team)

        # move 'debuging_data' dictionary to the end of self.data dictionary
        aux = self.data['debugging_data']
        self.data.pop('debugging_data')
        self.data['debugging_data'] = aux

        # uncomment the line below to remove 'debugging_data' from JSON
        del self.data['debugging_data']

    def _filters(self, anc_input_from_instr_team):
        """
        Apply filters with parms defined in settings file (provided by IMAP/GLOWS Instrument Team)
        Args:
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
        """
        # read filter settings
        file_handler = open(anc_input_from_instr_team['settings'], 'r')
        settings = json.load(file_handler)
        file_handler.close()

        # set uv-source-related part of histogram flag array
        self._set_histogram_flag_array_uv_sources(anc_input_from_instr_team, settings)

        # set excluded-regions part of histogram flag array
        self._set_histogram_flag_array_excl_regions(anc_input_from_instr_team, settings)

        # set excluded-by-instrument-team bit in histogram flag array
        self._set_histogram_flag_array_by_instr_team(anc_input_from_instr_team, 'exclusions_by_instr_team')

        # set suspected_transient bit in histogram flag array
        self._set_histogram_flag_array_by_instr_team(anc_input_from_instr_team, 'suspected_transients')

        # apply filter based on daily statistical error and set related flag
        self._flag_based_on_daily_statistical_error(settings)

        # apply filter based on filter-temperature spread and set related flag
        self._flag_compare_with_threshold('filter_temperature_std_dev',
                                          settings['filter_based_on_temperature_std_dev']
                                          ['std_dev_threshold__celsius_deg'],
                                          'is_temperature_std_dev_beyond_threshold')

        # apply filter based on HV-voltage spread and set related flag
        self._flag_compare_with_threshold('hv_voltage_std_dev',
                                          settings['filter_based_on_hv_voltage_std_dev']
                                          ['std_dev_threshold__volt'],
                                          'is_hv_voltage_std_dev_beyond_threshold')

        # apply filter based on spin-period spread and set related flag
        self._flag_compare_with_threshold('spin_period_std_dev',
                                          settings['filter_based_on_spin_period_std_dev']
                                          ['std_dev_threshold__sec'],
                                          'is_spin_period_std_dev_beyond_threshold')

        # apply filter based on pulse-length spread and set related flag
        self._flag_compare_with_threshold('pulse_length_std_dev',
                                          settings['filter_based_on_pulse_length_std_dev']
                                          ['std_dev_threshold__usec'],
                                          'is_pulse_length_std_dev_beyond_threshold')

        # apply filter based on relative difference between onboard and ground spin periods
        self._flag_relative_difference_vs_threshold(
            'spin_period_average',
            'spin_period_ground_average',
            settings['filter_based_on_comparison_of_spin_periods']
            ['relative_difference_threshold'],
            'is_spin_period_difference_beyond_threshold')

        # set a fancy pattern 0x5555 in histogram_flag_array
        # this is for testing purposes, uncomment only if know exactly what you are doing
        #self._set_histogram_flag_array_test_pattern()

    def _flag_based_on_daily_statistical_error(self, settings):
        """
        Set flag based on daily statistical error, set True if the total number of counts in
            the current block-accumulated histogram is beyond n_sigma * sigma
        Args:
            settings: filter settings from a file provided by IMAP/GLOWS Instrument Team
        """
        n_lo = settings['filter_based_on_daily_statistical_error']['n_sigma_threshold_lower']
        n_up = settings['filter_based_on_daily_statistical_error']['n_sigma_threshold_upper']
        if n_lo < 0 or n_up < 0:
            # negative value means that filter is turned off, so we can immediately set the flag
            self.data['flags']['is_beyond_daily_statistical_error'] = False
        else:
            # filter is turned on, so we need to compare the total number of counts in the current
            # histogram with thresholds and set the flag accordingly
            total_counts = np.sum(self.data['histogram'])
            self.data['debugging_data']['total_counts_in_histogram'] = int(total_counts)
            total_counts_daily_average, total_counts_daily_std_dev = \
                self.total_counts_daily_statistics()
            self.data['debugging_data']['total_counts_daily_average'] = total_counts_daily_average
            self.data['debugging_data']['total_counts_daily_std_dev'] = total_counts_daily_std_dev
            if total_counts_daily_average - n_lo*total_counts_daily_std_dev <= total_counts <= \
               total_counts_daily_average + n_up*total_counts_daily_std_dev:
                self.data['flags']['is_beyond_daily_statistical_error'] = False
            else:
                self.data['flags']['is_beyond_daily_statistical_error'] = True

    def _flag_compare_with_threshold(self, parameter_name, threshold, flag_name):
        """
        Set flag based on comparison of a parameter vs. threshold, set True if beyond threshold
        Args:
            parameter_name: dictionary key to identify parameter of interest in self.data dictionary
            threshold: threshold value
            flag_name: name of the flag to be set
        """
        if threshold < 0.0:
            # negative value means that filter is turned off, so we can immediately set the flag
            self.data['flags'][flag_name] = False
        else:
            # filter is turned on, so we need to compare the parameter value in the current
            # histogram with threshold and set the flag accordingly
            if self.data[parameter_name] < threshold:
                self.data['flags'][flag_name] = False
            else:
                self.data['flags'][flag_name] = True

    def _flag_relative_difference_vs_threshold(self, parameter_1_name, parameter_2_name, threshold,
                                               flag_name):
        """
        Set flag based on comparison of relative difference of two parameters vs. threshold,
            set True if beyond threshold
        Args:
            parameter_1_name: dictionary key to identify parameter 1 of interest in self.data
            parameter_2_name: dictionary key to identify parameter 2 of interest in self.data
            threshold: threshold value
            flag_name: name of the flag to be set
        """
        if threshold < 0.0:
            # negative value means that filter is turned off, so we can immediately set the flag
            self.data['flags'][flag_name] = False
        else:
            # filter is turned on, so we need to compare the relative difference in the current
            # histogram with threshold and set the flag accordingly
            relative_difference = np.fabs(self.data[parameter_1_name] - \
                                          self.data[parameter_2_name])/ \
                                          (self.data[parameter_1_name] + \
                                           self.data[parameter_2_name])
            self.data['debugging_data']['spin_period_relative_difference'] = relative_difference
            if relative_difference < threshold:
                self.data['flags'][flag_name] = False
            else:
                self.data['flags'][flag_name] = True

    def _process_anc_input_from_sdc(self, anc_input_from_sdc):
        """
        Processing for ancillary input from IMAP SDC (included in general IMAP telemetry)
        Args:
            anc_input_from_sdc: ancillary input provided by IMAP SDC (general telemetry)
        """

        # read ancillary variables provided by SDC (IMAP general telemetry)
        # note that finally different CSV files can be used for different quantities
        anc_data = pd.read_csv(anc_input_from_sdc['anc_data_file_name'], sep=' ', comment="#")
        # TMP_SOLUTION: we use here a csv file with mock ancillary data generated by
        # GLOWS Team using generate_anc_file.py script
        # the following procedures invoked below:
        # compute_spin_period_ground_estimate(), compute_position_angle(),
        # compute_spin_axis_orientation(), compute_spacecraft_location(),
        # compute_spacecraft_velocity()
        # need to be thoroughly updated when ancillary data from SDC will be available
        # e.g., it is expected that finally there will be several anc_data files instead
        # of one containing all required data

        self.data['spin_period_ground_average'], self.data['spin_period_ground_std_dev'] = \
            self._compute_spin_period_ground_estimate(anc_data)
        self.data['position_angle_offset_average'], self.data['position_angle_offset_std_dev'] = \
            self._compute_position_angle(anc_data)
        self.data['spin_axis_orientation_average'], self.data['spin_axis_orientation_std_dev'] = \
            self._compute_spin_axis_orientation(anc_data)
        self.data['spacecraft_location_average'], self.data['spacecraft_location_std_dev'] = \
            self._compute_spacecraft_location(anc_data)
        self.data['spacecraft_velocity_average'], self.data['spacecraft_velocity_std_dev'] = \
            self._compute_spacecraft_velocity(anc_data)

    def read_from_file(self, file_name):
        """
        Read L1b data from file
        Args:
            file_name: name of of file with L1b data
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        #    modules for IMAP will be provided by SDC
        self._read_from_json_file(file_name)

    def _read_from_json_file(self, file_name):
        """
        Read L1b data from JSON file
        Args:
            file_name: name of of file with L1b data
        """
        file_handler = open(file_name, 'r')
        self.data = json.load(file_handler)
        file_handler.close()

        # hexadecimal values saved as strings require special treatment to make them integers
        self.data['histogram_flag_array'] = \
            [int(x, base=16) for x in self.data['histogram_flag_array']]

        # additional field to keep track where data come from
        self.data['l1b_file_name'] = file_name

    def save_data_to_file(self):
        """
        Save generated L1b data to a file
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        # modules for IMAP will be provided by SDC
        self._save_data_to_json_file()

    def _save_data_to_json_file(self):
        """
        Save generated L1b data to JSON file
        Output file name is set automatically here (TBC if perhaps it should be method argument)
        """
        # define formats for printing values to JSON file
        fmt = {
            'arrays': {
                'histogram': '%d',
                'imap_spin_angle_bin_cntr': '%.2f',
            },
            'scalar variables': {
                'glows_start_time': '%.7f',
                'glows_end_time_offset': '%.7f',
                'imap_start_time': '%.7f',
                'imap_end_time_offset': '%.7f',
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
        for key in fmt['arrays'].keys():
            data[key] = '[' + ', '.join(map(lambda x: fmt['arrays'][key] % x, list(data[key]))) + ']'
        for key in fmt['scalar variables'].keys():
            data[key] = '|'+ fmt['scalar variables'][key] % data[key] +'|'
        for key in fmt['lon lat variables'].keys():
            for sec_key in ['lon', 'lat']:
                data[key][sec_key] = '|'+ fmt['lon lat variables'][key] % data[key][sec_key] +'|'
        for key in fmt['vector variables'].keys():
            for sec_key in ['x', 'y', 'z']:
                data[key][sec_key] = '|'+ fmt['vector variables'][key] % data[key][sec_key] +'|'

        # histogram_flag_array requires special hexadecimal treatment
        data['histogram_flag_array'] = '[' + \
            ', '.join(map(lambda x: '"0x' + format(x, 'x').zfill(4) +'"',
            list(data['histogram_flag_array']))) + ']'

        # generate JSON content to be saved in output file
        json_content = json.dumps(data, indent=4, default=vars)

        # some corrections to get numbers back
        json_content = json_content.replace('"[', '[').replace(']"', ']')
        json_content = json_content.replace('|"', '').replace('"|', '')
        json_content = json_content.replace('\\"', '"')

        if 1==2:
            # some fields need to be temporarily converted to string arrays, because we want to force
            # them to be saved as one-liners (i.e., without \n after each element) by json.dumps()
            spin_angle_tmp = self.data['imap_spin_angle_bin_cntr']
            histogram_tmp = self.data['histogram']
            flags_tmp = self.data['histogram_flag_array']
            self.data['histogram'] = '[' + ', '.join(map(str, list(histogram_tmp))) + ']'
            self.data['imap_spin_angle_bin_cntr'] = \
                '[' + ', '.join(map(lambda x: '%.2f' % x, list(spin_angle_tmp))) + ']'
            self.data['histogram_flag_array'] = '[' + \
                ', '.join(map(lambda x: '"0x' + format(x, 'x').zfill(4) +'"', list(flags_tmp))) + ']'
            json_content = json.dumps(self.data, indent=4, default=vars)
            self.data['imap_spin_angle_bin_cntr'] = spin_angle_tmp
            self.data['histogram'] = histogram_tmp
            self.data['histogram_flag_array'] = flags_tmp
            # modify json_content to make string arrays be seen finally as number arrays
            json_content = json_content.replace('"[', '[').replace(']"', ']')
            json_content = json_content.replace('\\"', '"')

        # start_time_utc string for output file name, based on unique_block_identifier
        start_time_utc = ''.join(filter(str.isdigit, self.data['unique_block_identifier']))

        # the output file name is constructed using a convention defined in Sec. 1.4.2 of
        # "IMAP SDC to Instrument Team ICD"
        file_name = 'data_l1b_histograms/imap_glows_l1b_%s_orbX_modX_p_v00.json' % \
                    (start_time_utc)

        # save data
        file_handler = open(file_name, 'w')
        print(json_content, file=file_handler)
        file_handler.close()

        # TO_BE_REMOVED append file name to l1b list needed for the l1b-to-l2 processing
        #file_handler = open('l1b_list.dat', 'a')
        #print(file_name, file=file_handler)
        #file_handler.close()

    def _set_histogram_flag_array_by_instr_team(self, anc_input_from_instr_team, type_of_list):
        """
        Set histogram flag array using lists defined by IMAP/GLOWS Instrument Team
            bin indexes in the list are marked by True
        Args:
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
            type_of_list: 'exclusions_by_instr_team' or 'suspected_transients'
        """
        file_handler = open(anc_input_from_instr_team[type_of_list], 'r')
        line = file_handler.readline()
        while line:
            if line[0] != '#':
                # sanity check
                assert line[:2] == '20' and line[10] == 'T', \
                    'Bad identifier ' + line[:20] + ' in ' + \
                    anc_input_from_instr_team[type_of_list]
                identifier = line.rstrip().split(' ')[0]
                flags = line.rstrip().split(' ')[1]

                # sanity check
                # this check turned out to be inappropriate because number of bins is not always the same
                #assert len(flags) == self.data['number_of_bins_per_histogram'], \
                #    'Bad length of mask_array for identifier ' + identifier + ' in ' + \
                #    anc_input_from_instr_team[type_of_list]

                if identifier == self.data['unique_block_identifier']:
                    flags = np.array(list(flags)).astype(bool)
#                    print(np.shape(flags))
                    if type_of_list == 'exclusions_by_instr_team':
                        flags = np.left_shift(flags, L1B_L2_HIST_FLAG_FIELDS['is_excluded_by_instr_team'])
                    if type_of_list == 'suspected_transients':
                        flags = np.left_shift(flags, L1B_L2_HIST_FLAG_FIELDS['is_suspected_transient'])
                    self.data['histogram_flag_array'] = self.data['histogram_flag_array'] | flags
            line = file_handler.readline()
        file_handler.close()

    def _set_histogram_flag_array_excl_by_instr_team(self, anc_input_from_instr_team):
        """
        Set histogram flag array using exclusion list defined by IMAP/GLOWS Instrument Team
            bin indexes in the list are marked by True
        Args:
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
        """
        file_handler = open(anc_input_from_instr_team['exclusions_by_instr_team'], 'r')
        line = file_handler.readline()
        while line:
            if line[0] != '#':
                # sanity check
                assert line[:2] == '20' and line[10] == '_', \
                    'Bad identifier ' + line[:20] + ' in ' + \
                    anc_input_from_instr_team['exclusions_by_instr_team']
                identifier = line.rstrip().split(' ')[0]
                flags = line.rstrip().split(' ')[1]

                # sanity check
                assert len(flags) == self.data['number_of_bins_per_histogram'], \
                    'Bad length of mask_array for identifier ' + identifier + ' in ' + \
                    anc_input_from_instr_team['exclusions_by_instr_team']

                if identifier == self.data['unique_block_identifier']:
                    flags = np.array(list(flags)).astype(bool)
                    flags = np.left_shift(flags, L1B_L2_HIST_FLAG_FIELDS['is_excluded_by_instr_team'])
                    self.data['histogram_flag_array'] = self.data['histogram_flag_array'] | flags
            line = file_handler.readline()
        file_handler.close()

    def _set_histogram_flag_array_uv_sources(self, anc_input_from_instr_team, settings):
        """
        Set histogram flag array using a map of bright UV sources, close-to-map-points bins are
            denoted by True
        Args:
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
            settings: filter settings
        """

        if check_if_contains_actual_data(anc_input_from_instr_team['map_of_uv_sources']):
            # read ecliptic coords and angular radiuses for the masking of bright uv point sources
            uv_sources_coords_and_radiuses = np.loadtxt(anc_input_from_instr_team['map_of_uv_sources'],
                                               usecols=[1, 2, 3])

            # compute ecliptic coords (lon, lat) for histogram bin centers
            histogram_bins_ecl_coords = self._compute_histogram_bins_ecl_coords()

            flag = np.zeros_like(self.data['histogram'], bool)

            # a radius_bins grouping is used below for code speed up, because otherwise we would need to use
            # coord1.separation(coord2) from astropy, which turned out to be slow in tests
            n_radius_bins = 10 # actually used bins, see +2 in np.linspace() argument below
            radius_bins = np.linspace(uv_sources_coords_and_radiuses[:, 2].min()-1.0e-6,
                                      uv_sources_coords_and_radiuses[:, 2].max()+1.0e-6, n_radius_bins+2)
            # digitizing/binning into radius_bins
            radius_idxs = np.digitize(uv_sources_coords_and_radiuses[:, 2], bins=radius_bins)

            for i in range(1, n_radius_bins+2): # looping over all radius_bins groups
                rejection_radius = radius_bins[i]
                idxs = np.nonzero(radius_idxs == i)[0]
                if len(idxs) > 0: # check if we have any uv sources to check in the current group
                    match_catalog = SkyCoord(uv_sources_coords_and_radiuses[idxs, 0], uv_sources_coords_and_radiuses[idxs, 1],
                                             frame='geocentrictrueecliptic', unit='deg')
                    search_catalog = SkyCoord(histogram_bins_ecl_coords[:, 0],
                                              histogram_bins_ecl_coords[:, 1],
                                              frame='geocentrictrueecliptic', unit='deg')

                    # find bin vs. map point pairs that are closer than rejection_radius
                    idxs_match, __, __, __ = \
                        match_catalog.search_around_sky(search_catalog, rejection_radius*u.deg)
                    # find indexes of bins that are too close to map points
                    idxs_to_reject = np.unique(idxs_match)
                    # set histogram bin flags
                    flag[idxs_to_reject] = True

            flag = np.left_shift(flag, L1B_L2_HIST_FLAG_FIELDS['is_close_to_uv_source'])
            self.data['histogram_flag_array'] = self.data['histogram_flag_array'] | flag

    def _set_histogram_flag_array_excl_regions(self, anc_input_from_instr_team, settings):
        """
        Set histogram flag array using a map of excluded regions, close-to-map-points bins are
            denoted by True
        Args:
            anc_input_from_instr_team: ancillary input provided by IMAP/GLOWS Instrument Team
            settings: filter settings
        """

        if check_if_contains_actual_data(anc_input_from_instr_team['map_of_excluded_regions']):
            # read ecliptic coords of points defining excluded regions
            excluded_pts_coords = np.loadtxt(anc_input_from_instr_team['map_of_excluded_regions'])

            # set angular radius below which bins are too close to map points
            rejection_radius = settings['filter_based_on_maps'] \
                                       ['angular_radius_for_excl_regions__deg']

            if rejection_radius <= 0.0:
                # negative or zero value of rejection_radius means that the filter is turned off and we
                # can immediately set False value in the histogram_flag_array for all bins
                flag = np.zeros_like(self.data['histogram'], bool)
            else:
                # positive value of rejection_radius means that the filter is turned on and we must
                # compare bin positions in the sky with map-point positions to set True/False values
                # properly for the histogram_flag_array

                # compute ecliptic coords (lon, lat) for histogram bin centers
                histogram_bins_ecl_coords = self._compute_histogram_bins_ecl_coords()

                search_catalog = SkyCoord(excluded_pts_coords[:, 0], excluded_pts_coords[:, 1],
                                          frame='geocentrictrueecliptic', unit='deg')
                match_catalog = SkyCoord(histogram_bins_ecl_coords[:, 0],
                                         histogram_bins_ecl_coords[:, 1],
                                         frame='geocentrictrueecliptic', unit='deg')

                # find bin vs. map point pairs that are closer than rejection_radius
                __, idxs_match, __, __ = \
                    match_catalog.search_around_sky(search_catalog, rejection_radius*u.deg)
                # find indexes of bins that are too close to map points
                idxs_to_reject = np.unique(idxs_match)
                # set histogram bin flags
                flag = np.zeros_like(self.data['histogram'], bool)
                flag[idxs_to_reject] = True

            flag = np.left_shift(flag, L1B_L2_HIST_FLAG_FIELDS['is_inside_excluded_region'])
            self.data['histogram_flag_array'] = self.data['histogram_flag_array'] | flag

    def _set_histogram_flag_array_test_pattern(self):
        """
        Set a specific pattern for histogram_flag_array for unit-test purposes
        """
        for i in range(15):
            if i % 2 == 0:
                flag = np.ones_like(self.data['histogram'], bool)
            else:
                flag = np.zeros_like(self.data['histogram'], bool)
            flag = np.left_shift(flag, i)
            self.data['histogram_flag_array'] = self.data['histogram_flag_array'] | flag

    def set_times(self, l1a_data):
        """
        Set times
        Args:
            l1a_data: L1a data provided in the form of data field of L1aDataHistogram object
        """
        self.data['glows_start_time'] = \
            time_sec_subsec_to_float64(l1a_data['glows_start_time']['seconds'],
                                       l1a_data['glows_start_time']['subseconds'])
        self.data['glows_end_time_offset'] = \
            time_sec_subsec_to_float64(l1a_data['glows_end_time_offset']['seconds'],
                                       l1a_data['glows_end_time_offset']['subseconds'])
        self.data['imap_start_time'] = \
            time_sec_subsec_to_float64(l1a_data['imap_start_time']['seconds'],
                                       l1a_data['imap_start_time']['subseconds'])
        self.data['imap_end_time_offset'] = \
            time_sec_subsec_to_float64(l1a_data['imap_end_time_offset']['seconds'],
                                       l1a_data['imap_end_time_offset']['subseconds'])

    def total_counts_daily_statistics(self):
        """
        Compute daily average for the total number of counts in L1a histograms
        """
        file_name = 'data_l1a_histograms/total_counts_file.dat'
        file_handler = open(file_name, 'r')
        line = file_handler.readline()

        # sanity check
        assert line.rstrip().split(' ')[1] == self.data['block_header']['pkts_file_name'], \
            'Packets file name ' + line.rstrip().split(' ')[1] + ' in ' + file_name + \
            ' is different from currently processed packets file ' + \
            self.data['block_header']['pkts_file_name']

        # the first column of data array below: 'seq_count_in_pkts_file'
        # the second column: total counts in a L1a histogram
        data = np.loadtxt(file_handler)
        file_handler.close()
        total_counts_average = np.mean(data[:, 1])
        total_counts_std_dev = np.std(data[:, 1])
        return total_counts_average, total_counts_std_dev
