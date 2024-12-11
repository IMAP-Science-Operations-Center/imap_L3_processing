"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""
import json
import numpy as np
from .constants import VERSION, MAIN_TIME_FIELD
from .time import Time
from .funcs import time_sec_subsec_to_utc

class L1aDataHistogram():
    """
    L1aDataHistogram() class for GLOWS-L1a-histogram data
    """

    def __init__(self):
        """
        Constructor for this class
        """
        self.data = {}

    def create_l1a_from_l0_data(self, l0_data, pkts_file_name: str, packet_number: int):
        """
        Sets values of L1a fields using L0 data read from file with CCSDS packets
        Args:
            l0_data: L0 data provided in the form of L0DataHistogram object
            pkts_file_name: name of of the file with CCSDS packets
            packet_number: CCSDS packet sequence count (per APID)
        """

        # this is supposed to carry information about errors to be passed up at return
        error_messages = []

        self.data['first_spin_id_in_block'] = l0_data['first_spin_id_in_block']
        self.data['last_spin_id_in_block'] = self.data['first_spin_id_in_block'] + \
                                              l0_data['diff_spin_id_in_block']

        # sanity check
        # TMP_SOLUTION: instrument emulator does not set l0_data['diff_spin_id_in_block'] properly
        # thus we temporarily comment out this assert
        #assert l0_data['diff_spin_id_in_block'] == l0_data['number_of_spins_per_block'], \
        #    'Inconsistency between spin-numbering field l0_data[\'diff_spin_id_in_block\'] and' + \
        #    'histogram-parameter field l0_data[\'number_of_spins_per_block\']'

        self.data['block_header'] = {
            'flight_software_version': l0_data['software_version'],
            'ground_software_version': VERSION,
            'pkts_file_name': pkts_file_name,
            # note: packet number is seq_count (per apid!) field in CCSDS header
            'seq_count_in_pkts_file': packet_number
        }

        # times for L1a are kept in the same format as for L0 (i.e., using seconds and subseconds)
        self.data['glows_start_time'] = Time(l0_data['glows_start_time_second'],
                                             l0_data['glows_start_time_subsecond'])
        self.data['glows_end_time_offset'] = Time(l0_data['glows_diff_second'],
                                                  l0_data['glows_diff_subsecond'])
        self.data['imap_start_time'] = Time(l0_data['imap_start_time_second'],
                                            l0_data['imap_start_time_subsecond'])
        self.data['imap_end_time_offset'] = Time(l0_data['imap_diff_second'],
                                                 l0_data['imap_diff_subsecond'])

        self.data['number_of_spins_per_block'] = l0_data['number_of_spins_per_block']
        self.data['number_of_bins_per_histogram'] = l0_data['number_of_bins_per_histogram']
        self.data['histogram'] = l0_data['histogram']
        # TMP_SOLUTION
        self.data['number_of_events'] = l0_data['number_of_events']

        # sanity check
        if self.data['number_of_events'] != np.sum(self.data['histogram']):
          error_messages.append('HIST: number of events is not equal to the sum over histogram %d %d' % \
                                (self.data['number_of_events'], np.sum(self.data['histogram'])))

        self.data['filter_temperature_average'] = l0_data['filter_temperature_average']
        self.data['filter_temperature_variance'] = l0_data['filter_temperature_variance']
        self.data['hv_voltage_average'] = l0_data['hv_voltage_average']
        self.data['hv_voltage_variance'] = l0_data['hv_voltage_variance']
        self.data['spin_period_average'] = l0_data['spin_period_average']
        self.data['spin_period_variance'] = l0_data['spin_period_variance']
        self.data['pulse_length_average'] = l0_data['pulse_length_average']
        self.data['pulse_length_variance'] = l0_data['pulse_length_variance']

        self.data['flags'] = {}
        self.data['flags']['flags_set_onboard'] = l0_data['histogram_validity_flags']
        self.data['flags']['is_generated_on_ground'] = False

        return error_messages

    def read_from_file(self, file_name):
        """
        Read L1a data from file
        Args:
            file_name: name of file with L1a data
        """
        # TMP_SOLUTION currently JSON format is used, it needs to be changed to CDF when python-cdf
        #    modules for IMAP will be provided by SDC
        self.read_from_json_file(file_name)

    def read_from_json_file(self, file_name):
        """
        Read L1a data from file
        Args:
            file_name: name of file with L1a data
        """
        file_handler = open(file_name, 'r')
        self.data = json.load(file_handler)

        # hexadecimal values saved as strings require special treatment to make them integers
        self.data['flags']['flags_set_onboard'] = int(self.data['flags']['flags_set_onboard'],
                                                      base=16)

        # additional field to keep track where data come from
        self.data['l1a_file_name'] = file_name
        file_handler.close()

    def save_data_to_file(self):
        """
        Save generated L1a data to a file
        """
        # TMP_SOLUTION Currently JSON format is used, it needs to be changed to CDF when python-cdf
        # modules for IMAP will be provided by SDC
        self.save_data_to_json_file()

    def save_data_to_json_file(self):
        """
        Save generated L1a data to JSON file
        Output file name is set automatically here (TBC if perhaps it should be method argument)
        """
        # histogram needs to be temporarily converted to string, because we want to force that it
        # is saved as one line (i.e., without \n after each element) by json.dumps()
        histogram_tmp = self.data['histogram']
        flag_tmp = self.data['flags']['flags_set_onboard']
        self.data['histogram'] = '[' + ', '.join(map(str, list(histogram_tmp))) + ']'
        self.data['flags']['flags_set_onboard'] = '0x' + format(flag_tmp, 'x').zfill(4)
        json_content = json.dumps(self.data, indent=4, default=vars)
        self.data['flags']['flags_set_onboard'] = flag_tmp
        self.data['histogram'] = histogram_tmp
        # modify json_content to make histogram string be seen finally as a number array
        json_content = json_content.replace('"[', '[').replace(']"', ']')

        # file names should be tagged with utc time as suggested in Sec. 1.4.2 of
        # "IMAP SDC to Instrument Team ICD"
        start_time_utc = time_sec_subsec_to_utc(self.data[MAIN_TIME_FIELD].seconds,
                                                self.data[MAIN_TIME_FIELD].subseconds
                                               ).strftime('%Y%m%d%H%M%S')

        # the output file name is constructed using a convention defined in Sec. 1.4.2 of
        # "IMAP SDC to Instrument Team ICD"
        file_name = 'data_l1a_histograms/imap_glows_l1a_%s_orbX_modX_p_v00.json' % \
                    (start_time_utc)

        file_handler = open(file_name, 'w')
        print(json_content, file=file_handler)
        file_handler.close()

    def total_counts_file_init(self, tag):
        """
        Initialize a file for storing total count numbers for L1a histograms
        This kind of information is needed when L1b is generated (for one of the filters)
        """
        file_handler = open('data_l1a_histograms/total_counts_file.dat', 'w')
        file_handler.write('%s %s\n' % (tag, self.data['block_header']['pkts_file_name']))
        file_handler.close()

    def total_counts_file_update(self):
        """
        Update a file for storing total counts numbers for L1a histograms
        """
        file_handler = open('data_l1a_histograms/total_counts_file.dat', 'a')
        file_handler.write('%d %d\n' % (self.data['block_header']['seq_count_in_pkts_file'],
                                        int(np.sum(self.data['histogram']))))
        file_handler.close()
