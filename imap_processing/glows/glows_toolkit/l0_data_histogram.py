"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""

class L0DataHistogram():
    """
    L0DataHistogram class (based on HistogramPayload by KPLabs) for GLOWS-L0-histogram data
    """
    def __init__(self):
        """
        Constructor for this class
        """
        # keys in self.data dictionary are based on sheet "P_GLX_TMSCHIST" in
        # doc/TLM_GLX_2023_06_22.xlsx, see deserialize() method below for details
        self.data = {}

    def deserialize(self, reader):
        """
        Read the after-CCSDS-header part of the histogram packet and decode information in
            its bit structure
        Args:
            reader: Reader-class object needed for reading binary data stream
        """
        # start reading the data
        self.data['first_spin_id_in_block'] = reader.read_uint32_be()
        self.data['diff_spin_id_in_block'] = reader.read_uint16_be()
        self.data['histogram_validity_flags'] = reader.read_uint16_be()
        self.data['software_version'] = reader.read_uint24_be() # 11

        # IMAP time
        self.data['imap_start_time_second'] = reader.read_uint32_be()
        self.data['imap_start_time_subsecond'] = reader.read_uint24_be()
        self.data['imap_diff_second'] = reader.read_uint16_be()
        self.data['imap_diff_subsecond'] = reader.read_uint24_be() # 23

        # GLOWS time
        self.data['glows_start_time_second'] = reader.read_uint32_be()
        self.data['glows_start_time_subsecond'] = reader.read_uint24_be()
        self.data['glows_diff_second'] = reader.read_uint16_be()
        self.data['glows_diff_subsecond'] = reader.read_uint24_be() # 35

        # histogram parameters
        self.data['number_of_spins_per_block'] = reader.read_uint8() + 1
        self.data['number_of_bins_per_histogram'] = reader.read_uint16_be() # 38

        # ancillary parameters collected onboard
        self.data['filter_temperature_average'] = reader.read_uint8()
        self.data['filter_temperature_variance'] = reader.read_uint16_be()
        self.data['hv_voltage_average'] = reader.read_uint16_be()
        self.data['hv_voltage_variance'] = reader.read_uint32_be()
        self.data['spin_period_average'] = reader.read_uint16_be()
        self.data['spin_period_variance'] = reader.read_uint32_be()
        self.data['pulse_length_average'] = reader.read_uint8()
        self.data['pulse_length_variance'] = reader.read_uint16_be() # 56

        # TMP_SOLUTION
        # in TLM_GLX_2023_06_22.xls a new field (number of events in histogram) was introduced
        # the line below needs to be commented out, if earlier versions of mock data are used
        self.data['number_of_events'] = reader.read_uint32_be()

        # histogram array
        self.data['histogram'] = bytearray(reader.read_buffer(
            self.data['number_of_bins_per_histogram']))

        # additional padding when odd number of bins?
        if self.data['number_of_bins_per_histogram'] % 2 == 1:
          aux = reader.read_uint8()

        return self.data

    def print_data_keys(self):
        """
        Print data keys for L0 data
        """
        print('\n\nL0 data keys:')
        for key in self.data:
            print(key)
