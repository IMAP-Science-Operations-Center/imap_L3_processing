"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
"""

class L0DataDirectEvents():
    """
    L0DataDirectEvents class for GLOWS-L0-direct-events data
    """
    def __init__(self):
        """
        Constructor for this class
        """
        # keys in self.data dictionary are based on sheet "P_GLX_TMSCDE" in
        # doc/TLM_GLX_2023_06_22.xlsx, see deserialize() method
        self.data = {}

    def deserialize(self, reader, packet_length):
        """
        Read the after-CCSDS-header part of the direct-events packet and decode information in
            its bit structure
        Args:
            reader: Reader-class object needed for reading binary data stream
        """
        self.data['imap_start_time_second'] = reader.read_uint32_be()
        self.data['number_of_de_packets'] = reader.read_uint16_be()
        self.data['seq_num_of_de_packet'] = reader.read_uint16_be()

        # TMP_SOLUTION
        # read 4078 bytes if zero-padding is used
        #self.data['de_data'] = bytearray(reader.read_buffer(4078))

        # TMP_SOLUTION
        # read packet_length-with-correction bytes if zero-padding is not used
        # + 1 see definition of CCSDS header, -4 sclk, -4 - 2 - 2 if for the three reads above
        packet_length_correction = 1 - 4 - 4 - 2 - 2
        assert (packet_length + packet_length_correction) > 0, \
            'wrong packet length for direct-event packet'
        self.data['de_data'] = bytearray(reader.read_buffer(packet_length + packet_length_correction))

        # DEBUG TEST
        #print()
        #print(len(self.data['de_data']))
        #import binascii
        #print(binascii.hexlify(self.data['de_data']))
        #print(' '.join(format(x, '02x') for x in self.data['de_data'][:40]))
        #print(' '.join(format(x, '02x') for x in self.data['de_data'][40:]))
        #if len(self.data['de_data']) != 40:
        #  print('\nerror: len(self.data[\'de_data\']) != 40')
        #  exit()
        #return self.data

    def print_data_keys(self):
        """
        Print data keys for L0 data
        """
        print('\n\nL0 data keys:')
        for key in self.data:
            print(key)
