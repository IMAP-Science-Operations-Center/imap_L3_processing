"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
Class for CCSDS headers
"""

class CCSDSHeader():
    """
    Class for CCSDS headers
    """

    def __init__(self):
        """
        Constructor for this class
        """
        # keys in data dictionary are based on packet definitions in
        # doc/TLM_GLX_2023_06_22.xlsx, see deserialize() method
        self.data = {}

    def deserialize(self, reader):
        """
        Read the header and decode information in its bit structure
        Args:
            reader: Reader-class object needed for reading binary data stream
        """
        aux = reader.read_uint16_be()
        if reader.overrun:
            return
        bit_seq_as_string = format(aux, 'b').zfill(16)
        #print('first', bit_seq_as_string)
        self.data['ccsds_ver'] = int(bit_seq_as_string[:3], 2)
        assert self.data['ccsds_ver'] == 0, \
            'Field ccsds_ver in CCSDS header has incorrect value' + ' %d' % self.data['ccsds_ver']
        self.data['pkt_type'] = bool(int(bit_seq_as_string[3]))
        self.data['sec_header_flag'] = bool(int(bit_seq_as_string[4]))
        self.data['apid'] = int(bit_seq_as_string[5:], 2)

        #print(bit_seq_as_string, end=' ')
        assert self.data['apid'] in [0x5a1, 0x5a2, 0x5a3, 0x5a4, 0x5a5, 0x5a8, 0x5a9, 0x5b1, 0x5c8, 0x5c9], \
          'error deserialize(): wrong apid %d 0x%x' % (self.data['apid'], self.data['apid'])

        aux = reader.read_uint16_be()
        bit_seq_as_string = format(aux, 'b').zfill(16)
        #print('second', bit_seq_as_string)
        self.data['grouping_flag'] = int(bit_seq_as_string[:2], 2)
        self.data['seq_count'] = int(bit_seq_as_string[2:], 2)

        #print(bit_seq_as_string, end=' ')

        self.data['packet_length'] = reader.read_uint16_be()
        #print('third', format(self.data['packet_length'], 'b').zfill(16))

        #print(self.data['apid'], self.data['packet_length'], 0x5a1)

        #bit_seq_as_string = format(self.data['packet_length'], 'b').zfill(16)
        #print(bit_seq_as_string)

#        if self.data['packet_length'] == 289:
#          self.data['packet_length'] = 288
#        if self.data['apid'] == 0x5c8 and self.data['packet_length'] == 289:
#          print('error deserialize(): bad histogram', self.data['apid'], self.data['packet_length'])
#          exit()

        if not reader.overrun:
            assert self.data['packet_length'] >= 3 and self.data['packet_length'] <= 4089, \
                'Field packet_length in CCSDS header has incorrect value' + \
                ' %d' % self.data['packet_length']

        # SCLK/MET field is expected to be included in every CCSDS packet for GLOWS so it is
        # safe to always read it
        self.data['sclk_seconds'] = reader.read_uint32_be()
        #print(self.data['sclk_seconds'])

    def print_data_keys(self):
        """
        Print data keys for CCSDS header
        """
        print('\n\nCCSDS header keys:')
        for key in self.data:
            print(key)
