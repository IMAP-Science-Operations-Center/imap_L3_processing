import unittest

from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTableCollection
from tests.test_helpers import get_test_data_path


class TestInstrumentResponseLookupTable(unittest.TestCase):
    def test_get_table_for_energy_bin(self):
        lut_zip_path = get_test_data_path('../swapi/test_data/truncated_swapi_response_simion_v1.zip')
        lut = InstrumentResponseLookupTableCollection.from_file(lut_zip_path)
        energy_bin_index = 2
        result = lut.get_table_for_energy_bin(energy_bin_index)
        self.assertEqual(103.07800, result.energy[0])
        self.assertEqual(107.04900, result.energy[-1])
        self.assertEqual(2.000, result.elevation[0])
        self.assertEqual(6.000, result.elevation[-1])
        self.assertEqual(-149.000, result.azimuth[0])
        self.assertEqual(-149.000, result.azimuth[-1])
        self.assertEqual(0.97411, result.d_energy[0])
        self.assertEqual(1.01163, result.d_energy[-1])
        self.assertEqual(1, result.d_elevation[0])
        self.assertEqual(1, result.d_elevation[-1])
        self.assertEqual(1, result.d_azimuth[0])
        self.assertEqual(1, result.d_azimuth[-1])
        self.assertEqual(0.0160000000, result.response[0])
        self.assertEqual(0.1250000000, result.response[-1])

        expected_row_count = 16
        self.assertEqual(expected_row_count, len(result.energy))


if __name__ == '__main__':
    unittest.main()
