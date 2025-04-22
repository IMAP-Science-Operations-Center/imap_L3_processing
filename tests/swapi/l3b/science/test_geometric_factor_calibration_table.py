import unittest

from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from tests.test_helpers import get_test_data_path


class TestGeometricFactorTable(unittest.TestCase):
    def test_geometric_factor_table_from_file(self):
        file_path = get_test_data_path("swapi/imap_swapi_energy-gf-lut_20240923_v000.dat")

        calibration_table = GeometricFactorCalibrationTable.from_file(file_path)

        self.assertEqual(62, len(calibration_table.grid))
        self.assertEqual((62,), calibration_table.geometric_factor_grid.shape)

        known_energy = 8165.393844536367
        energy_to_interpolate = 14194.87288073211
        self.assertEqual(6.419796603112413e-13, calibration_table.lookup_geometric_factor(known_energy))
        self.assertAlmostEqual(5.711128783363629e-13, calibration_table.lookup_geometric_factor(energy_to_interpolate))
