import unittest

from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from tests.test_helpers import get_test_data_path


class TestInflowVector(unittest.TestCase):
    def test_from_file(self):
        vector = InflowVector.from_file(get_test_data_path("swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"))
        self.assertIsInstance(vector, InflowVector)
        self.assertEqual(22, vector.speed_km_per_s)
        self.assertEqual(252.2, vector.longitude_deg_eclipj2000)
        self.assertEqual(9.0, vector.latitude_deg_eclipj2000)

    def test_raises_parse_error(self):
        with self.assertRaises(AssertionError) as exc_ctx:
            InflowVector.from_file(
                get_test_data_path("swapi/imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"))

        self.assertEqual(
            "Failed to parse Inflow Vector from imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat",
            str(exc_ctx.exception))


if __name__ == '__main__':
    unittest.main()
