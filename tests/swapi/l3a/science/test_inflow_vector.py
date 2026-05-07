"""Tests for `InflowVector`."""

import tempfile
import unittest
from pathlib import Path

from imap_l3_processing.swapi.l3a.science.inflow_vector import InflowVector
from tests.test_helpers import get_test_data_path


class TestInflowVectorFromFile(unittest.TestCase):
    def test_from_hydrogen_file(self):
        v = InflowVector.from_file(
            get_test_data_path(
                "swapi/imap_swapi_hydrogen-inflow-vector_20100101_v001.dat"
            )
        )
        self.assertEqual(v.speed_km_per_s, 22.0)
        self.assertEqual(v.longitude_deg_eclipj2000, 252.2)
        self.assertEqual(v.latitude_deg_eclipj2000, 9.0)

    def test_from_helium_file(self):
        v = InflowVector.from_file(
            get_test_data_path(
                "swapi/imap_swapi_helium-inflow-vector_20100101_v001.dat"
            )
        )
        self.assertEqual(v.speed_km_per_s, 25.4)
        self.assertEqual(v.longitude_deg_eclipj2000, 255.7)
        self.assertEqual(v.latitude_deg_eclipj2000, 5.1)

    def test_raises_on_wrong_number_of_columns(self):
        # Files with anything other than 3 columns per row must fail at parse time.
        for content in ("30.5  100.0\n", "30.5  100.0  -10.0  99.9\n"):
            with self.subTest(content=content):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".dat", delete=False
                ) as tmp:
                    tmp.write("# synthetic file\n")
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                try:
                    with self.assertRaises(AssertionError) as ctx:
                        InflowVector.from_file(tmp_path)
                    self.assertIn("Failed to parse Inflow Vector", str(ctx.exception))
                finally:
                    tmp_path.unlink()

    def test_handles_extra_whitespace_and_comment_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as tmp:
            tmp.write("# header line ignored by np.loadtxt\n")
            tmp.write("   30.5   100.0   -10.0   \n")
            tmp_path = Path(tmp.name)
        try:
            v = InflowVector.from_file(tmp_path)
            self.assertEqual(v.speed_km_per_s, 30.5)
            self.assertEqual(v.longitude_deg_eclipj2000, 100.0)
            self.assertEqual(v.latitude_deg_eclipj2000, -10.0)
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
