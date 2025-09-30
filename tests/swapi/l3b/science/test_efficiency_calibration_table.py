import unittest
from datetime import datetime

from scipy.constants import milli
from spacepy import pycdf

from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from tests.test_helpers import get_test_data_path


class TestEfficiencyCalibrationTable(unittest.TestCase):
    def test_loads_calibration_table_and_returns_proton_efficiency_for_a_time(self):
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        self.assertEqual(
            efficiency_table.get_proton_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1))), 0.1)
        self.assertEqual(
            efficiency_table.get_proton_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2013, month=10, day=1))),
            0.1)
        self.assertEqual(
            efficiency_table.get_proton_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2014, month=10, day=3))),
            0.09)
        self.assertEqual(
            efficiency_table.get_proton_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=1))),
            0.09)
        self.assertEqual(
            efficiency_table.get_proton_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=3))),
            0.0882)

    def test_loads_calibration_table_and_returns_alpha_efficiency_for_a_time(self):
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        self.assertEqual(
            efficiency_table.get_alpha_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1))), 0.9)
        self.assertEqual(
            efficiency_table.get_alpha_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2013, month=10, day=1))),
            0.9)
        self.assertEqual(
            efficiency_table.get_alpha_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2014, month=10, day=3))),
            0.95)
        self.assertEqual(
            efficiency_table.get_alpha_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=1))),
            0.95)
        self.assertEqual(
            efficiency_table.get_alpha_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=3))),
            0.99)

    def test_get_efficiency_function_handles_float_input(self):
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        self.assertEqual(
            efficiency_table.get_alpha_efficiency_for(
                float(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1)))), 0.9)

        self.assertEqual(
            efficiency_table.get_proton_efficiency_for(
                float(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1)))), 0.1)

    def test_loads_calibration_table_raises_exception_if_ask_for_time_before_the_table_starts(self):
        calibration_table_path = get_test_data_path("swapi/imap_swapi_efficiency-lut-test_20241020_v001.dat")
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        with self.assertRaises(ValueError) as content_manager:
            time = datetime(year=1999, month=1, day=4)
            efficiency_table.get_proton_efficiency_for(pycdf.lib.datetime_to_tt2000(time))

        self.assertEqual((f"No efficiency data for {time}",), content_manager.exception.args)
