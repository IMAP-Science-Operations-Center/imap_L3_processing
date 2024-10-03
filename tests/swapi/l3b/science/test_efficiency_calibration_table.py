import unittest
from datetime import datetime
from pathlib import Path

from spacepy import pycdf

import imap_l3_data_processor
from imap_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable


class TestEfficiencyCalibrationTable(unittest.TestCase):
    def test_loads_calibration_table_and_returns_efficiency_for_a_time(self):
        calibration_table_path = Path(
            imap_l3_data_processor.__file__).parent / "swapi" / "test_data" / "imap_swapi_l2_efficiency-lut-text-not-cdf_20241020_v001.cdf"
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        self.assertEqual(
            efficiency_table.get_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2001, month=2, day=1))), 0.1)
        self.assertEqual(
            efficiency_table.get_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2013, month=10, day=1))),
            0.1)
        self.assertEqual(
            efficiency_table.get_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2014, month=10, day=3))),
            0.09)
        self.assertEqual(
            efficiency_table.get_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=1))),
            0.09)
        self.assertEqual(
            efficiency_table.get_efficiency_for(pycdf.lib.datetime_to_tt2000(datetime(year=2024, month=10, day=3))),
            0.0882)

    def test_loads_calibration_table_raises_exception_if_ask_for_time_before_the_table_starts(self):
        calibration_table_path = Path(
            imap_l3_data_processor.__file__).parent / "swapi" / "test_data" / "imap_swapi_l2_efficiency-lut-text-not-cdf_20241020_v001.cdf"
        efficiency_table = EfficiencyCalibrationTable(calibration_table_path)

        with self.assertRaises(ValueError) as content_manager:
            time = datetime(year=1999, month=1, day=4)
            efficiency_table.get_efficiency_for(pycdf.lib.datetime_to_tt2000(time))

        self.assertEqual((f"No efficiency data for {time}",), content_manager.exception.args)
