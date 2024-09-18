from pathlib import Path
from unittest import TestCase

import imap_processing
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    calculate_clock_angle, ClockAngleCalibrationTable


class TestCalculateProtonSolarWindClockAndDeflectionAngles(TestCase):
    def setUp(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_clock-angle-and-flow-deflection-lut-text-not-cdf_20240918_v001.cdf'
        self.lookup_table = ClockAngleCalibrationTable.from_file(file_path)

    def test_calculate_clock_angle(self):
        proton_solar_wind_speed, a, phi, b = (405.00, 15.0, 33.3, 1_000.0)
        clock_angle = calculate_clock_angle(self.lookup_table, proton_solar_wind_speed, a, phi, b)
        self.assertAlmostEqual(phi + 6.09, clock_angle, 4)

    def test_calculate_clock_angle_with_interpolation(self):
        proton_solar_wind_speed, a, phi, b = (850.00, 48.75, 33.3, 1_000.0)
        clock_angle = calculate_clock_angle(self.lookup_table, proton_solar_wind_speed, a, phi, b)
        self.assertAlmostEqual(phi + 6.95, clock_angle, 4)

    def test_clock_angle_calibration_table_from_file(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / 'imap_swapi_l2_clock-angle-and-flow-deflection-lut-text-not-cdf_20240918_v001.cdf'

        proton_solar_wind_speed = 405.00
        a_over_b = 0.01625

        calibration_table = ClockAngleCalibrationTable.from_file(file_path)

        self.assertEqual(6.105, calibration_table.lookup_clock_offset(proton_solar_wind_speed, a_over_b))
        self.assertEqual(1.625, calibration_table.lookup_flow_deflection(proton_solar_wind_speed, a_over_b))
