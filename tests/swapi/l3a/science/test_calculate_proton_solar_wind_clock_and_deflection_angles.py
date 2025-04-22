from pathlib import Path
from unittest import TestCase

from uncertainties import ufloat

import imap_l3_processing
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    calculate_clock_angle, ClockAngleCalibrationTable, calculate_deflection_angle


class TestCalculateProtonSolarWindClockAndDeflectionAngles(TestCase):
    def setUp(self):
        file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / 'imap_swapi_clock-angle-and-flow-deflection-lut_20240918_v000.dat'
        self.lookup_table = ClockAngleCalibrationTable.from_file(file_path)

    def test_calculate_clock_angle(self):
        proton_solar_wind_speed, a, phi, b = (
            ufloat(405.00, .1), ufloat(15.0, .1), ufloat(33.3, .1),
            ufloat(1_000.0, .1))
        clock_angle = calculate_clock_angle(self.lookup_table, proton_solar_wind_speed, a, phi, b)
        self.assertAlmostEqual(phi.n + 6.09, clock_angle.n, 4)
        self.assertAlmostEqual(0.1, clock_angle.s, 4)

    def test_calculate_clock_angle_with_interpolation(self):
        proton_solar_wind_speed, a, phi, b = (ufloat(845.00, .1),
                                              ufloat(31.25, .2),
                                              ufloat(33.3, .2),
                                              ufloat(1_000.0, .2))
        clock_angle = calculate_clock_angle(self.lookup_table, proton_solar_wind_speed, a, phi, b)
        self.assertAlmostEqual(phi.n + 6.39, clock_angle.n, 4)
        self.assertAlmostEqual(phi.s, clock_angle.s, 3)

    def test_calculate_clock_angle_mods(self):
        proton_solar_wind_speed, a, phi, b = (ufloat(845.00, .1),
                                              ufloat(31.25, .2),
                                              ufloat(358.3, .2),
                                              ufloat(1_000.0, .2))
        clock_angle = calculate_clock_angle(self.lookup_table, proton_solar_wind_speed, a, phi, b)
        self.assertAlmostEqual(4.69, clock_angle.n, 3)
        self.assertAlmostEqual(phi.s, clock_angle.s, 3)

    def test_calculate_deflection_angle_with_interpolation(self):
        proton_solar_wind_speed, a, phi, b = (ufloat(845.00, .1),
                                              ufloat(31.25, .2),
                                              ufloat(33.3, .2),
                                              ufloat(1_000.0, .2))
        deflection_angle = calculate_deflection_angle(self.lookup_table, proton_solar_wind_speed, a, phi, b)
        self.assertAlmostEqual(3.125, deflection_angle.n, 4)
        self.assertAlmostEqual(0.02, deflection_angle.s, 4)

    def test_clock_angle_calibration_table_from_file(self):
        file_path = Path(
            imap_l3_processing.__file__).parent.parent / 'tests' / 'test_data' / 'swapi' / 'imap_swapi_clock-angle-and-flow-deflection-lut_20240918_v000.dat'

        proton_solar_wind_speed = 405.00
        a_over_b = 0.01625

        calibration_table = ClockAngleCalibrationTable.from_file(file_path)

        self.assertEqual(6.105, calibration_table.lookup_clock_offset(proton_solar_wind_speed, a_over_b))
        self.assertEqual(1.625, calibration_table.lookup_flow_deflection(proton_solar_wind_speed, a_over_b))
