import math
import unittest

import numpy as np

from imap_processing.constants import ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR
from imap_processing.swe.l3.science.moment_calculations import compute_velocity_vectors, convert_energy_to_speed


class TestMomentsCalculation(unittest.TestCase):
    def test_convert_energy_to_velocity(self):
        self.assertAlmostEqual(ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(2.5),
                               convert_energy_to_speed(2.5))

    def test_compute_velocity_vectors(self):
        energy_bins = 2
        spin_angle_bins = 2
        declination_angle_bins = 3

        corrected_energy = np.array([
            [[1, 2, 3],
             [4, 5, 6]
             ],
            [[7, 8, 9],
             [10, 11, 12]
             ]])

        corrected_energy = np.random.rand(energy_bins, spin_angle_bins, declination_angle_bins)
        spin_angles = np.arange(0, 360, 360 / spin_angle_bins)
        declination_angles = np.arange(0, 180, 180 / declination_angle_bins)

        velocities = compute_velocity_vectors(corrected_energy, spin_angles, declination_angles)

        self.assertEqual((energy_bins * spin_angle_bins * declination_angle_bins, 3), velocities.shape)

        vx1 = -1 * convert_energy_to_speed(corrected_energy[0][0][0]) * (
            np.sin(spin_angles[0] * math.pi / 180)) * np.cos(
            declination_angles[0] * math.pi / 180)
        vy1 = -1 * convert_energy_to_speed(corrected_energy[0][0][0]) * (
            np.sin(spin_angles[0] * math.pi / 180)) * np.sin(
            declination_angles[0] * math.pi / 180)
        vz1 = -1 * convert_energy_to_speed(corrected_energy[0][0][0]) * np.cos(spin_angles[0] * math.pi / 180)

        vx2 = -1 * convert_energy_to_speed(corrected_energy[0][0][1]) * (
            np.sin(spin_angles[0] * math.pi / 180)) * np.cos(
            declination_angles[1] * math.pi / 180)
        vy2 = -1 * convert_energy_to_speed(corrected_energy[0][0][1]) * (
            np.sin(spin_angles[0] * math.pi / 180)) * np.sin(
            declination_angles[1] * math.pi / 180)
        vz2 = -1 * convert_energy_to_speed(corrected_energy[0][0][1]) * np.cos(spin_angles[0] * math.pi / 180)

        np.testing.assert_array_almost_equal([vx1, vy1, vz1], velocities[0])
        np.testing.assert_array_almost_equal([vx2, vy2, vz2], velocities[1])
