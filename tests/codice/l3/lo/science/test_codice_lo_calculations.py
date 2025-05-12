import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, call

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, ElevationLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, PriorityEvent, CodiceLo3dData
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_total_number_of_events, calculate_normalization_ratio, calculate_mass, calculate_mass_per_charge, \
    rebin_to_counts_by_azimuth_spin_sector


class TestCodiceLoCalculations(unittest.TestCase):

    def test_calculate_total_number_of_events(self):
        priority_0_tcrs = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]],
            [[2, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]]])
        acquisition_time = np.array([1, 2, 3, 4]) * 1_000_000

        expected_total_number_of_events = [6 + 30 + 72 + 132, 7 + 30 + 72 + 132]

        actual_total_number_of_events = calculate_total_number_of_events(priority_0_tcrs, acquisition_time)
        np.testing.assert_array_equal(actual_total_number_of_events, expected_total_number_of_events)

    def test_calculate_normalization_ratio(self):
        energy_and_spin_angle_counts = {
            EnergyAndSpinAngle(energy=1, spin_angle=6): 30,
            EnergyAndSpinAngle(energy=2, spin_angle=5): 25,
            EnergyAndSpinAngle(energy=3, spin_angle=4): 20,
        }

        total_number_of_events = 300
        normalization_ratios = calculate_normalization_ratio(energy_and_spin_angle_counts, total_number_of_events)

        expected_normalization_ratio = np.full((128, 12), np.nan)

        expected_normalization_ratio[1][6] = 10
        expected_normalization_ratio[2][5] = 12
        expected_normalization_ratio[3][4] = 15

        np.testing.assert_array_equal(expected_normalization_ratio, normalization_ratios)

    def test_calculate_mass(self):
        priority_event = PriorityEvent(
            apd_energy=np.array([
                [np.exp(1)], [np.exp(2)],
            ]),
            tof=np.array([
                [np.exp(50)], [np.exp(60)],
            ]),
            apd_gain=np.array([]),
            apd_id=np.array([]),
            data_quality=np.array([]),
            energy_step=np.array([]),
            multi_flag=np.array([]),
            num_events=np.array([]),
            spin_angle=np.array([]),
            elevation=np.array([]),
        )

        lookup = MassCoefficientLookup(np.array([10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6]))
        actual_mass = calculate_mass(priority_event, lookup)

        expected_mass_1 = lookup[0] + lookup[1] * 1 + lookup[2] * 50 + lookup[3] * 1 * 50 + lookup[4] * 1 + lookup[
            5] * np.power(50, 3)
        expected_mass_2 = lookup[0] + lookup[1] * 2 + lookup[2] * 60 + lookup[3] * 2 * 60 + lookup[4] * np.power(2, 2) + \
                          lookup[5] * np.power(60, 3)

        expected_mass_calculation = np.array([
            [np.e ** expected_mass_1],
            [np.e ** expected_mass_2]])

        np.testing.assert_array_equal(actual_mass, expected_mass_calculation)

    def test_calculate_mass_per_charge(self):
        priority_event = PriorityEvent(
            apd_energy=np.array([]),
            tof=np.array([
                [5], [6],
            ]),
            apd_gain=np.array([]),
            apd_id=np.array([]),
            data_quality=np.array([]),
            energy_step=np.array([
                [1], [2],
            ]),
            multi_flag=np.array([]),
            num_events=np.array([]),
            spin_angle=np.array([]),
            elevation=np.array([]),
        )

        POST_ACCELERATION_VOLTAGE_IN_KV = 15
        ENERGY_LOST_IN_CARBON_FOIL = 0
        CONVERSION_CONSTANT_K = 1.692e-5

        mass_per_charge_1 = (1 + POST_ACCELERATION_VOLTAGE_IN_KV - ENERGY_LOST_IN_CARBON_FOIL) * (
                5 ** 2) * CONVERSION_CONSTANT_K

        mass_per_charge_2 = (2 + POST_ACCELERATION_VOLTAGE_IN_KV - ENERGY_LOST_IN_CARBON_FOIL) * (
                6 ** 2) * CONVERSION_CONSTANT_K

        actual_mass_per_charge = calculate_mass_per_charge(priority_event)
        np.testing.assert_array_equal(actual_mass_per_charge, np.array([[mass_per_charge_1], [mass_per_charge_2]]))

    def test_calculate_partial_densities(self):
        rng = np.random.default_rng()
        epochs = np.array([datetime.now(), datetime.now() + timedelta(days=1)])
        energy_steps = np.geomspace(100000, 1, num=128)
        intensities = rng.random((len(epochs), len(energy_steps), 1))
        mass_per_charge = 10

        partial_densities = calculate_partial_densities(intensities, energy_steps, mass_per_charge)

        expected_partial_densities = np.sum(
            (1 / np.sqrt(2)) * np.deg2rad(30) * np.deg2rad(30) * 100 * intensities * np.sqrt(
                energy_steps[np.newaxis, :, np.newaxis]) * np.sqrt(mass_per_charge), axis=(1, 2))

        np.testing.assert_array_equal(expected_partial_densities, partial_densities)

    def test_rebin_by_species(self):
        num_epochs = 2
        num_priorities = 2
        num_spin_angles = 24
        num_elevation = 13
        num_esa_steps = 128
        num_species = 4

        # first 3 are first priority, last is second priority
        expected_species_returned = [
            "He+", "Fe", "He+", "O+5",
            "Mg", "Mg", "Mg", "O+5"
        ]
        mock_species_mass_range_lookup = Mock(spec=MassSpeciesBinLookup)
        mock_species_mass_range_lookup.get_species.side_effect = expected_species_returned
        mock_species_mass_range_lookup.get_num_species.return_value = num_species

        def mock_get_species_index(species, direction):
            return_values = {
                ("He+", EventDirection.Sunward): 0,
                ("Fe", EventDirection.NonSunward): 1,
                ("Mg", EventDirection.NonSunward): 2,
                ("O+5", EventDirection.Sunward): 3,

            }
            return return_values[(species, direction)]

        mock_species_mass_range_lookup.get_species_index = mock_get_species_index

        spin_angle = np.array([
            [[37.5, 22.5, 37.5, np.nan], [7.5, np.nan, np.nan, np.nan]],
            [[37.5, 37.5, 37.5, np.nan], [7.5, np.nan, np.nan, np.nan]]
        ])

        elevation = np.array([
            [[15, 45, 15, np.nan], [0, np.nan, np.nan, np.nan]],
            [[120, 120, 120, np.nan], [0, np.nan, np.nan, np.nan]]
        ])

        energy_step = np.array([
            [[0, 100, 0, np.nan], [0, np.nan, np.nan, np.nan]],
            [[127, 127, 127, np.nan], [0, np.nan, np.nan, np.nan]]
        ])

        mass = np.array([
            [[6, 5, 4, np.nan], [7, np.nan, np.nan, np.nan]],
            [[3, 3, 3, np.nan], [7, np.nan, np.nan, np.nan]],
        ])

        mass_per_charge = np.array([
            [[1, 2, 3, np.nan], [8, np.nan, np.nan, np.nan]],
            [[4, 4, 4, np.nan], [9, np.nan, np.nan, np.nan]],
        ])

        spin_angle_lut = SpinAngleLookup()
        elevation_lut = ElevationLookup()

        actual_counts_3d_data = rebin_to_counts_by_azimuth_spin_sector(mass, mass_per_charge, energy_step, spin_angle,
                                                                       elevation, mock_species_mass_range_lookup,
                                                                       spin_angle_lut, elevation_lut)

        mock_species_mass_range_lookup.get_species.assert_has_calls([
            call(mass[0, 0, 0], mass_per_charge[0, 0, 0], EventDirection.Sunward),
            call(mass[0, 0, 1], mass_per_charge[0, 0, 1], EventDirection.NonSunward),
            call(mass[0, 0, 2], mass_per_charge[0, 0, 2], EventDirection.Sunward),

            call(mass[0, 1, 0], mass_per_charge[0, 1, 0], EventDirection.Sunward),

            call(mass[1, 0, 0], mass_per_charge[1, 0, 0], EventDirection.NonSunward),
            call(mass[1, 0, 1], mass_per_charge[1, 0, 1], EventDirection.NonSunward),
            call(mass[1, 0, 2], mass_per_charge[1, 0, 2], EventDirection.NonSunward),

            call(mass[1, 1, 0], mass_per_charge[1, 1, 0], EventDirection.Sunward),

        ])

        self.assertIsInstance(actual_counts_3d_data, CodiceLo3dData)

        self.assertEqual((num_epochs, num_priorities, num_species, num_elevation, num_spin_angles, num_esa_steps),
                         actual_counts_3d_data.data_in_3d_bins.shape)

        rebinned_shape = (num_epochs, num_priorities, num_elevation, num_spin_angles, num_esa_steps)
        expected_he_plus_counts = np.zeros(rebinned_shape)
        expected_he_plus_counts[0, 0, 1, 2, 0] = 2
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("He+", EventDirection.Sunward),
                                      expected_he_plus_counts)

        expected_fe_counts = np.zeros(rebinned_shape)
        expected_fe_counts[0, 0, 3, 1, 100] = 1
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("Fe", EventDirection.NonSunward),
                                      expected_fe_counts)

        expected_mg_counts = np.zeros(rebinned_shape)
        expected_mg_counts[1, 0, 8, 2, 127] = 3
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("Mg", EventDirection.NonSunward),
                                      expected_mg_counts)

        expected_o_counts = np.zeros(rebinned_shape)
        expected_o_counts[0, 1, 0, 0, 0] = 1
        expected_o_counts[1, 1, 0, 0, 0] = 1
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("O+5", EventDirection.Sunward),
                                      expected_o_counts)
