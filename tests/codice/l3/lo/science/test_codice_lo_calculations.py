import dataclasses
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np

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
             [7, 8, 9]],
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]])
        acquisition_time = np.array([
            [[2, 2, 2],
             [3, 3, 3],
             [1, 1, 1]],
            [[2, 2, 2],
             [3, 3, 3],
             [1, 1, 1]]]) * 1_000_000

        expected_total_number_of_events = [12 + 45 + 24, 12 + 45 + 24]

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
        num_spin_angles = 13
        num_azimuth = 24
        num_esa_steps = 128
        num_events_per_epoch = 4

        expected_species_returned = ["He+", "Fe", "He+", "Mg", "Mg", "Mg"]
        mock_species_mass_range_lookup = Mock(spec=MassSpeciesBinLookup)
        mock_species_mass_range_lookup.get_species.side_effect = expected_species_returned

        def mock_get_species_index(species, direction):
            return_values = {
                ("He+", EventDirection.Sunward): 0,
                ("Fe", EventDirection.Sunward): 1,
                ("Mg", EventDirection.Sunward): 2,
            }
            return return_values[(species, direction)]

        mock_species_mass_range_lookup.get_species_index = mock_get_species_index

        apd_id = np.array([[0, 15, 30, np.nan], [15, 15, 15, np.nan]])
        spin_angle = np.array([[30, 15, 0, np.nan], [30, 30, 30, np.nan]])
        energy_step = np.array([[0, 100, 127, np.nan], [127, 127, 127, np.nan]])
        num_events = np.array([[1, 2, 3, np.nan], [2, 2, 2, np.nan]])
        mass = np.array([[6, 5, 4, np.nan], [3, 3, 3, np.nan]])
        mass_per_charge = np.array([[1, 2, 3, np.nan], [4, 4, 4, np.nan]])

        priority_event = self.create_empty_priority_event(num_epochs, num_events_per_epoch)
        priority_event = dataclasses.replace(priority_event,
                                             apd_id=apd_id,
                                             energy_step=energy_step,
                                             num_events=num_events,
                                             spin_angle=spin_angle
                                             )

        actual_counts_3d_data = rebin_to_counts_by_azimuth_spin_sector(mass, mass_per_charge, priority_event,
                                                                       mock_species_mass_range_lookup)

        self.assertIsInstance(actual_counts_3d_data, CodiceLo3dData)

        rebinned_shape = (num_epochs, num_spin_angles, num_azimuth, num_esa_steps)
        expected_he_plus_counts = np.zeros(rebinned_shape)
        expected_he_plus_counts[0][0][2][0] = 1
        expected_he_plus_counts[0][2][0][127] = 3
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("He+", EventDirection.Sunward),
                                      expected_he_plus_counts)

        expected_fe_counts = np.zeros(rebinned_shape)
        expected_fe_counts[0][1][1][100] = 2
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("Fe", EventDirection.Sunward),
                                      expected_fe_counts)

        expected_mg_counts = np.zeros(rebinned_shape)
        expected_mg_counts[1][1][2][127] = 6
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("Mg", EventDirection.Sunward),
                                      expected_mg_counts)

    def create_empty_priority_event(self, num_epochs: int, num_events_per_epoch: int) -> PriorityEvent:
        return PriorityEvent(
            **{field.name: np.zeros((num_epochs, num_events_per_epoch)) for field in dataclasses.fields(PriorityEvent)})
