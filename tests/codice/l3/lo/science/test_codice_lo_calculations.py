import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, call

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, \
    PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, PriorityEvent, CodiceLo3dData
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_total_number_of_events, calculate_normalization_ratio, calculate_mass, calculate_mass_per_charge, \
    rebin_to_counts_by_species_elevation_and_spin_sector, rebin_counts_by_energy_and_spin_angle, \
    CODICE_LO_NUM_AZIMUTH_BINS, normalize_counts, combine_priorities_and_convert_to_rate, \
    rebin_3d_distribution_azimuth_to_elevation, convert_count_rate_to_intensity
from imap_l3_processing.constants import ONE_SECOND_IN_MICROSECONDS
from tests.test_helpers import create_dataclass_mock


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
            position=np.array([]),
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
            position=np.array([]),
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

    def test_calculate_mass_per_charge_handles_fill_value(self):
        priority_event = Mock()
        priority_event.tof = np.ma.masked_array([[432, 234], [434, 347]], mask=[[False, True], [True, False]])
        priority_event.energy_step = np.array([[np.nan, 2342.2], [4324.8, np.nan]])

        actual_mass_per_charge = calculate_mass_per_charge(priority_event)

        self.assertIsInstance(actual_mass_per_charge, np.ma.masked_array)
        actual_filled_with_nan = np.ma.filled(actual_mass_per_charge, np.nan)

        self.assertTrue(np.all(np.isnan(actual_filled_with_nan)))

    def test_calculate_mass_handles_fill_value(self):
        rng = np.random.default_rng()
        priority_event = Mock()
        mass_coefficients = MassCoefficientLookup(coefficients=rng.random(size=6))

        priority_event.tof = np.ma.masked_array([[432, 234], [434, 347]], mask=[[False, True], [True, False]])
        priority_event.apd_energy = np.array([[np.nan, 2342.2], [4324.8, np.nan]])

        actual_mass = calculate_mass(priority_event, mass_coefficients)

        self.assertIsInstance(actual_mass, np.ma.masked_array)
        actual_filled_with_nan = np.ma.filled(actual_mass, np.nan)

        self.assertTrue(np.all(np.isnan(actual_filled_with_nan)))

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
        num_esa_steps = 128
        num_species = 4

        # first 3 are first priority, last is second priority
        expected_species_returned = [
            "He+", "Fe", "He+", "O+5",
            "Mg", "Mg", None
        ]
        mock_species_mass_range_lookup = Mock(spec=MassSpeciesBinLookup)
        mock_species_mass_range_lookup.get_species.side_effect = expected_species_returned
        mock_species_mass_range_lookup.get_num_species.return_value = num_species

        def mock_get_species_index(species):
            return_values = {"He+": 0, "Fe": 1, "Mg": 2, "O+5": 3, }
            return return_values[species]

        mock_species_mass_range_lookup.get_species_index = mock_get_species_index

        num_events = np.array([[3, 1], [3, 1]])

        mock_energy_lookup = Mock(spec=EnergyLookup)
        mock_energy_lookup.get_energy_index.side_effect = [0, 100, 0, 0, 127, 127, 0]
        mock_energy_lookup.num_bins = num_esa_steps
        mock_energy_lookup.bin_centers = Mock()

        spin_angle = np.array([
            [[37.5, 22.5, 37.5, -9999], [7.5, np.nan, np.nan, np.nan]],
            [[37.5, 37.5, 37.5, np.nan], [7.5, np.nan, np.nan, np.nan]]
        ])

        position = np.ma.masked_array(data=np.array([
            [[2, 4, 2, 255], [1, 255, 255, 255]],
            [[9, 9, 9, 255], [1, 255, 255, 255]]
        ]), mask=np.array([
            [[False, False, False, True], [False, True, True, True]],
            [[False, True, False, True], [False, True, True, True]],
        ]))

        energy_step = np.array([
            [[0.0, 1234.2, 0.0, np.nan], [0.0, np.nan, np.nan, np.nan]],
            [[345.2, 345.2, 345.2, np.nan], [0.0, np.nan, np.nan, np.nan]]
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

        actual_counts_3d_data = rebin_to_counts_by_species_elevation_and_spin_sector(num_events, mass, mass_per_charge,
                                                                                     energy_step,
                                                                                     spin_angle,
                                                                                     position,
                                                                                     mock_species_mass_range_lookup,
                                                                                     spin_angle_lut,
                                                                                     mock_energy_lookup)

        mock_species_mass_range_lookup.get_species.assert_has_calls([
            call(mass[0, 0, 0], mass_per_charge[0, 0, 0]),
            call(mass[0, 0, 1], mass_per_charge[0, 0, 1]),
            call(mass[0, 0, 2], mass_per_charge[0, 0, 2]),

            call(mass[0, 1, 0], mass_per_charge[0, 1, 0]),

            call(mass[1, 0, 0], mass_per_charge[1, 0, 0]),
            call(mass[1, 0, 2], mass_per_charge[1, 0, 2]),

            call(mass[1, 1, 0], mass_per_charge[1, 1, 0]),
        ])

        mock_energy_lookup.get_energy_index.assert_has_calls([
            call(0.0),
            call(1234.2),
            call(0.0),
            call(0.0),
            call(345.2),
            call(345.2),
        ])

        self.assertIsInstance(actual_counts_3d_data, CodiceLo3dData)

        self.assertEqual(
            (num_species, num_epochs, num_priorities, CODICE_LO_NUM_AZIMUTH_BINS, num_spin_angles, num_esa_steps),
            actual_counts_3d_data.data_in_3d_bins.shape)

        self.assertEqual(mock_energy_lookup.bin_centers, actual_counts_3d_data.energy_per_charge)
        np.testing.assert_array_equal(np.arange(1, CODICE_LO_NUM_AZIMUTH_BINS + 1),
                                      actual_counts_3d_data.azimuth_or_elevation)
        np.testing.assert_array_equal(spin_angle_lut.bin_centers, actual_counts_3d_data.spin_angle)

        rebinned_shape = (num_epochs, num_priorities, CODICE_LO_NUM_AZIMUTH_BINS, num_spin_angles, num_esa_steps)
        expected_he_plus_counts = np.zeros(rebinned_shape)
        expected_he_plus_counts[0, 0, 1, 2, 0] = 2
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("He+"), expected_he_plus_counts)

        expected_fe_counts = np.zeros(rebinned_shape)
        expected_fe_counts[0, 0, 3, 1, 100] = 1
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("Fe"), expected_fe_counts)

        expected_mg_counts = np.zeros(rebinned_shape)
        expected_mg_counts[1, 0, 8, 2, 127] = 2
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("Mg"), expected_mg_counts)

        expected_o_counts = np.zeros(rebinned_shape)
        expected_o_counts[0, 1, 0, 0, 0] = 1
        np.testing.assert_array_equal(actual_counts_3d_data.get_3d_distribution("O+5"), expected_o_counts)

    def test_rebin_counts_to_by_energy_and_spin_angle(self):
        mock_energy_lookup = Mock(spec=EnergyLookup)
        num_energy_bins = 30
        mock_energy_lookup.num_bins = num_energy_bins
        mock_energy_lookup.get_energy_index.side_effect = [
            np.array([7]),
            np.array([1, 1]),
            np.array([1, 3, 1, 4])
        ]

        mock_spin_angle_lookup = Mock(spec=SpinAngleLookup)
        num_spin_angle_bins = 20
        mock_spin_angle_lookup.num_bins = num_spin_angle_bins
        mock_spin_angle_lookup.get_spin_angle_index.side_effect = [
            np.array([3]),
            np.array([2, 2]),
            np.array([5, 6, 5, 7])
        ]

        priority_event = create_dataclass_mock(PriorityEvent)
        rng = np.random.default_rng()

        priority_event.num_events = np.array([1, 2, 4])
        num_epochs = len(priority_event.num_events)
        priority_event.spin_angle = rng.random((num_epochs, 15))
        priority_event.energy_step = rng.random((num_epochs, 15))
        result = rebin_counts_by_energy_and_spin_angle(priority_event, mock_spin_angle_lookup, mock_energy_lookup)

        expected_rebinned_counts = np.zeros((num_epochs, num_energy_bins, num_spin_angle_bins))
        expected_rebinned_counts[0, 7, 3] = 1
        expected_rebinned_counts[1, 1, 2] = 2
        expected_rebinned_counts[2, 1, 5] = 2
        expected_rebinned_counts[2, 3, 6] = 1
        expected_rebinned_counts[2, 4, 7] = 1

        np.testing.assert_array_equal(result, expected_rebinned_counts)

    def test_normalize_counts(self):
        num_epochs = 2
        num_priorities = 3
        num_azimuth_bins = 5
        num_spin_angles = 6
        num_energies = 7

        rng = np.random.default_rng()

        counts = np.zeros((num_epochs, num_priorities, num_azimuth_bins, num_spin_angles, num_energies))
        counts[:, :, 0, :, :] = 3
        counts[:, :, 1, :, :] = 4

        normalization_factor = rng.random((num_epochs, num_priorities, num_energies, num_spin_angles))

        actual_normalized_counts = normalize_counts(counts, normalization_factor)

        transposed_normalization_factor = np.transpose(normalization_factor, axes=(0, 1, 3, 2))
        np.testing.assert_array_equal(actual_normalized_counts[:, :, 0, :, :], 3 * transposed_normalization_factor)
        np.testing.assert_array_equal(actual_normalized_counts[:, :, 1, :, :], 4 * transposed_normalization_factor)

    def test_convert_to_count_rates_combine_priority(self):
        num_epochs = 2
        num_azimuth_bins = 5
        num_spin_angles = 6
        num_energies = 7

        rng = np.random.default_rng()
        priority_1 = rng.random((num_epochs, num_azimuth_bins, num_spin_angles, num_energies))
        priority_2 = rng.random((num_epochs, num_azimuth_bins, num_spin_angles, num_energies))
        priority_3 = rng.random((num_epochs, num_azimuth_bins, num_spin_angles, num_energies))

        counts = np.stack((priority_1, priority_2, priority_3), axis=2)

        acquisition_durations_in_seconds = rng.random((num_energies,))
        acquisition_duration_in_microseconds = acquisition_durations_in_seconds * ONE_SECOND_IN_MICROSECONDS

        actual_count_rates = combine_priorities_and_convert_to_rate(counts, acquisition_duration_in_microseconds)
        expected_summed_counts = priority_1 + priority_2 + priority_3

        self.assertEqual((num_epochs, num_azimuth_bins, num_spin_angles, num_energies),
                         actual_count_rates.shape)

        for index, _ in np.ndenumerate(np.ones((num_epochs, num_azimuth_bins, num_spin_angles))):
            np.testing.assert_array_almost_equal(actual_count_rates[*index, :],
                                                 expected_summed_counts[*index, :] / acquisition_durations_in_seconds)

    def test_convert_count_rate_to_intensity(self):
        num_epochs = 3
        num_azimuth_bins = 4
        num_spin_angles = 5
        num_energies = 6

        rng = np.random.default_rng()
        count_rates = rng.random((num_epochs, num_azimuth_bins, num_spin_angles, num_energies))
        energy_per_charge = EnergyLookup(bin_centers=rng.random(num_energies),
                                         bin_edges=rng.random(num_energies),
                                         delta_plus=rng.random(num_energies),
                                         delta_minus=rng.random(num_energies))
        geometric_factor = rng.random((num_epochs, num_energies))

        efficiency = rng.random((num_azimuth_bins, num_energies))

        mock_efficiency_lookup = Mock()
        mock_efficiency_lookup.efficiency_data = efficiency

        intensity_data = convert_count_rate_to_intensity(count_rates, energy_per_charge, mock_efficiency_lookup,
                                                         geometric_factor)

        expected_denominator = (energy_per_charge.bin_centers
                                * geometric_factor[:, np.newaxis, np.newaxis, :]
                                * efficiency[np.newaxis, :, np.newaxis, :])

        np.testing.assert_array_almost_equal(intensity_data * expected_denominator, count_rates)

    def test_rebin_azimuth_to_elevation(self):
        num_epochs = 3
        num_azimuth_bins = 4
        num_spin_angles = 5
        num_energies = 6

        intensity_data = np.zeros((num_epochs, num_azimuth_bins, num_spin_angles, num_energies))
        azimuth_1_intensity = 1
        azimuth_2_intensity = 2
        azimuth_3_intensity = 3
        azimuth_4_intensity = 4

        intensity_data[:, 0, :, :] = azimuth_1_intensity
        intensity_data[:, 1, :, :] = azimuth_2_intensity
        intensity_data[:, 2, :, :] = azimuth_3_intensity
        intensity_data[:, 3, :, :] = azimuth_4_intensity

        azimuths = np.array([1, 2, 3, 4])

        num_elevation_bins = 3
        mock_position_to_elevation_lookup = Mock(spec=PositionToElevationLookup)
        mock_position_to_elevation_lookup.num_bins = num_elevation_bins
        mock_position_to_elevation_lookup.bin_centers = np.arange(num_elevation_bins)
        mock_position_to_elevation_lookup.apd_to_elevation_index.return_value = np.array([1, 0, 2, 2])

        actual_rebinned = rebin_3d_distribution_azimuth_to_elevation(intensity_data,
                                                                     azimuths,
                                                                     mock_position_to_elevation_lookup)

        mock_position_to_elevation_lookup.apd_to_elevation_index.assert_called_once_with(azimuths)

        self.assertEqual((num_epochs, num_elevation_bins, num_spin_angles, num_energies),
                         actual_rebinned.shape)

        self.assertTrue(np.all(actual_rebinned[:, 0, :, :] == azimuth_2_intensity))
        self.assertTrue(np.all(actual_rebinned[:, 1, :, :] == azimuth_1_intensity))
        self.assertTrue(np.all(actual_rebinned[:, 2, :, :] == (azimuth_3_intensity + azimuth_4_intensity)))
