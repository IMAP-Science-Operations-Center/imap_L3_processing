from __future__ import annotations

import dataclasses
from typing import TypeVar

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, \
    PositionToElevationLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, PriorityEvent, CodiceLo3dData
from imap_l3_processing.constants import ONE_SECOND_IN_MICROSECONDS

POST_ACCELERATION_VOLTAGE_IN_KV = 15
ENERGY_LOST_IN_CARBON_FOIL = 0
CONVERSION_CONSTANT_K = 1.692e-5

AZIMUTH_STEP_SIZE = np.deg2rad(30)
ELEVATION_STEP_SIZE = np.deg2rad(30)
ENERGY_STEP_SIZE = 100

CODICE_LO_NUM_AZIMUTH_BINS = 24


def calculate_partial_densities(intensities: np.ndarray, esa_steps: np.ndarray, mass_per_charge: float):
    return np.sum((1 / np.sqrt(2)) * AZIMUTH_STEP_SIZE * ELEVATION_STEP_SIZE * ENERGY_STEP_SIZE * intensities * np.sqrt(
        esa_steps[np.newaxis, :, np.newaxis]) * np.sqrt(mass_per_charge), axis=(1, 2))


def calculate_total_number_of_events(priority_rate_variable: np.ndarray, acquisition_time: np.ndarray) -> np.ndarray[
    int]:
    acquisition_time_in_seconds = acquisition_time / 1_000_000
    counts = np.multiply(priority_rate_variable, acquisition_time_in_seconds[np.newaxis, :, np.newaxis])
    return np.sum(counts, axis=(1, 2), dtype=int)


def calculate_normalization_ratio(energy_and_spin_angle_counts: dict[EnergyAndSpinAngle, int],
                                  total_number_of_events: int):
    normalization_ratio = np.full((128, 12), np.nan)
    for (energy, spin_angle), counts in energy_and_spin_angle_counts.items():
        normalization_ratio[energy, spin_angle] = total_number_of_events / counts
    return normalization_ratio


def calculate_mass(priority_event: PriorityEvent, mass_coefficients: MassCoefficientLookup) -> np.ndarray:
    energy = np.log(priority_event.apd_energy)
    tof = np.log(priority_event.tof)

    mass_calculation = mass_coefficients[0] + (mass_coefficients[1] * energy) + (mass_coefficients[2] * tof) + (
            mass_coefficients[3] * energy * tof) + (mass_coefficients[4] * np.power(energy, 2)) + (
                               mass_coefficients[5] * np.power(tof, 3))
    return np.e ** mass_calculation


def calculate_mass_per_charge(priority_event: PriorityEvent) -> np.ndarray:
    return (priority_event.energy_step + POST_ACCELERATION_VOLTAGE_IN_KV - ENERGY_LOST_IN_CARBON_FOIL) * (
            priority_event.tof ** 2) * CONVERSION_CONSTANT_K


def rebin_counts_by_energy_and_spin_angle(priority_event: PriorityEvent,
                                          spin_angle_lookup: SpinAngleLookup,
                                          energy_lookup: EnergyLookup) -> np.ndarray:
    num_epochs = len(priority_event.num_events)
    num_energies = energy_lookup.num_bins
    num_spin_bins = spin_angle_lookup.num_bins

    rebinned_output = np.zeros((num_epochs, num_energies, num_spin_bins))

    for time_index, num_events in enumerate(priority_event.num_events):
        spin_angle_in_degrees = priority_event.spin_angle[time_index, :num_events]
        energy_in_keV = priority_event.energy_step[time_index, :num_events]

        spin_angle_indices = spin_angle_lookup.get_spin_angle_index(spin_angle_in_degrees)
        energy_indices = energy_lookup.get_energy_index(energy_in_keV)

        for energy_index, spin_angle_index in zip(energy_indices, spin_angle_indices):
            rebinned_output[time_index, energy_index, spin_angle_index] += 1

    return rebinned_output


def rebin_to_counts_by_species_elevation_and_spin_sector(mass: np.ndarray, mass_per_charge: np.ndarray,
                                                         energy: np.ndarray,
                                                         spin_angle: np.ndarray, apd_id: np.ndarray,
                                                         mass_species_bin_lookup: MassSpeciesBinLookup,
                                                         spin_angle_lut: SpinAngleLookup,
                                                         position_elevation_lut: PositionToElevationLookup,
                                                         energy_lut: EnergyLookup) -> CodiceLo3dData:
    num_epochs = mass.shape[0]
    num_priorities = mass.shape[1]

    output = np.full((mass_species_bin_lookup.get_num_species(), num_epochs, num_priorities,
                      CODICE_LO_NUM_AZIMUTH_BINS, spin_angle_lut.num_bins, energy_lut.num_bins), 0)

    def filter_nan(a: np.array) -> np.array:
        return a[~np.isnan(a)]

    for epoch_i in range(num_epochs):
        for priority_i in range(num_priorities):
            masked_spin_angle = filter_nan(spin_angle[epoch_i, priority_i])
            masked_apd_id = filter_nan(apd_id[epoch_i, priority_i])
            masked_mass = filter_nan(mass[epoch_i, priority_i])
            masked_mass_per_charge = filter_nan(mass_per_charge[epoch_i, priority_i])
            masked_energy = filter_nan(energy[epoch_i, priority_i])

            event_directions = [position_elevation_lut.event_direction_for_apd(apd) for apd in masked_apd_id]

            energy_indices = energy_lut.get_energy_index(masked_energy)

            species = [mass_species_bin_lookup.get_species(m, mpc, event_direction) for m, mpc, event_direction in
                       zip(masked_mass, masked_mass_per_charge, event_directions)]

            species_indices = [mass_species_bin_lookup.get_species_index(s, event_direction) for s, event_direction in
                               zip(species, event_directions)]

            apd_indices = masked_apd_id - 1
            spin_angle_indices = spin_angle_lut.get_spin_angle_index(masked_spin_angle)

            indexes = np.column_stack(
                (species_indices, apd_indices, spin_angle_indices, energy_indices)).astype(int)

            for species_i, position_i, spin_angle_i, energy_i in indexes:
                output[species_i, epoch_i, priority_i, position_i, spin_angle_i, energy_i] += 1

    return CodiceLo3dData(data_in_3d_bins=output, mass_bin_lookup=mass_species_bin_lookup,
                          energy_per_charge=energy_lut.bin_centers, spin_angle=spin_angle_lut.bin_centers,
                          azimuth_or_elevation=np.arange(1, CODICE_LO_NUM_AZIMUTH_BINS + 1))


EPOCH = TypeVar("EPOCH")
PRIORITY = TypeVar("PRIORITY")
SPECIES = TypeVar("SPECIES")
AZIMUTH = TypeVar("AZIMUTH")
SPIN_ANGLE = TypeVar("SPIN_ANGLE")
ENERGY = TypeVar("ENERGY")


def normalize_counts(counts: CodiceLo3dData,
                     normalization_factor: np.ndarray[(EPOCH, PRIORITY, ENERGY, SPIN_ANGLE)]) \
        -> CodiceLo3dData:
    reshaped_normalization_factor = np.transpose(normalization_factor, (0, 1, 3, 2))
    reshaped_normalization_factor = reshaped_normalization_factor[:, :, np.newaxis, np.newaxis, :, :]
    return dataclasses.replace(counts, data_in_3d_bins=reshaped_normalization_factor * counts.data_in_3d_bins)


def combine_priorities_and_convert_to_rate(counts: CodiceLo3dData,
                                           acquisition_times: np.ndarray[(ENERGY,)]) \
        -> CodiceLo3dData:
    return dataclasses.replace(counts, data_in_3d_bins=np.sum(counts.data_in_3d_bins, axis=1) / (
            acquisition_times / ONE_SECOND_IN_MICROSECONDS))
