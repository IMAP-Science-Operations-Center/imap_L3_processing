import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, PriorityEvent, CodiceLo3dData

POST_ACCELERATION_VOLTAGE_IN_KV = 15
ENERGY_LOST_IN_CARBON_FOIL = 0
CONVERSION_CONSTANT_K = 1.692e-5

AZIMUTH_STEP_SIZE = np.deg2rad(30)
ELEVATION_STEP_SIZE = np.deg2rad(30)
ENERGY_STEP_SIZE = 100


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


def rebin_to_counts_by_azimuth_spin_sector(mass: np.ndarray, mass_per_charge: np.ndarray, priority_event: PriorityEvent,
                                           mass_species_bin_lookup: MassSpeciesBinLookup) -> CodiceLo3dData:
    azimuth_lut = {0: 0, 15: 1, 30: 2}
    spin_sector_lut = {0: 0, 15: 1, 30: 2}

    num_epochs = len(mass)

    output = np.full((num_epochs, 25, 13, 24, 128), 0)

    def filter_nan(a: np.array) -> np.array:
        return a[~np.isnan(a)]

    for i in range(num_epochs):
        apd_id = filter_nan(priority_event.apd_id[i])
        masked_spin_angle = filter_nan(priority_event.spin_angle[i])
        masked_num_event = filter_nan(priority_event.num_events[i])
        masked_mass = filter_nan(mass[i])
        masked_mass_per_charge = filter_nan(mass_per_charge[i])
        masked_energy_step = filter_nan(priority_event.energy_step[i])

        species = [mass_species_bin_lookup.get_species(m, mpc, EventDirection.Sunward) for m, mpc in
                   zip(masked_mass, masked_mass_per_charge)]
        species_indices = [mass_species_bin_lookup.get_species_index(s, EventDirection.Sunward) for s in species]
        azimuth_indices = [azimuth_lut[id] for id in apd_id]
        spin_sector_indices = [spin_sector_lut[spin_angle] for spin_angle in masked_spin_angle]

        indexes = np.column_stack(
            (species_indices, azimuth_indices, spin_sector_indices, masked_energy_step)).astype(int)
        counts = masked_num_event

        for index, count in zip(indexes, counts):
            output[i, *index] += count

    return CodiceLo3dData(data_in_3d_bins=output, mass_bin_lookup=mass_species_bin_lookup)
