import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, PriorityEvent

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
    counts = priority_rate_variable * acquisition_time_in_seconds
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
