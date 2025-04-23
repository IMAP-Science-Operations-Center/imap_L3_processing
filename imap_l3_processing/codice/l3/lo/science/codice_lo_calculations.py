import numpy as np

from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle


def calculate_partial_densities(species_intensities: np.ndarray):
    return NotImplementedError


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
