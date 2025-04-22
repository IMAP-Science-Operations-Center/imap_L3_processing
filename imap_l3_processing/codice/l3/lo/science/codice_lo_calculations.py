import numpy as np


def calculate_partial_densities(species_intensities: np.ndarray):
    return NotImplementedError


def calculate_total_number_of_events(priority_rate_variable: np.ndarray, acquisition_time: np.ndarray) -> float:
    acquisition_time_in_seconds = acquisition_time / 1_000_000

    return np.sum(priority_rate_variable * acquisition_time_in_seconds, dtype=float)
