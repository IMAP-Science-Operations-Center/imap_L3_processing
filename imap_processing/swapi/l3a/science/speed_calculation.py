import numpy as np
from uncertainties import wrap


def get_peak_indices(count_rates, width, mask=True) -> slice:
    max_indices = np.argwhere(
            (count_rates == np.max(count_rates, where=mask, initial=0)) & mask)
    left_min_index = np.min(max_indices)
    right_min_index = np.max(max_indices)

    if right_min_index - left_min_index > 1:
        raise Exception("Count rates contains multiple distinct peaks")

    return slice(max(0, left_min_index - width), right_min_index + width+1)

def find_peak_center_of_mass_index(peak_slice, count_rates):
    indices = np.arange(len(count_rates))
    peak_counts = count_rates[peak_slice]
    center_of_mass_index = np.sum(indices[peak_slice] * peak_counts) / np.sum(peak_counts)
    return center_of_mass_index

def interpolate_energy(center_of_mass_index, energies):
    interpolate_lambda = lambda x: np.exp(np.interp(x, np.arange(len(energies)), np.log(energies)))
    interpolate = wrap(interpolate_lambda)
    return interpolate(center_of_mass_index)


def times_for_sweep(start_time):
    time_step_per_bin = 12 / 72
    return start_time + time_step_per_bin * np.arange(72)


def extract_coarse_sweep(data: np.ndarray):
    if data.ndim > 1:
        return data[:, 1:63]
    else:
        return data[1:63]