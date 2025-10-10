import dataclasses
from datetime import datetime, timedelta

import numpy as np

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.hit.l3.models import HitL2Data


def get_hit_bin_polar_coordinates(declination_bins=8, inclination_bins=15) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    declination_starts, declination_step = np.linspace(0, 180, declination_bins, endpoint=False, retstep=True)
    declination_delta = declination_step / 2
    declinations = declination_starts + declination_delta
    inclination_starts, inclination_step = np.linspace(0, 360, inclination_bins, endpoint=False, retstep=True)
    inclination_delta = inclination_step / 2
    inclinations = inclination_starts + inclination_delta
    declination_deltas = np.array([declination_delta] * declination_bins)
    inclination_deltas = np.array([inclination_delta] * inclination_bins)

    return declinations, inclinations, declination_deltas, inclination_deltas


def get_sector_unit_vectors(declinations_degrees: np.ndarray, inclinations_degrees: np.ndarray) -> np.ndarray:
    declinations = np.deg2rad(declinations_degrees)
    inclinations = np.deg2rad(inclinations_degrees)
    declinations = declinations[:, np.newaxis]
    z = np.cos(declinations)
    sin_dec = np.sin(declinations)
    x = sin_dec * np.cos(inclinations)
    y = sin_dec * np.sin(inclinations)

    stacked = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    return stacked

def transform_to_10_minute_chunks(hit_data: HitL2Data) -> HitL2Data:
    species_energy = [
        ("h", 3),
        ("he4", 2),
        ("cno", 2),
        ("nemgsi", 2),
        ("fe", 1),
    ]

    transformed_hit_dict = {}

    input_hit_dict = dataclasses.asdict(hit_data)

    species_i = 0
    for species, num_energy_levels in species_energy:
        input_species_data = input_hit_dict[species]
        new_species_shape = (input_species_data.shape[0] // 10, *input_species_data.shape[1:])

        transformed_hit_dict[species] = np.full(new_species_shape, np.nan)
        transformed_hit_dict[f"delta_plus_{species}"] = np.full(new_species_shape, np.nan)
        transformed_hit_dict[f"delta_minus_{species}"] = np.full(new_species_shape, np.nan)

        for energy_i in range(num_energy_levels):
            transformed_hit_dict[species][:, energy_i] = input_species_data[species_i::10, energy_i]
            transformed_hit_dict[f"delta_plus_{species}"][:, energy_i] = input_hit_dict[f"delta_plus_{species}"][species_i::10, energy_i]
            transformed_hit_dict[f"delta_minus_{species}"][:, energy_i] = input_hit_dict[f"delta_minus_{species}"][species_i::10, energy_i]
            species_i += 1

    minute_cadence_epochs = input_hit_dict['epoch']
    ten_minute_cadence_epochs = minute_cadence_epochs.reshape(-1, 10)
    new_epochs = []
    for chunk in ten_minute_cadence_epochs:
        start_time = chunk[0]
        end_time = chunk[-1]
        new_epochs.append(start_time + (end_time - start_time) / 2 - timedelta(minutes=10))
    transformed_hit_dict['epoch'] = np.array(new_epochs)
    transformed_hit_dict['epoch_delta'] = np.full(np.array(new_epochs).shape, timedelta(minutes=5))

    return dataclasses.replace(hit_data, **transformed_hit_dict)

