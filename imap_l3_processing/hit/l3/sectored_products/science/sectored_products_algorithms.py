import numpy as np


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
