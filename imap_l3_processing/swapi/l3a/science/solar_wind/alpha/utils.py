import numpy as np
from numpy.typing import ArrayLike

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR


def esa_voltage_to_alpha_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * ALPHA_PARTICLE_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / ALPHA_PARTICLE_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def get_alpha_peak_indices(residuals, energies, proton_peak_index) -> slice:
    energies = np.asarray(energies)
    assert np.all(energies[:-1] >= energies[1:]), "Energies must be decreasing"

    min_energy = 1.5 * energies[proton_peak_index]
    max_energy = 4.0 * energies[proton_peak_index]

    def find_start_of_alpha_particle_peak():
        start_bin = None
        for i in reversed(range(proton_peak_index)):
            if energies[i] >= min_energy:
                start_bin = i
                break
        if start_bin is None:
            return None
        for i in reversed(range(start_bin + 1)):
            if residuals[i] > residuals[i + 1] and residuals[i - 1] > residuals[i]:
                return i
        return None

    start_of_alpha_peak = find_start_of_alpha_particle_peak()
    if start_of_alpha_peak is None:
        raise Exception("Alpha peak not found")

    end_of_alpha_peak = np.searchsorted(-energies, -max_energy)
    return slice(int(end_of_alpha_peak), start_of_alpha_peak + 1)
