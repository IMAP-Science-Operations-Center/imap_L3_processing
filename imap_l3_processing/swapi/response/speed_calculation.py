import numpy as np
import uncertainties
from numpy.typing import ArrayLike
from uncertainties import umath, unumpy

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    METERS_PER_KILOMETER,
)

# Revised SWAPI ESA k-factor from high-resolution SIMION simulations (Q ≈ 1.89 eV/V at θ = 0).
# Used internally by L3 for passband normalization and central-speed conversions.
SWAPI_K_FACTOR = 1.89

# Pre-launch k-factor used by the L2 product to label its `esa_energy` field as
# `esa_energy = SWAPI_L2_K_FACTOR × |voltage|`. Different from SWAPI_K_FACTOR — divide L2's
# `esa_energy` by this to recover true ESA voltage before any L3 processing.
SWAPI_L2_K_FACTOR = 1.93

# SWAPI ESA sweep bin layout (72 bins total, indices 0–71):
#   Index 0       : always discarded (hardware artifact, never science data)
#   Indices 1–62  : coarse sweep passbands (62 bins, uniform energy steps)
#   Indices 63–71 : fine sweep passbands (9 bins, higher resolution near the proton peak)
SWAPI_DISCARDED_BIN = 0
SWAPI_COARSE_SWEEP_BINS = slice(1, 63)  # indices 1–62
SWAPI_FINE_SWEEP_BINS = slice(63, 72)  # indices 63–71
SWAPI_SCIENCE_BINS = slice(1, 72)  # indices 1–71, all usable bins (coarse + fine)


def esa_voltage_to_proton_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * PROTON_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def calculate_sw_speed(particle_mass, particle_charge, energy):
    """Energy-per-charge → speed for an ion of given mass/charge. Handles scalars,
    arrays, and uncertainties.UFloat values."""
    if np.size(energy) == 0:
        return np.array([])
    dimensions = np.asanyarray(energy).ndim
    if dimensions > 0:
        if isinstance(np.ravel(energy)[0], uncertainties.UFloat):
            return (
                unumpy.sqrt(2 * energy * particle_charge / particle_mass)
                / METERS_PER_KILOMETER
            )
        return (
            np.sqrt(2 * energy * particle_charge / particle_mass) / METERS_PER_KILOMETER
        )
    else:
        return (
            umath.sqrt(2 * energy * particle_charge / particle_mass)
            / METERS_PER_KILOMETER
        )


def calculate_sw_speed_h_plus(energy):
    return calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, energy)
