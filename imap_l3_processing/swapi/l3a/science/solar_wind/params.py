import math
from typing import NamedTuple

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    METERS_PER_KILOMETER,
)

# layout of "x" vector for the least-squares optimizer
LOG_DENSITY_IDX = 0
LOG_TEMPERATURE_IDX = 1
VELOCITY_SLICE = slice(2, 5)
N_STATE = 5


class SolarWindParams(NamedTuple):
    density: float
    bulk_velocity_rtn: ndarray  # shape (3,), km/s, inertial RTN
    temperature: float  # K
    mass: float  # kg

    def to_vector(self) -> ndarray:
        state = np.empty(N_STATE)
        state[LOG_DENSITY_IDX] = math.log(self.density)
        state[LOG_TEMPERATURE_IDX] = math.log(self.temperature)
        state[VELOCITY_SLICE] = self.bulk_velocity_rtn
        return state

    @classmethod
    def from_vector(cls, state: ndarray, mass: float) -> "SolarWindParams":
        return cls(
            density=math.exp(state[LOG_DENSITY_IDX]),
            bulk_velocity_rtn=state[VELOCITY_SLICE],
            temperature=math.exp(state[LOG_TEMPERATURE_IDX]),
            mass=mass,
        )


@numba.njit
def bulk_speed(sw_params: SolarWindParams) -> float:
    """||bulk_velocity_rtn||, km/s."""
    v = sw_params.bulk_velocity_rtn
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@numba.njit
def thermal_speed(sw_params: SolarWindParams) -> float:
    """Maxwellian standard deviation σ = √(kT/m), km/s."""
    temperature_k = sw_params.temperature
    mass_kg = sw_params.mass
    return temperature_to_thermal_speed(mass_kg, temperature_k)


@numba.njit
def temperature_to_thermal_speed(mass: float, temperature: float) -> float:
    """mass [kg], temperature [K] -> thermal speed [km/s]."""
    return (
        math.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * temperature / mass)
        / METERS_PER_KILOMETER
    )


@numba.njit
def thermal_speed_to_temperature(thermal_speed: float, mass: float) -> float:
    """thermal speed [km/s], mass [kg] -> temperature [K]."""
    return (
        mass
        * (thermal_speed * METERS_PER_KILOMETER) ** 2
        / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    )
