from typing import NamedTuple, Self

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.response.swapi_response import SwapiResponse


class SolarWindFitContext(NamedTuple):
    count_rate: ndarray
    esa_voltage: ndarray
    response_grids: numba.typed.List
    rotation_matrices: ndarray
    mass_kg: float

    def subset(self, indices: ndarray) -> Self:
        return self._replace(
            count_rate=self.count_rate.ravel()[indices],
            esa_voltage=self.esa_voltage.ravel()[indices],
            response_grids=numba.typed.List([self.response_grids[i] for i in indices]),
            rotation_matrices=self.rotation_matrices[indices],
        )


def build_solar_wind_fit_context(
    count_rate: ndarray,
    esa_voltage: ndarray,
    swapi_response: SwapiResponse,
    central_effective_area_scale: float,
    rotation_matrices: ndarray,
    mass_kg: float,
    mass_per_charge_m_p_per_e: float,
) -> SolarWindFitContext:
    flat_voltage = esa_voltage.ravel()
    keep = (flat_voltage > 0) & np.isfinite(flat_voltage)
    if not np.all(keep):
        esa_voltage = flat_voltage[keep]
        count_rate = count_rate.ravel()[keep]
        rotation_matrices = rotation_matrices[keep]
        flat_voltage = esa_voltage

    response_grids = numba.typed.List(
        [
            swapi_response.get_response_grid(
                v, mass_per_charge_m_p_per_e, central_effective_area_scale
            )
            for v in flat_voltage
        ]
    )

    return SolarWindFitContext(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        response_grids=response_grids,
        rotation_matrices=rotation_matrices,
        mass_kg=float(mass_kg),
    )
