"""Shared fixtures and helpers for SWAPI tests.

Consolidates the `proton_params`, `load_swapi_response`, and
`synthesize_count_rates` helpers that were previously duplicated across
the SWAPI test suite.
"""

from typing import Optional

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


# Calibration CSVs shipped with the repo. Loading the full SwapiResponse from
# these triggers the same code path the production pipeline uses.
SWAPI_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
SWAPI_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
SWAPI_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


def load_swapi_response(
    warm_cache_voltages: Optional[np.ndarray] = None,
) -> SwapiResponse:
    """Build a `SwapiResponse` from the shipped instrument-team CSVs.

    Pass `warm_cache_voltages` to populate the passband cache for callers that
    will later call `create_response_grid` / `get_response_grid` (those raise
    on a cache miss)."""
    response = SwapiResponse.from_files(
        SWAPI_AZIMUTHAL_TRANSMISSION_PATH,
        SWAPI_CENTRAL_EFFECTIVE_AREA_PATH,
        SWAPI_PASSBAND_FIT_COEFFICIENTS_PATH,
    )
    if warm_cache_voltages is not None:
        response.warm_cache(warm_cache_voltages)
    return response


def proton_params(
    *,
    density: float = 5.0,
    velocity_rtn=(0.0, -450.0, 0.0),
    temperature: float = 100_000.0,
) -> SolarWindParams:
    """Solar-wind proton `SolarWindParams` at typical slow-wind values.

    Default `velocity_rtn = (0, -450, 0)` puts the bulk wind along -Y_RTN; under
    the identity SWAPI→RTN rotation that maps to -Y_inst, i.e. straight down
    the boresight (the SG passband center). Tests that need a different
    geometry pass `velocity_rtn` explicitly."""
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.asarray(velocity_rtn, dtype=float),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


def synthesize_count_rates(ctx, sw_params: SolarWindParams) -> np.ndarray:
    """Forward-model deadtime-applied coincidence count rates from `sw_params`
    using the same model the fitter inverts. No Poisson noise — the
    parameter-recovery tests exercise orchestration, not the noise budget."""
    ideal, _ = model_solar_wind_ideal_coincidence_rates(sw_params, ctx)
    return ideal * deadtime_factor(ideal)
