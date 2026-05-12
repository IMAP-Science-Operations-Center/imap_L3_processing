from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams


@dataclass
class OptimizeSolarWindParamsResult:
    sw_params: SolarWindParams
    residuals: ndarray  # count-rate residuals at the solution
    jacobian: ndarray  # ∂residuals/∂state, columns ordered per the state vector
    success: bool

    @property
    def mse(self) -> float:
        return float(np.mean(self.residuals**2))


def optimize_solar_wind_params(
    initial_guess: SolarWindParams, ctx: SolarWindFitContext
) -> OptimizeSolarWindParamsResult:
    evaluator = _Evaluator(ctx)

    raw: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
        evaluator.residues,
        initial_guess.to_vector(),
        jac=evaluator.jacobian,
        method="lm",
        xtol=1e-4,
    )

    return OptimizeSolarWindParamsResult(
        sw_params=SolarWindParams.from_vector(raw.x, ctx.mass_kg),
        residuals=raw.fun,
        jacobian=raw.jac,
        success=bool(raw.success),
    )


class _Evaluator:
    """Caches the most recent (residuals, jacobian) so scipy.least_squares' separate
    `fun` and `jac` callbacks share a single forward-model evaluation per state."""

    def __init__(self, ctx: SolarWindFitContext):
        self.ctx = ctx
        self._last_state: ndarray | None = None
        self._last_resid: ndarray | None = None
        self._last_jac: ndarray | None = None

    def _eval(self, state: ndarray) -> None:
        sw = SolarWindParams.from_vector(state, self.ctx.mass_kg)
        rate_ideal, jacobian_ideal = model_solar_wind_ideal_coincidence_rates(
            sw, self.ctx
        )

        df = deadtime_factor(rate_ideal)

        rate_observable = rate_ideal * df
        jacobian = jacobian_ideal * np.square(df)[:, np.newaxis]

        residues = rate_observable - self.ctx.count_rate.ravel()

        self._last_state = state.copy()
        self._last_residues = residues
        self._last_jacobian = jacobian

    def _refresh(self, state: ndarray) -> None:
        if self._last_state is None or not np.array_equal(state, self._last_state):
            self._eval(state)

    def residues(self, state: ndarray) -> ndarray:
        self._refresh(state)
        return self._last_residues

    def jacobian(self, state: ndarray) -> ndarray:
        self._refresh(state)
        return self._last_jacobian
