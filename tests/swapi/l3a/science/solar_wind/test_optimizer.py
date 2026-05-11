"""Tests for `solar_wind.optimizer.optimize_solar_wind_params` and `OptimizeSolarWindParamsResult`."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.optimize

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    OptimizeSolarWindParamsResult,
    optimize_solar_wind_params,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    N_STATE,
    SolarWindParams,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


# ----- module-level fixture constants --------------------------------------

# Calibration CSVs shipped with the repo. Loading the full SwapiResponse
# triggers the same code path the production pipeline uses.
AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)

# A realistic proton sweep — voltages chosen to span the proton peak at
# ~450 km/s (V ≈ m_p v²/(2 e k) ≈ 560 V at k=1.89). Sixteen bins gives the
# LM problem enough information to recover all 5 parameters without making
# the synthetic data prohibitively expensive to forward-model.
# Geometric spacing matches the SWAPI L2 sweep (ratio ~1.115 per step).
_TYPICAL_PROTON_ESA_VOLTAGES = 300.0 * (1.115 ** np.arange(16))

# Slow-wind proton ground truth (RTN): 450 km/s along -Y_RTN, 5 cm^-3,
# 1e5 K. With identity instrument-to-RTN rotation, this puts the wind
# centered on SWAPI's boresight (+Y_inst), squarely inside the SG passband
# where the LM optimizer is well-behaved given a nearby initial guess.
_TRUE_DENSITY = 5.0
_TRUE_VELOCITY_RTN = np.array([0.0, -450.0, 0.0])
_TRUE_TEMPERATURE_K = 1.0e5


def _load_swapi_response() -> SwapiResponse:
    """Load `SwapiResponse` from the shipped instrument-team CSVs."""
    response = SwapiResponse.from_files(
        AZIMUTHAL_TRANSMISSION_PATH,
        CENTRAL_EFFECTIVE_AREA_PATH,
        PASSBAND_FIT_COEFFICIENTS_PATH,
    )
    response.warm_cache(_TYPICAL_PROTON_ESA_VOLTAGES)
    return response


def _proton_params(
    *,
    density: float = _TRUE_DENSITY,
    velocity_rtn=_TRUE_VELOCITY_RTN,
    temperature: float = _TRUE_TEMPERATURE_K,
) -> SolarWindParams:
    """Build a slow-wind proton `SolarWindParams` with optional overrides."""
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.array(velocity_rtn, dtype=float),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


def _build_proton_fit_context(count_rate: np.ndarray):
    """Build a real `SolarWindFitContext` for the fixture proton voltages.

    Uses identity SWAPI→RTN rotation matrices so the wind direction
    interpretation is straightforward (instrument frame == RTN)."""
    response = _load_swapi_response()
    n_sweeps = len(_TYPICAL_PROTON_ESA_VOLTAGES)
    rotation_matrices = np.broadcast_to(np.eye(3), (n_sweeps, 3, 3)).copy()
    return build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=_TYPICAL_PROTON_ESA_VOLTAGES,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )


def _zero_count_proton_context():
    """Proton fit context whose `count_rate` is all zeros — the forward model
    only depends on `count_rate` through the residual, so this is the cheapest
    seed when only the predicted side matters."""
    return _build_proton_fit_context(np.zeros_like(_TYPICAL_PROTON_ESA_VOLTAGES))


def _synthetic_count_rate_for(sw_params: SolarWindParams) -> np.ndarray:
    """Forward-model deadtime-applied count rates from `sw_params` against
    the fixture voltages. Used to seed both the "recovers known params" and
    cache-counting tests with a realistic measurement vector.

    Two-step: build a placeholder context with zero counts, run the forward
    model to get ideal rates, then apply the deadtime factor exactly the way
    `_Evaluator._eval` does (so the residual at the truth is zero)."""
    ctx = _zero_count_proton_context()
    ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(sw_params, ctx)
    return ideal_rates * deadtime_factor(ideal_rates)


class TestOptimizeSolarWindParamsResultMSE(unittest.TestCase):
    """Tests for `OptimizeSolarWindParamsResult.mse`."""

    def test_mse_is_mean_of_squared_residuals(self):
        """Given a residuals vector [1, -2, 3], mse returns 14/3 — the per-bin mean of squared residuals."""
        result = OptimizeSolarWindParamsResult(
            sw_params=_proton_params(),
            residuals=np.array([1.0, -2.0, 3.0]),
            jacobian=np.zeros((3, N_STATE)),
            success=True,
        )
        self.assertAlmostEqual(result.mse, 14.0 / 3.0)

    def test_mse_handles_all_zero_residuals_without_dividing_by_zero(self):
        """At the truth (noise-free fit) all residuals are zero and mse reports exactly 0.0 rather than NaN/Inf, so the wrong-basin chi^2 comparator can be applied unconditionally."""
        result = OptimizeSolarWindParamsResult(
            sw_params=_proton_params(),
            residuals=np.zeros(5),
            jacobian=np.zeros((5, N_STATE)),
            success=True,
        )
        self.assertEqual(result.mse, 0.0)


class TestOptimizeSolarWindParamsRecoversTruth(unittest.TestCase):
    """End-to-end tests for `optimize_solar_wind_params` driving an LM fit to a known synthetic solar wind state."""

    @classmethod
    def setUpClass(cls):
        cls.true_params = _proton_params()
        cls.count_rate = _synthetic_count_rate_for(cls.true_params)
        cls.ctx = _build_proton_fit_context(count_rate=cls.count_rate)
        # Initial guess perturbed in density (+10%), speed (+3%), and
        # temperature (+20%). Sized so LM converges in one basin without
        # invoking the wrong-basin flip; the basin-of-attraction bounds live
        # in docs/swapi/solar-wind-moments.md § Wrong-basin detection.
        cls.initial_guess = _proton_params(
            density=cls.true_params.density * 1.1,
            velocity_rtn=cls.true_params.bulk_velocity_rtn * 1.03,
            temperature=cls.true_params.temperature * 1.2,
        )
        cls.result = optimize_solar_wind_params(cls.initial_guess, cls.ctx)

    def test_optimizer_recovers_density(self):
        """Starting from a +10% density perturbation against noise-free synthetic data, the fitted density returns to the truth within 0.1%."""
        np.testing.assert_allclose(
            self.result.sw_params.density,
            self.true_params.density,
            rtol=1e-3,
        )

    def test_optimizer_recovers_temperature(self):
        """Starting from a +20% temperature perturbation against noise-free synthetic data, the fitted temperature returns to the truth within 0.1%."""
        np.testing.assert_allclose(
            self.result.sw_params.temperature,
            self.true_params.temperature,
            rtol=1e-3,
        )

    def test_optimizer_recovers_bulk_velocity(self):
        """Starting from a +3% speed perturbation against noise-free synthetic data, the fitted bulk velocity returns to the truth within 0.5 km/s on each RTN component."""
        np.testing.assert_allclose(
            self.result.sw_params.bulk_velocity_rtn,
            self.true_params.bulk_velocity_rtn,
            atol=0.5,
        )

    def test_optimizer_reports_success(self):
        """LM converges cleanly on the well-posed synthetic problem and the result's success flag is True."""
        self.assertTrue(self.result.success)

    def test_residuals_at_solution_are_small(self):
        """At the converged solution against noise-free data, mse is below (peak_rate)^2 * 1e-4 — residuals are limited only by LM's xtol=1e-4 tolerance."""
        peak_rate_squared = float(np.max(self.count_rate)) ** 2
        self.assertLess(self.result.mse, peak_rate_squared * 1e-4)


class TestOptimizeSolarWindParamsResultShape(unittest.TestCase):
    """Tests for `optimize_solar_wind_params` result-object shape and type contracts the wrong-basin detector and uncertainty derivation depend on."""

    @classmethod
    def setUpClass(cls):
        true_params = _proton_params()
        count_rate = _synthetic_count_rate_for(true_params)
        cls.ctx = _build_proton_fit_context(count_rate=count_rate)
        cls.result = optimize_solar_wind_params(true_params, cls.ctx)

    def test_sw_params_is_a_solar_wind_params(self):
        """The returned `sw_params` field is a `SolarWindParams` instance, not a raw state vector."""
        self.assertIsInstance(self.result.sw_params, SolarWindParams)

    def test_residuals_length_matches_ctx_count_rate(self):
        """The residuals array has one entry per observed count-rate bin in the fit context."""
        self.assertEqual(self.result.residuals.shape, self.ctx.count_rate.shape)

    def test_jacobian_shape_is_n_residuals_by_n_state(self):
        """The jacobian has shape (n_residuals, N_STATE) — one row per residual, one column per fit parameter."""
        self.assertEqual(
            self.result.jacobian.shape, (self.ctx.count_rate.size, N_STATE)
        )

    def test_success_is_bool(self):
        """The success field is a plain Python bool rather than numpy's bool-like wrapper."""
        self.assertIsInstance(self.result.success, bool)

    def test_sw_params_carries_context_mass(self):
        """The fitted `SolarWindParams.mass` is propagated from `ctx.mass_kg` rather than defaulting to proton mass."""
        self.assertEqual(self.result.sw_params.mass, self.ctx.mass_kg)


class TestOptimizerLeastSquaresKwargs(unittest.TestCase):
    """Tests that `optimize_solar_wind_params` invokes scipy with the doc-pinned LM kwargs."""

    def test_uses_lm_with_xtol_from_doc_spec(self):
        """When the optimizer runs, scipy.optimize.least_squares is called with `method='lm'` and `xtol=1e-4` per the doc spec."""
        true_params = _proton_params()
        count_rate = _synthetic_count_rate_for(true_params)
        ctx = _build_proton_fit_context(count_rate=count_rate)

        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.optimizer.scipy.optimize.least_squares"
        ) as mock_least_squares:
            mock_least_squares.return_value = scipy.optimize.OptimizeResult(
                x=true_params.to_vector(),
                fun=np.zeros_like(count_rate),
                jac=np.zeros((count_rate.size, N_STATE)),
                success=True,
            )

            optimize_solar_wind_params(true_params, ctx)

            kwargs = mock_least_squares.call_args.kwargs
            self.assertEqual(kwargs["method"], "lm")
            self.assertEqual(kwargs["xtol"], 1e-4)




if __name__ == "__main__":
    unittest.main()
