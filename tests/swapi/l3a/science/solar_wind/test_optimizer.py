"""Tests for `solar_wind.optimizer` — `optimize_solar_wind_params` and the
internal `_Evaluator` cache.

The optimizer wraps `scipy.optimize.least_squares(method='lm')` on the
deadtime-corrected forward model. `_Evaluator` caches the most recent
(residuals, jacobian) so scipy's separate `fun` and `jac` callbacks share
one forward-model evaluation per state. Doc spec lives in
`docs/swapi/solar-wind-moments.md` § Least-Squares Fitting Procedure.
"""

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
    _Evaluator,
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
    """`OptimizeSolarWindParamsResult.mse` returns the mean of squared
    residuals (per-bin), used as the chi^2 surrogate by the wrong-basin
    detector."""

    def test_mse_is_mean_of_squared_residuals(self):
        result = OptimizeSolarWindParamsResult(
            sw_params=_proton_params(),
            residuals=np.array([1.0, -2.0, 3.0]),
            jacobian=np.zeros((3, N_STATE)),
            success=True,
        )
        # mean([1, 4, 9]) = 14/3
        self.assertAlmostEqual(result.mse, 14.0 / 3.0)

    def test_mse_handles_all_zero_residuals_without_dividing_by_zero(self):
        # Edge case: at the truth (or in synthetic noise-free fits) residuals
        # are exactly zero. mse must report 0.0 rather than NaN/Inf so the
        # wrong-basin chi^2 comparator can be applied unconditionally.
        result = OptimizeSolarWindParamsResult(
            sw_params=_proton_params(),
            residuals=np.zeros(5),
            jacobian=np.zeros((5, N_STATE)),
            success=True,
        )
        self.assertEqual(result.mse, 0.0)


class TestOptimizeSolarWindParamsRecoversTruth(unittest.TestCase):
    """End-to-end: feed the optimizer count rates synthesized from a known
    `SolarWindParams`, start from a perturbed initial guess, and verify the
    LM fit converges to the truth within tolerance.

    This is the integration test for the doc § Least-Squares Fitting
    Procedure — exercises forward model + deadtime correction + LM through
    `optimize_solar_wind_params` end to end."""

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
        # Self-consistency: data was generated with the same forward model
        # the LM optimizes against, so residuals drive to zero up to LM's
        # `xtol=1e-4` (see docs/swapi/solar-wind-moments.md § Least-Squares
        # Fitting Procedure).
        np.testing.assert_allclose(
            self.result.sw_params.density,
            self.true_params.density,
            rtol=1e-3,
        )

    def test_optimizer_recovers_temperature(self):
        np.testing.assert_allclose(
            self.result.sw_params.temperature,
            self.true_params.temperature,
            rtol=1e-3,
        )

    def test_optimizer_recovers_bulk_velocity(self):
        np.testing.assert_allclose(
            self.result.sw_params.bulk_velocity_rtn,
            self.true_params.bulk_velocity_rtn,
            atol=0.5,  # km/s
        )

    def test_optimizer_reports_success(self):
        self.assertTrue(self.result.success)

    def test_residuals_at_solution_are_small(self):
        # MSE relative to the (large) count rates should be tiny — we built
        # the data noise-free using the same forward model the LM is fitting,
        # so residuals are limited only by LM convergence (`xtol=1e-4`, see
        # docs/swapi/solar-wind-moments.md § Least-Squares Fitting Procedure).
        peak_rate_squared = float(np.max(self.count_rate)) ** 2
        self.assertLess(self.result.mse, peak_rate_squared * 1e-4)


class TestOptimizeSolarWindParamsResultShape(unittest.TestCase):
    """Verify the returned `OptimizeSolarWindParamsResult` is fully
    populated and has the right shapes/types — these are the contracts the
    wrong-basin detector and uncertainty derivation depend on."""

    @classmethod
    def setUpClass(cls):
        true_params = _proton_params()
        count_rate = _synthetic_count_rate_for(true_params)
        cls.ctx = _build_proton_fit_context(count_rate=count_rate)
        cls.result = optimize_solar_wind_params(true_params, cls.ctx)

    def test_sw_params_is_a_solar_wind_params(self):
        self.assertIsInstance(self.result.sw_params, SolarWindParams)

    def test_residuals_length_matches_ctx_count_rate(self):
        self.assertEqual(self.result.residuals.shape, self.ctx.count_rate.shape)

    def test_jacobian_shape_is_n_residuals_by_n_state(self):
        self.assertEqual(
            self.result.jacobian.shape, (self.ctx.count_rate.size, N_STATE)
        )

    def test_success_is_bool(self):
        self.assertIsInstance(self.result.success, bool)

    def test_sw_params_carries_context_mass(self):
        self.assertEqual(self.result.sw_params.mass, self.ctx.mass_kg)


class TestOptimizerLeastSquaresKwargs(unittest.TestCase):
    """The doc pins `method='lm'` and `xtol=1e-4` — the LM-specific
    convergence tolerance the SWAPI fitter relies on. Verify the actual
    scipy call gets these."""

    def test_uses_lm_with_xtol_from_doc_spec(self):
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


class TestEvaluatorDeadtimeApplication(unittest.TestCase):
    """The optimizer fits the *deadtime-corrected* predicted rate to the
    measured rate — `r_obs = r_ideal · f_dt(r_ideal)`. Verify this is
    actually applied to the residuals and jacobian inside `_Evaluator._eval`.
    See docs/swapi/solar-wind-moments.md § Deadtime correction (§ 5% at
    ~2.7e5 Hz at peak slow-wind rates)."""

    @classmethod
    def setUpClass(cls):
        cls.true_params = _proton_params()
        cls.ctx = _zero_count_proton_context()
        cls.ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(
            cls.true_params, cls.ctx
        )
        cls.evaluator = _Evaluator(cls.ctx)

    def test_residual_applies_deadtime_to_predicted_rate(self):
        # With ctx.count_rate == 0, residual == predicted == r_ideal · f_dt
        # — so the residual exposes the deadtime-applied predicted rate
        # directly.
        residuals = self.evaluator.residues(self.true_params.to_vector())
        df = deadtime_factor(self.ideal_rates)
        np.testing.assert_allclose(residuals, self.ideal_rates * df)

    def test_residual_subtracts_observed_count_rate(self):
        # Non-zero count_rate path: residual == r_ideal · f_dt − count_rate.
        observed = self.ideal_rates * 0.5  # any measurable rate vector
        ctx = _build_proton_fit_context(observed)
        evaluator = _Evaluator(ctx)
        residuals = evaluator.residues(self.true_params.to_vector())
        df = deadtime_factor(self.ideal_rates)
        np.testing.assert_allclose(residuals, self.ideal_rates * df - observed)

    def test_jacobian_includes_deadtime_chain_rule_factor(self):
        # Doc § Deadtime correction derives ∂C_obs/∂C_model = f_dt² via the
        # quotient rule, so the residual jacobian = ideal_jacobian · f_dt².
        _, jacobian_ideal = model_solar_wind_ideal_coincidence_rates(
            self.true_params, self.ctx
        )
        jacobian = self.evaluator.jacobian(self.true_params.to_vector())
        df_squared = np.square(deadtime_factor(self.ideal_rates))[:, np.newaxis]
        # Some columns are ~0 (e.g. the v_T column when the wind has
        # v_T==0); use atol to absorb the float64 ULP noise on those entries.
        np.testing.assert_allclose(
            jacobian, jacobian_ideal * df_squared, rtol=1e-6, atol=1e-9
        )


class TestEvaluatorCachesPerState(unittest.TestCase):
    """`_Evaluator` caches the most recent forward-model evaluation so a
    `residues` call followed by a `jacobian` call at the same state runs
    the forward model exactly once. State-change invalidates the cache."""

    @classmethod
    def setUpClass(cls):
        true_params = _proton_params()
        cls.count_rate = _synthetic_count_rate_for(true_params)
        cls.ctx = _build_proton_fit_context(count_rate=cls.count_rate)
        cls.true_params = true_params

    def test_residues_then_jacobian_at_same_state_evaluates_forward_model_once(self):
        evaluator = _Evaluator(self.ctx)
        state = self.true_params.to_vector()
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.optimizer.model_solar_wind_ideal_coincidence_rates",
            wraps=model_solar_wind_ideal_coincidence_rates,
        ) as spy:
            evaluator.residues(state)
            evaluator.jacobian(state)
            self.assertEqual(spy.call_count, 1)

    def test_jacobian_then_residues_at_same_state_evaluates_forward_model_once(self):
        # Order-independence: scipy may call jac before fun on a given state.
        evaluator = _Evaluator(self.ctx)
        state = self.true_params.to_vector()
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.optimizer.model_solar_wind_ideal_coincidence_rates",
            wraps=model_solar_wind_ideal_coincidence_rates,
        ) as spy:
            evaluator.jacobian(state)
            evaluator.residues(state)
            self.assertEqual(spy.call_count, 1)

    def test_state_change_re_evaluates_forward_model(self):
        # When LM steps, the next call hits a new state and the cache must
        # invalidate — otherwise the residual/jacobian become stale.
        evaluator = _Evaluator(self.ctx)
        state_a = self.true_params.to_vector()
        state_b = state_a + 1.0
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.optimizer.model_solar_wind_ideal_coincidence_rates",
            wraps=model_solar_wind_ideal_coincidence_rates,
        ) as spy:
            residuals_a = evaluator.residues(state_a).copy()
            residuals_b = evaluator.residues(state_b)
            self.assertEqual(spy.call_count, 2)
            # The new state must produce a different residual vector — if it
            # didn't, the test would also pass when the cache served stale
            # data alongside a redundant model re-evaluation.
            self.assertFalse(np.array_equal(residuals_a, residuals_b))

    def test_first_call_evaluates_forward_model_once(self):
        # The "no cache yet" branch — verifies the first call evaluates the
        # forward model rather than returning an uninitialized cache slot.
        evaluator = _Evaluator(self.ctx)
        state = self.true_params.to_vector()
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.optimizer.model_solar_wind_ideal_coincidence_rates",
            wraps=model_solar_wind_ideal_coincidence_rates,
        ) as spy:
            evaluator.residues(state)
            self.assertEqual(spy.call_count, 1)


if __name__ == "__main__":
    unittest.main()
