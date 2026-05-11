"""Direct tests for `solar_wind.forward_model` — the JIT-compiled solar-wind
forward model and its analytic Jacobian.

The contract under test is taken from `docs/swapi/solar-wind-moments.md`:
- `model_solar_wind_ideal_coincidence_rates(sw_params, ctx)` returns one
  predicted count rate (Hz) and one Jacobian row of shape `(N_STATE,)` per
  sweep in the fit context.
- `calculate_integral(sw_params, response_grid, rotation_matrix)` evaluates
  the same model for one sweep.
- The Jacobian columns are the analytic derivatives of the rate with respect
  to the LM state vector `(ln n, ln T, v_R, v_T, v_N)`. The density column is
  the rate itself; the log-temperature and velocity columns must agree with
  finite differences of the rate.

These tests build a real `SwapiResponse` from the shipped instrument-team CSVs
and a real `SolarWindFitContext` so that the JIT integrand and quadrature run
end-to-end."""

import math
import unittest

import numpy as np

from imap_l3_processing.constants import (
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    calculate_integral,
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    N_STATE,
    SolarWindParams,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


# Reference state used by every test in this module — typical slow solar wind:
# 5 cm⁻³, 100 kK protons, bulk velocity along the boresight in the instrument
# frame (which is +T_RTN under identity rotation, since boresight is body +Y
# and Y_inst = T_RTN at the identity).
_REF_DENSITY_CM3 = 5.0
_REF_TEMPERATURE_K = 100_000.0
_REF_BULK_SPEED_KM_S = 450.0

# The flow direction in instrument XYZ at (theta=0, phi=0) is `(0, -1, 0)`
# (look-direction convention). With identity rotation that maps to -T_RTN, so
# a bulk velocity of (0, -450, 0) RTN points right at the boresight — gives a
# maximum-rate, easy-to-reason-about geometry.
_BORESIGHT_BULK_VELOCITY_RTN = np.array([0.0, -_REF_BULK_SPEED_KM_S, 0.0])

# Off-axis bulk velocity used for the velocity-component Jacobian tests.
# Adding small components in R and N breaks the symmetry that makes vR/vN
# Jacobian entries near zero, so finite-difference noise doesn't dominate.
_OFF_AXIS_BULK_VELOCITY_RTN = np.array([30.0, -_REF_BULK_SPEED_KM_S, 20.0])

# Index aliases for the RTN velocity components inside the LM state vector.
_VELOCITY_R_IDX, _VELOCITY_T_IDX, _VELOCITY_N_IDX = (
    VELOCITY_SLICE.start,
    VELOCITY_SLICE.start + 1,
    VELOCITY_SLICE.start + 2,
)


def _esa_voltage_at_speed(speed_km_s: float) -> float:
    """ESA voltage (V) whose central proton speed equals `speed_km_s`. The
    relation is `V = ½ m_p v² / (e · k_SWAPI)`."""
    return (
        0.5
        * PROTON_MASS_KG
        * (speed_km_s * 1e3) ** 2
        / PROTON_CHARGE_COULOMBS
        / SWAPI_K_FACTOR
    )


# Putting the bulk speed at the passband center maximizes the rate and
# minimizes truncation effects in the quadrature.
_ESA_VOLTAGE_AT_REF_SPEED_V = _esa_voltage_at_speed(_REF_BULK_SPEED_KM_S)

# Numerical-differentiation parameters. `1e-5` is small enough that truncation
# from finite differences is well below the GL quadrature noise floor for the
# scales involved (≈10⁴ Hz rates, ≈10² km/s velocities) but large enough that
# round-off error stays well below 1%.
_FD_RELATIVE_STEP = 1e-5

# Tolerances for analytic-vs-finite-difference Jacobian comparison. The Jacobian
# is exact in the continuum but is evaluated with the same Gauss-Legendre
# quadrature as the rate; the small mismatch comes from the dynamic-quadrature
# limits shifting (slightly) when the state is perturbed for finite differences,
# which moves the GL nodes off the exact integrand peaks. Measured worst case
# on this fixture (off-axis bulk, 100 kK, 450 km/s) is ~2.74% on the vR column;
# 5% leaves headroom for the same machinery to stay green on adjacent fixtures
# while still flagging an analytic-derivative bug (sign flip or factor-of-2,
# which would shift the column by ~100% or ~50%). To remeasure: evaluate
# `model_solar_wind_ideal_coincidence_rates` at the test fixture state and
# compare each column against `_finite_difference_jacobian_column`.
_JACOBIAN_RTOL = 0.05
_JACOBIAN_ATOL = 1e-3


def _proton_params(
    density: float = _REF_DENSITY_CM3,
    velocity_rtn=_BORESIGHT_BULK_VELOCITY_RTN,
    temperature: float = _REF_TEMPERATURE_K,
) -> SolarWindParams:
    """Solar-wind proton state at typical slow-wind values, parameterizable
    so individual tests can perturb a single field. Signature mirrors
    `tests/swapi/l3a/science/solar_wind/test_state.py::_proton_params`."""
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.asarray(velocity_rtn, dtype=float),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


def _build_fit_context_for_voltages(
    response: SwapiResponse,
    esa_voltages: np.ndarray,
    rotation_matrices: np.ndarray = None,
):
    """Wrap `build_solar_wind_fit_context` with sensible defaults for these
    tests: identity rotations and dummy count rates."""
    voltages_array = np.asarray(esa_voltages, dtype=float)
    if rotation_matrices is None:
        rotation_matrices = np.stack([np.eye(3)] * len(voltages_array))
    response.warm_cache(voltages_array)
    return build_solar_wind_fit_context(
        count_rate=np.full(len(voltages_array), 100.0),
        esa_voltage=voltages_array,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=1.0,
    )


class _ForwardModelFixture(unittest.TestCase):
    """Loads the SwapiResponse once per class — the CSV parsing is not free
    and none of these tests mutate the response."""

    @classmethod
    def setUpClass(cls):
        cls.response = SwapiResponse.from_files(
            get_test_instrument_team_data_path(
                "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
            ),
            get_test_instrument_team_data_path(
                "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
            ),
            get_test_instrument_team_data_path(
                "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
            ),
        )


class TestModelSolarWindIdealCoincidenceRatesShape(_ForwardModelFixture):
    """`model_solar_wind_ideal_coincidence_rates` must return one rate per
    sweep and a `(N_sweeps, N_STATE)` Jacobian."""

    def test_returns_one_rate_and_one_jacobian_row_per_sweep(self):
        ctx = _build_fit_context_for_voltages(
            self.response,
            np.array(
                [
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 0.95,
                    _ESA_VOLTAGE_AT_REF_SPEED_V,
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 1.05,
                ]
            ),
        )
        rates, jacobian = model_solar_wind_ideal_coincidence_rates(
            _proton_params(), ctx
        )
        self.assertEqual(rates.shape, (3,))
        self.assertEqual(jacobian.shape, (3, N_STATE))


class TestCalculateIntegralRateBehavior(_ForwardModelFixture):
    """Behavior of the rate value (independent of the Jacobian)."""

    def setUp(self):
        self.ctx = _build_fit_context_for_voltages(
            self.response, np.array([_ESA_VOLTAGE_AT_REF_SPEED_V])
        )
        self.response_grid = self.ctx.response_grids[0]
        self.rotation_matrix = self.ctx.rotation_matrices[0]

    def test_rate_is_positive_when_bulk_velocity_aligns_with_boresight(self):
        """With the bulk speed at the passband center and the bulk direction
        along boresight, the rate must be strictly positive."""
        rate, _ = calculate_integral(
            _proton_params(), self.response_grid, self.rotation_matrix
        )
        self.assertGreater(rate, 0.0)

    def test_rate_is_linear_in_density(self):
        """The forward-model count rate is `n × (passband-weighted Maxwellian
        integral)`. Doubling density at fixed temperature and geometry must
        double the rate exactly (no quadrature dependence)."""
        rate_at_5, _ = calculate_integral(
            _proton_params(density=5.0),
            self.response_grid,
            self.rotation_matrix,
        )
        rate_at_10, _ = calculate_integral(
            _proton_params(density=10.0),
            self.response_grid,
            self.rotation_matrix,
        )
        np.testing.assert_allclose(rate_at_10, 2.0 * rate_at_5, rtol=1e-12)

    def test_returns_zero_rate_and_jacobian_when_bulk_speed_misses_passband_above(self):
        """When the bulk-speed dynamic window lies entirely above the passband
        at this voltage (passband centered at 450 km/s, half-width ≲35 km/s;
        bulk at 2000 km/s is hundreds of σ away), the model returns rate=0
        and a zero Jacobian row."""
        very_fast = _proton_params(velocity_rtn=np.array([0.0, -2000.0, 0.0]))
        rate, jacobian_row = calculate_integral(
            very_fast, self.response_grid, self.rotation_matrix
        )
        self.assertEqual(rate, 0.0)
        np.testing.assert_array_equal(jacobian_row, np.zeros(N_STATE))

    def test_returns_zero_rate_and_jacobian_when_bulk_speed_misses_passband_below(self):
        """Mirror of the above on the low side: the passband at this voltage
        spans roughly [418, 482] km/s; a bulk of 50 km/s is many σ below the
        lower edge so rate=0 with a zero Jacobian row."""
        very_slow = _proton_params(velocity_rtn=np.array([0.0, -50.0, 0.0]))
        rate, jacobian_row = calculate_integral(
            very_slow, self.response_grid, self.rotation_matrix
        )
        self.assertEqual(rate, 0.0)
        np.testing.assert_array_equal(jacobian_row, np.zeros(N_STATE))


class TestAnalyticJacobianIdentities(_ForwardModelFixture):
    """The `solar-wind-moments.md` derivation gives one closed-form identity
    that holds independent of the quadrature — `∂C/∂(ln n) = C`. Pin it."""

    def test_log_density_jacobian_column_equals_the_rate(self):
        """`f_p` is linear in `n`, so `∂f_p/∂(ln n) = f_p`. The integral
        inherits the identity: the log-density Jacobian column is exactly
        the rate, with no quadrature error."""
        ctx = _build_fit_context_for_voltages(
            self.response,
            np.array(
                [
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 0.95,
                    _ESA_VOLTAGE_AT_REF_SPEED_V,
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 1.05,
                ]
            ),
        )
        rates, jacobian = model_solar_wind_ideal_coincidence_rates(
            _proton_params(velocity_rtn=_OFF_AXIS_BULK_VELOCITY_RTN),
            ctx,
        )
        np.testing.assert_array_equal(jacobian[:, LOG_DENSITY_IDX], rates)


def _rate_at_state(state: np.ndarray, ctx) -> np.ndarray:
    """Evaluate the rate vector at a state-vector point (no Jacobian)."""
    sw_local = SolarWindParams.from_vector(state, mass=PROTON_MASS_KG)
    rates, _ = model_solar_wind_ideal_coincidence_rates(sw_local, ctx)
    return rates.copy()


def _finite_difference_jacobian_column(state: np.ndarray, ctx, j: int) -> np.ndarray:
    """Central-difference column `j` of the Jacobian: `(C(x+ε e_j) − C(x−ε e_j)) / 2ε`."""
    eps = _FD_RELATIVE_STEP * max(abs(state[j]), 1.0)
    forward = state.copy()
    forward[j] += eps
    backward = state.copy()
    backward[j] -= eps
    return (_rate_at_state(forward, ctx) - _rate_at_state(backward, ctx)) / (2 * eps)


class TestAnalyticJacobianAgainstFiniteDifferences(_ForwardModelFixture):
    """The headline contract from `docs/swapi/solar-wind-moments.md` §Analytic
    Jacobian: the per-column analytic derivatives must agree with central
    finite differences of the rate. The numerical agreement is bounded by GL
    quadrature noise that's the same for both — so the relative difference
    should be a few tenths of a percent for typical solar-wind cases."""

    def setUp(self):
        # Two-sweep context with a small voltage gradient so each sweep
        # contributes a different, non-degenerate row to the Jacobian.
        self.ctx = _build_fit_context_for_voltages(
            self.response,
            np.array(
                [
                    _ESA_VOLTAGE_AT_REF_SPEED_V,
                    _ESA_VOLTAGE_AT_REF_SPEED_V * 1.05,
                ]
            ),
        )
        # Off-axis bulk velocity so all three velocity components produce
        # nonzero Jacobian entries (boresight-aligned bulk has vR/vN ≈ 0
        # by symmetry, and the FD signal-to-noise on those columns is poor).
        self.sw = _proton_params(velocity_rtn=_OFF_AXIS_BULK_VELOCITY_RTN)
        self.state = self.sw.to_vector()

    def test_log_temperature_jacobian_is_correct(self):
        """`∂C/∂(ln T) = ∫ f_p · (|v − v_b|² / (2 v_th²) − 3/2) ...` per the
        derivation in §Temperature."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, LOG_TEMPERATURE_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, LOG_TEMPERATURE_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )

    def test_velocity_r_jacobian_is_correct(self):
        """`∂C/∂v_R = ∫ (f_p / v_th²)(v · d̂_R − v_R) ...` per §Bulk velocity
        components."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, _VELOCITY_R_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, _VELOCITY_R_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )

    def test_velocity_t_jacobian_is_correct(self):
        """Analogous to vR for the T component."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, _VELOCITY_T_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, _VELOCITY_T_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )

    def test_velocity_n_jacobian_is_correct(self):
        """Analogous to vR for the N component."""
        _, analytic_jacobian = model_solar_wind_ideal_coincidence_rates(
            self.sw, self.ctx
        )
        fd_column = _finite_difference_jacobian_column(
            self.state, self.ctx, _VELOCITY_N_IDX
        )
        np.testing.assert_allclose(
            analytic_jacobian[:, _VELOCITY_N_IDX],
            fd_column,
            rtol=_JACOBIAN_RTOL,
            atol=_JACOBIAN_ATOL,
        )


class TestNonIdentityRotation(_ForwardModelFixture):
    """The forward model is invariant under a joint rotation of the bulk
    velocity and the SWAPI→RTN matrix: rotating both by the same matrix
    leaves the rate unchanged because the integrand only sees the bulk
    direction relative to the look direction in the instrument frame."""

    def test_rate_is_invariant_under_joint_rotation_of_bulk_and_response(self):
        """Apply the same rotation `Q` to both the bulk velocity (in RTN) and
        the SWAPI→RTN matrix; the rate must match the identity-rotation
        baseline."""
        baseline_ctx = _build_fit_context_for_voltages(
            self.response, np.array([_ESA_VOLTAGE_AT_REF_SPEED_V])
        )
        baseline_rate, _ = calculate_integral(
            _proton_params(velocity_rtn=_OFF_AXIS_BULK_VELOCITY_RTN),
            baseline_ctx.response_grids[0],
            baseline_ctx.rotation_matrices[0],
        )

        # 30° rotation about RTN +R. Rotating both the bulk velocity and the
        # instrument-to-RTN matrix by the same Q preserves the bulk direction
        # in the instrument frame, which is all the integrand sees.
        angle_rad = math.radians(30.0)
        rotation_about_r = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(angle_rad), -math.sin(angle_rad)],
                [0.0, math.sin(angle_rad), math.cos(angle_rad)],
            ]
        )
        rotated_velocity_rtn = rotation_about_r @ _OFF_AXIS_BULK_VELOCITY_RTN
        rotated_xyz_to_rtn = rotation_about_r
        rotated_ctx = _build_fit_context_for_voltages(
            self.response,
            np.array([_ESA_VOLTAGE_AT_REF_SPEED_V]),
            rotation_matrices=np.stack([rotated_xyz_to_rtn]),
        )
        rotated_rate, _ = calculate_integral(
            _proton_params(velocity_rtn=rotated_velocity_rtn),
            rotated_ctx.response_grids[0],
            rotated_ctx.rotation_matrices[0],
        )

        # Agreement is limited only by GL quadrature node positions, which
        # are placed on the same bulk-frame angular window in both cases.
        np.testing.assert_allclose(rotated_rate, baseline_rate, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
