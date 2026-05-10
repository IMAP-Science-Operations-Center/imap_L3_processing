"""Tests for `solar_wind.proton.uncertainties`: the HC3 sandwich covariance
estimator, the correlated-velocity wrapper, and the DPS-frame angle/sigma
derivation. See `docs/swapi/solar-wind-moments.md` for the contracts."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from uncertainties import correlated_values, covariance_matrix, ufloat

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.uncertainties import (
    derive_uncertainties,
    derive_velocity_angles,
    make_correlated_velocity,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    LOG_DENSITY_IDX,
    N_STATE,
    SolarWindParams,
)


# Module-level fixtures: a typical proton state vector and a stand-in epoch.
_TYPICAL_DENSITY = 5.0
_TYPICAL_TEMPERATURE = 100_000.0
_TYPICAL_VELOCITY_RTN = (-450.0, 0.0, 0.0)
# The patched `rotate_rtn_to_dps` ignores its epoch arg, so the value here
# does not matter — pick 0 for clarity.
_EPOCH_TT2000_NS = 0


def _proton_sw_params(
    density: float = _TYPICAL_DENSITY,
    temperature: float = _TYPICAL_TEMPERATURE,
) -> SolarWindParams:
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.array(_TYPICAL_VELOCITY_RTN),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


def _make_optimize_result(
    jacobian: np.ndarray,
    residuals: np.ndarray,
    sw_params: SolarWindParams,
) -> MagicMock:
    """Build a minimal stand-in for `OptimizeSolarWindParamsResult`. The
    function under test only reads `.jacobian`, `.residuals`, `.sw_params`."""
    result = MagicMock()
    result.jacobian = jacobian
    result.residuals = residuals
    result.sw_params = sw_params
    return result


def _identity_rotate_rtn_to_dps(vector_rtn, _epoch_tt2000_ns):
    """Stand-in for the SPICE-driven rotation: returns the input unchanged.
    Lets tests reason about angles in the RTN basis directly."""
    return vector_rtn


class TestDeriveUncertaintiesScalings(unittest.TestCase):
    """Sanity checks on the HC3 sandwich path with a non-degenerate
    Jacobian: finite outputs and the documented per-element scaling
    `sigma_n = n * sqrt(Sigma_x[0, 0])`, `sigma_T = T * sqrt(Sigma_x[1, 1])`."""

    def setUp(self):
        # Random 5-parameter Jacobian over 20 bins. The values themselves
        # don't matter — we just need the matrix to have full column rank.
        rng = np.random.default_rng(0)
        self.jacobian = rng.normal(size=(20, N_STATE))
        # Small-amplitude residuals so the resulting sigmas are reasonable
        # in size; magnitude does not affect any of the scaling assertions.
        self.residuals = rng.normal(size=20) * 0.1
        self.sw_params = _proton_sw_params()
        self.result = _make_optimize_result(
            self.jacobian, self.residuals, self.sw_params
        )

    def test_returns_finite_density_and_temperature_sigmas(self):
        sigma_n, sigma_T, _ = derive_uncertainties(self.result, MagicMock())
        self.assertTrue(np.isfinite(sigma_n))
        self.assertGreater(sigma_n, 0.0)
        self.assertTrue(np.isfinite(sigma_T))
        self.assertGreater(sigma_T, 0.0)

    def test_returns_finite_3x3_velocity_covariance(self):
        _, _, vel_cov = derive_uncertainties(self.result, MagicMock())
        self.assertEqual(vel_cov.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(vel_cov)))

    def test_density_sigma_scales_linearly_with_density(self):
        # `sigma_n = n * sqrt(Sigma_x[0, 0])`: doubling the nominal density
        # at fixed Jacobian/residuals doubles `sigma_n`.
        sigma_n_low, _, _ = derive_uncertainties(
            _make_optimize_result(
                self.jacobian, self.residuals, _proton_sw_params(density=2.0)
            ),
            MagicMock(),
        )
        sigma_n_high, _, _ = derive_uncertainties(
            _make_optimize_result(
                self.jacobian, self.residuals, _proton_sw_params(density=4.0)
            ),
            MagicMock(),
        )
        self.assertAlmostEqual(sigma_n_high / sigma_n_low, 2.0)

    def test_temperature_sigma_scales_linearly_with_temperature(self):
        # `sigma_T = T * sqrt(Sigma_x[1, 1])`: tripling temperature triples sigma_T.
        _, sigma_T_low, _ = derive_uncertainties(
            _make_optimize_result(
                self.jacobian, self.residuals, _proton_sw_params(temperature=50_000.0)
            ),
            MagicMock(),
        )
        _, sigma_T_high, _ = derive_uncertainties(
            _make_optimize_result(
                self.jacobian, self.residuals, _proton_sw_params(temperature=150_000.0)
            ),
            MagicMock(),
        )
        self.assertAlmostEqual(sigma_T_high / sigma_T_low, 3.0)


class TestDeriveUncertaintiesHighLeverageBin(unittest.TestCase):
    """The leverage reweight `(1 - h_ii)^-2` must dominate the variance
    estimate at high-leverage bins. Without it, the LM stationarity
    condition `J^T r = 0` biases `r_i^2` low at the bins that drive the fit."""

    def test_high_leverage_row_inflates_sigma_relative_to_unweighted(self):
        # Construct a Jacobian where row 0 has a single huge entry on
        # column LOG_DENSITY_IDX. That row alone determines the density
        # direction, so its hat-matrix diagonal `h_00` ≈ 1. Other rows: a
        # diagonal pattern over the remaining cols so the matrix has full
        # column rank.
        n_bins = 20
        jacobian = np.zeros((n_bins, N_STATE))
        # Any value ≥ 100 produces h_00 ≈ 1; 1000 chosen for clear separation.
        jacobian[0, LOG_DENSITY_IDX] = 1000.0
        # Cycle through the remaining state indices on the diagonal so the
        # Jacobian has full column rank (each state column has at least one
        # row with a non-zero entry).
        for i in range(1, n_bins):
            jacobian[i, i % N_STATE] = 1.0
        residuals = np.zeros(n_bins)
        residuals[0] = 0.1  # only the high-leverage row carries residual

        result = _make_optimize_result(jacobian, residuals, _proton_sw_params())
        sigma_n_hc3, _, _ = derive_uncertainties(result, MagicMock())

        # Hard-coded expected: HC3 sigma_n ≈ 5.0, HC0 sigma_n ≈ 5e-4 — the
        # `(1 - h_ii)^-2` reweight at the 0.9999 leverage clip multiplies
        # the variance by ~1e8 -> sigma ratio ≈ 1e4. Regenerated from
        # `derive_uncertainties` on the fixture above.
        self.assertAlmostEqual(sigma_n_hc3, 4.999985000045551, places=6)

    def test_unit_leverage_row_keeps_sigma_finite(self):
        # If a row has h_ii numerically equal to 1, the unclipped
        # `(1 - h_ii)^-2` is ∞ and the variance blows up. The implementation
        # clips leverage at 0.9999, so the result is always finite.
        n_bins = 5
        jacobian = np.zeros((n_bins, N_STATE))
        jacobian[0, 0] = 1.0
        # zero out everything else so row 0 is the *only* row that
        # touches column 0 — leverage of row 0 will hit exactly 1
        residuals = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
        # remaining cols need at least *some* non-zero rows to keep pinv
        # well-defined elsewhere; populate the rest of the diagonal
        for i in range(1, min(n_bins, N_STATE)):
            jacobian[i, i] = 1.0

        result = _make_optimize_result(jacobian, residuals, _proton_sw_params())
        sigma_n, _, _ = derive_uncertainties(result, MagicMock())
        self.assertTrue(np.isfinite(sigma_n))


class TestDeriveUncertaintiesNaNInputs(unittest.TestCase):
    """A NaN-laden Jacobian propagates NaN through `np.linalg.pinv` (which
    eventually raises `LinAlgError`). The function returns the documented
    NaN sentinel instead of crashing the caller."""

    def test_nan_jacobian_returns_nan_density_temperature_and_velocity_cov(self):
        jacobian = np.full((20, N_STATE), np.nan)
        residuals = np.zeros(20)
        result = _make_optimize_result(jacobian, residuals, _proton_sw_params())

        sigma_n, sigma_T, vel_cov = derive_uncertainties(result, MagicMock())

        self.assertTrue(np.isnan(sigma_n))
        self.assertTrue(np.isnan(sigma_T))
        self.assertEqual(vel_cov.shape, (3, 3))
        self.assertTrue(np.all(np.isnan(vel_cov)))


class TestMakeCorrelatedVelocity(unittest.TestCase):
    """`make_correlated_velocity(nominal, cov)` either returns three
    correlated `UFloat`s whose pairwise covariance reproduces the input
    matrix, or — on non-finite / non-PSD covariance — falls back to
    independent `ufloat(nominal, nan)` triplets. Both branches preserve
    nominal values."""

    def setUp(self):
        self.nominal = np.array([-400.0, 5.0, -3.0])

    def test_valid_covariance_returns_three_correlated_ufloats(self):
        cov = np.array(
            [
                [4.0, 1.0, 0.5],
                [1.0, 2.0, 0.3],
                [0.5, 0.3, 1.0],
            ]
        )
        v = make_correlated_velocity(self.nominal, cov)
        self.assertEqual(len(v), 3)

    def test_valid_covariance_round_trips_through_uncertainties(self):
        cov = np.array(
            [
                [4.0, 1.0, 0.5],
                [1.0, 2.0, 0.3],
                [0.5, 0.3, 1.0],
            ]
        )
        v = make_correlated_velocity(self.nominal, cov)

        recovered_cov = np.array(covariance_matrix(v))
        np.testing.assert_allclose(recovered_cov, cov, rtol=0, atol=1e-10)
        for component, expected in zip(v, self.nominal):
            self.assertAlmostEqual(component.nominal_value, expected)

    def test_non_finite_covariance_returns_nan_sigma_ufloats(self):
        cov = np.full((3, 3), np.nan)
        v = make_correlated_velocity(self.nominal, cov)

        self.assertEqual(len(v), 3)
        for component, expected in zip(v, self.nominal):
            self.assertEqual(component.nominal_value, expected)
            self.assertTrue(np.isnan(component.std_dev))

    def test_non_finite_covariance_returns_independent_ufloats(self):
        cov = np.full((3, 3), np.nan)
        v = make_correlated_velocity(self.nominal, cov)

        # Independent UFloats have no off-diagonal correlation. Off-diagonals
        # must be either 0 or NaN — definitely not the finite values that
        # `correlated_values` would have produced.
        recovered = np.array(covariance_matrix(v))
        self.assertFalse(np.any(np.isfinite(recovered) & (recovered != 0)))

    def test_negative_eigenvalue_covariance_falls_back_to_nan_sigma_ufloats(self):
        # Diagonal with one negative eigenvalue: not PSD, so the function
        # should refuse to call `correlated_values` (which would raise) and
        # return the independent NaN-sigma fallback.
        cov = np.diag([1.0, -1.0, 1.0])
        v = make_correlated_velocity(self.nominal, cov)

        self.assertEqual(len(v), 3)
        for component, expected in zip(v, self.nominal):
            self.assertEqual(component.nominal_value, expected)
            self.assertTrue(np.isnan(component.std_dev))


@patch(
    "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
    side_effect=_identity_rotate_rtn_to_dps,
)
class TestDeriveVelocityAnglesNominals(unittest.TestCase):
    """Pin the nominal-value formulas from the `Output Variables` table:
        speed     = |v|
        clock     = atan2(-v_y, -v_x) % 360
        deflection = arccos(-v_z / |v|)
    All evaluated on the DPS-frame velocity (we mock the rotation to be
    identity so DPS == input)."""

    def setUp(self):
        # Velocity (-400, +5, -3) in RTN; clock = atan2(-5, +400) % 360,
        # deflection = arccos(+3 / |v|).
        self.cov = np.diag([4.0, 1.0, 0.5])
        self.nominal = np.array([-400.0, 5.0, -3.0])
        self.v_rtn = correlated_values(self.nominal, self.cov)

    def test_speed_nominal_is_norm_of_velocity(self, _mock_rotate):
        speed, _, _ = derive_velocity_angles(self.v_rtn, _EPOCH_TT2000_NS)
        # |(-400, 5, -3)| ≈ 400.0424977424274
        self.assertAlmostEqual(speed.nominal_value, 400.0424977424274)

    def test_clock_angle_nominal_in_zero_to_360(self, _mock_rotate):
        _, clock, _ = derive_velocity_angles(self.v_rtn, _EPOCH_TT2000_NS)
        self.assertGreaterEqual(clock.nominal_value, 0.0)
        self.assertLess(clock.nominal_value, 360.0)
        # atan2(-5, +400) ≈ -0.716° -> 359.28° after the mod 360.
        # Regenerated from `math.degrees(math.atan2(5, 400)) % 360` over
        # nominal = (-400, 5, -3).
        self.assertAlmostEqual(clock.nominal_value, 359.2838400545296)

    def test_deflection_angle_nominal_in_zero_to_180(self, _mock_rotate):
        _, _, defl = derive_velocity_angles(self.v_rtn, _EPOCH_TT2000_NS)
        self.assertGreaterEqual(defl.nominal_value, 0.0)
        self.assertLessEqual(defl.nominal_value, 180.0)
        # arccos(3 / |(-400, 5, -3)|) ≈ 89.57°.
        # Regenerated from `math.degrees(math.acos(3 / norm))` over
        # nominal = (-400, 5, -3).
        self.assertAlmostEqual(defl.nominal_value, 89.57032327652315)


@patch(
    "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
    side_effect=_identity_rotate_rtn_to_dps,
)
class TestDeriveVelocityAnglesSigmas(unittest.TestCase):
    """When the rotated velocity has finite covariance, all three sigmas
    are finite. The `circstd` call inside the Monte Carlo path handles
    angular wraparound at the ±180° clock seam — pin that by placing the
    nominal clock right at 180°."""

    def test_returns_finite_sigmas_for_finite_covariance(self, _mock_rotate):
        nominal = np.array([-400.0, 5.0, -3.0])
        cov = np.diag([4.0, 1.0, 0.5])
        v_rtn = correlated_values(nominal, cov)

        speed, clock, defl = derive_velocity_angles(v_rtn, _EPOCH_TT2000_NS)

        self.assertTrue(np.isfinite(speed.std_dev))
        self.assertGreater(speed.std_dev, 0.0)
        self.assertTrue(np.isfinite(clock.std_dev))
        self.assertGreater(clock.std_dev, 0.0)
        self.assertTrue(np.isfinite(defl.std_dev))
        self.assertGreater(defl.std_dev, 0.0)

    def test_clock_sigma_is_finite_when_nominal_is_at_the_180_degree_wrap(
        self, _mock_rotate
    ):
        # Velocity = (+400, 0, 0) in DPS gives clock = atan2(0, -400) = 180°.
        # With non-zero variance on the y-component, MC samples straddle the
        # ±180° seam; `circstd(..., high=360)` must produce a finite sigma.
        # y-variance must straddle the 180° seam, so make sigma_y comparable
        # to |v|: 100 (= 10° at |v|=400) is enough to cross the seam often.
        nominal = np.array([400.0, 0.0, 0.0])
        cov = np.diag([10.0, 100.0, 10.0])
        v_rtn = correlated_values(nominal, cov)

        _, clock, _ = derive_velocity_angles(v_rtn, _EPOCH_TT2000_NS)

        self.assertAlmostEqual(clock.nominal_value, 180.0)
        self.assertTrue(np.isfinite(clock.std_dev))
        self.assertGreater(clock.std_dev, 0.0)


@patch(
    "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
    side_effect=_identity_rotate_rtn_to_dps,
)
class TestDeriveVelocityAnglesNonFiniteCovariance(unittest.TestCase):
    """When the rotated velocity carries non-finite covariance (the
    `make_correlated_velocity` fallback case), `derive_velocity_angles`
    must return finite nominal angles but NaN sigmas for clock and
    deflection — Monte Carlo over a NaN covariance is meaningless."""

    def setUp(self):
        nominal = np.array([-400.0, 5.0, -3.0])
        # Independent NaN-sigma UFloats: covariance_matrix on this triplet
        # returns NaN entries (since std_dev is NaN), so the early-return
        # branch fires.
        self.nominal = nominal
        self.v_rtn = (
            ufloat(nominal[0], np.nan),
            ufloat(nominal[1], np.nan),
            ufloat(nominal[2], np.nan),
        )

    def test_nominal_values_are_still_finite(self, _mock_rotate):
        speed, clock, defl = derive_velocity_angles(self.v_rtn, _EPOCH_TT2000_NS)
        self.assertAlmostEqual(speed.nominal_value, float(np.linalg.norm(self.nominal)))
        self.assertGreaterEqual(clock.nominal_value, 0.0)
        self.assertLess(clock.nominal_value, 360.0)
        self.assertGreaterEqual(defl.nominal_value, 0.0)
        self.assertLessEqual(defl.nominal_value, 180.0)

    def test_clock_and_deflection_sigmas_are_nan(self, _mock_rotate):
        _, clock, defl = derive_velocity_angles(self.v_rtn, _EPOCH_TT2000_NS)
        self.assertTrue(np.isnan(clock.std_dev))
        self.assertTrue(np.isnan(defl.std_dev))


if __name__ == "__main__":
    unittest.main()
