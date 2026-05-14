import unittest
from unittest.mock import MagicMock

import numpy as np
from uncertainties import covariance_matrix

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    derive_uncertainties,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
    compute_hc3_parameter_covariance,
    make_correlated_velocity,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    N_STATE,
    SolarWindParams,
)


# Module-level fixtures: a typical proton state vector.
_TYPICAL_DENSITY = 5.0
_TYPICAL_TEMPERATURE = 100_000.0
_TYPICAL_VELOCITY_RTN = (-450.0, 0.0, 0.0)


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


class TestComputeHc3ParameterCovariance(unittest.TestCase):
    """Tests for `compute_hc3_parameter_covariance`: the parameter-space-agnostic sandwich estimator shared by the proton and alpha fitters."""

    def test_returns_p_by_p_symmetric_matrix_for_full_rank_jacobian(self):
        """A full-rank (n×p) Jacobian with non-zero residuals returns a finite, symmetric (p×p) covariance."""
        rng = np.random.default_rng(0)
        jacobian = rng.normal(size=(30, 4))
        residuals = rng.normal(size=30) * 0.1

        cov = compute_hc3_parameter_covariance(jacobian, residuals)

        self.assertEqual(cov.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(cov)))
        np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-12)

    def test_high_leverage_row_inflates_corresponding_diagonal(self):
        """A single row with leverage near 1 in column j produces a much larger diagonal cov[j,j] than the same fit without the leverage spike."""
        n_bins = 20
        p = 3
        # Baseline: rank-3 jacobian with no high-leverage row; uniform contribution.
        baseline_jacobian = np.zeros((n_bins, p))
        for i in range(n_bins):
            baseline_jacobian[i, i % p] = 1.0
        residuals = np.zeros(n_bins)
        residuals[0] = 0.1

        cov_baseline = compute_hc3_parameter_covariance(baseline_jacobian, residuals)

        # High-leverage variant: column 0 of row 0 dominates the JᵀJ for column 0.
        leveraged_jacobian = baseline_jacobian.copy()
        leveraged_jacobian[0, 0] = 1000.0
        cov_leveraged = compute_hc3_parameter_covariance(leveraged_jacobian, residuals)

        self.assertGreater(cov_leveraged[0, 0], cov_baseline[0, 0] * 1e3)

    def test_returns_all_nan_matrix_for_nan_jacobian(self):
        """An all-NaN Jacobian triggers the `LinAlgError` fallback and returns a (p×p) all-NaN matrix."""
        jacobian = np.full((20, 5), np.nan)
        residuals = np.zeros(20)

        cov = compute_hc3_parameter_covariance(jacobian, residuals)

        self.assertEqual(cov.shape, (5, 5))
        self.assertTrue(np.all(np.isnan(cov)))


class TestDeriveUncertaintiesScalings(unittest.TestCase):
    """Tests for `derive_uncertainties`: sanity checks on the HC3 sandwich path with a non-degenerate Jacobian."""

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
        """A full-rank Jacobian with non-zero residuals yields strictly positive, finite density and temperature sigmas."""
        sigma_n, sigma_T, _ = derive_uncertainties(self.result, MagicMock())
        self.assertTrue(np.isfinite(sigma_n))
        self.assertGreater(sigma_n, 0.0)
        self.assertTrue(np.isfinite(sigma_T))
        self.assertGreater(sigma_T, 0.0)

    def test_returns_finite_3x3_velocity_covariance(self):
        """The returned velocity covariance is a finite 3x3 matrix sliced from the parameter covariance."""
        _, _, vel_cov = derive_uncertainties(self.result, MagicMock())
        self.assertEqual(vel_cov.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(vel_cov)))

    def test_density_sigma_scales_linearly_with_density(self):
        """Doubling the nominal density at fixed Jacobian/residuals doubles sigma_n, pinning the `sigma_n = n * sqrt(Sigma_x[0,0])` rule."""
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
        """Tripling the nominal temperature triples sigma_T, pinning the `sigma_T = T * sqrt(Sigma_x[1,1])` rule."""
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
    """Tests for `derive_uncertainties`: the `(1 - h_ii)^-2` leverage reweight at high-leverage rows."""

    def test_high_leverage_row_inflates_sigma_relative_to_unweighted(self):
        """A row with leverage near 1 inflates sigma_n by ~1e4 over the unweighted HC0 baseline, pinned against a regenerated reference value."""
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
        """A row with leverage exactly 1 still produces a finite sigma_n because the implementation clips leverage at 0.9999 before the (1-h)^-2 reweight."""
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
    """Tests for `derive_uncertainties`: NaN-sentinel return on a degenerate Jacobian that breaks `pinv`."""

    def test_nan_jacobian_returns_nan_density_temperature_and_velocity_cov(self):
        """An all-NaN Jacobian triggers the LinAlgError fallback, returning NaN sigmas and an all-NaN 3x3 velocity covariance."""
        jacobian = np.full((20, N_STATE), np.nan)
        residuals = np.zeros(20)
        result = _make_optimize_result(jacobian, residuals, _proton_sw_params())

        sigma_n, sigma_T, vel_cov = derive_uncertainties(result, MagicMock())

        self.assertTrue(np.isnan(sigma_n))
        self.assertTrue(np.isnan(sigma_T))
        self.assertEqual(vel_cov.shape, (3, 3))
        self.assertTrue(np.all(np.isnan(vel_cov)))


class TestMakeCorrelatedVelocity(unittest.TestCase):
    """Tests for `make_correlated_velocity`: PSD-covariance correlated triplets vs the non-finite / non-PSD NaN-sigma fallback."""

    def setUp(self):
        self.nominal = np.array([-400.0, 5.0, -3.0])

    def test_valid_covariance_returns_three_correlated_ufloats(self):
        """A valid 3x3 PSD covariance produces a length-3 tuple of UFloats from `correlated_values`."""
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
        """The pairwise covariance reconstructed from the returned UFloats matches the input matrix and nominal values are preserved exactly."""
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
        """An all-NaN covariance falls back to independent UFloats that preserve the nominal values but carry NaN std_devs."""
        cov = np.full((3, 3), np.nan)
        v = make_correlated_velocity(self.nominal, cov)

        self.assertEqual(len(v), 3)
        for component, expected in zip(v, self.nominal):
            self.assertEqual(component.nominal_value, expected)
            self.assertTrue(np.isnan(component.std_dev))

    def test_non_finite_covariance_returns_independent_ufloats(self):
        """The fallback UFloats from a NaN covariance are independent, so the recovered covariance matrix has no finite non-zero off-diagonals."""
        cov = np.full((3, 3), np.nan)
        v = make_correlated_velocity(self.nominal, cov)

        # Independent UFloats have no off-diagonal correlation. Off-diagonals
        # must be either 0 or NaN — definitely not the finite values that
        # `correlated_values` would have produced.
        recovered = np.array(covariance_matrix(v))
        self.assertFalse(np.any(np.isfinite(recovered) & (recovered != 0)))

    def test_negative_eigenvalue_covariance_falls_back_to_nan_sigma_ufloats(self):
        """A finite but non-PSD covariance (diag with a negative eigenvalue) routes to the NaN-sigma fallback rather than letting `correlated_values` raise."""
        cov = np.diag([1.0, -1.0, 1.0])
        v = make_correlated_velocity(self.nominal, cov)

        self.assertEqual(len(v), 3)
        for component, expected in zip(v, self.nominal):
            self.assertEqual(component.nominal_value, expected)
            self.assertTrue(np.isnan(component.std_dev))


if __name__ == "__main__":
    unittest.main()
