"""Tests for `solar_wind.alpha.calculate_alpha_solar_wind_moments` — the
Stage-2 alpha moments fitter.

Behavior spec lives in `docs/swapi/solar-wind-moments.md` § Alpha Particle
Moments. Stage-1 (the proton refit on coarse-only bins) is run by the
chunk-fits caller and passed into `fit_solar_wind_alpha_moments` as a
`ProtonSolarWindFitResult`; the function under test does **not** refit
protons, it only consumes them. Stage-2 fits `(n_α, T_α, Δv)` where
`v_α = v_p* + Δv·B̂`.
"""

import unittest
from unittest.mock import MagicMock, patch

import numba
import numpy as np
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha import (
    calculate_alpha_solar_wind_moments as alpha_module,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_alpha_solar_wind_moments import (
    AlphaSolarWindMoments,
    _alpha_initial_guess,
    _infer_sweep_layout,
    _nan_alpha_moments,
    fit_solar_wind_alpha_moments,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


# ----- module-level fixture constants --------------------------------------

# Calibration CSVs shipped with the repo. Loading the full SwapiResponse
# triggers the same code path the production pipeline uses.
_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)

# 5-sweep coarse-only voltage axis: 62 bins/sweep × 5 sweeps = 310 measurements,
# matching the Stage-2 axis described in the doc. The voltage values themselves
# are arbitrary log-spaced decreasing — `get_alpha_peak_indices` requires
# strictly decreasing energies, but the exact endpoints are not load-bearing
# here. The L2 voltage range divided by `SWAPI_L2_K_FACTOR` would give the
# physical voltages SWAPI sweeps in flight.
_N_BINS_PER_SWEEP = 62
_N_SWEEPS = 5
_ONE_SWEEP_VOLTAGE = np.logspace(
    np.log10(3500.0), np.log10(140.0), _N_BINS_PER_SWEEP
)
_FIVE_SWEEP_VOLTAGE = np.tile(_ONE_SWEEP_VOLTAGE, _N_SWEEPS)
_N_MEAS = len(_FIVE_SWEEP_VOLTAGE)

# Slow-wind ground-truth moments. RTN +R points sunward, so a sunward solar
# wind has v_R = -450 km/s. B̂ is taken along -R̂ so a positive Δv pushes
# alphas toward more-negative v_R (along +B̂).
_TRUE_PROTON_DENSITY_CM3 = 5.0
_TRUE_PROTON_TEMPERATURE_K = 1.0e5
_TRUE_PROTON_VELOCITY_RTN = np.array([-450.0, 0.0, 0.0])
_TRUE_ALPHA_DENSITY_CM3 = 0.2
_TRUE_ALPHA_TEMPERATURE_K = 4.0e5
_TRUE_DELTA_V_KM_S = 30.0
_B_HAT_RTN = np.array([-1.0, 0.0, 0.0])

# Stage-1 proton uncertainties used to seed `ProtonSolarWindFitResult`.
# Values are chosen small but nonzero so `bulk_velocity_rtn_covariance`
# (used downstream by `fit_solar_wind_alpha_moments` to add
# `σ_Δv²·B̂B̂ᵀ`) is well-defined. The exact values are not pinned by any
# test — they only need to be finite and positive.
_STAGE1_PROTON_DENSITY_SIGMA_CM3 = 0.05
_STAGE1_PROTON_TEMPERATURE_SIGMA_K = 1.0e3
_STAGE1_PROTON_VELOCITY_SIGMA_KM_S = 1.0

# Predicted ESA voltage of the alpha bump on the fixture spectrum. SWAPI's
# central-speed conversion is `v² = 2·k·V·(e/m_p) / (m/q in m_p/e)`, so
# inverting for V at the alpha truth speed (|v_p| + Δv along -R̂, ≈ 480 km/s)
# gives V_α ≈ 1264 V on this fixture.
_ALPHA_TRUTH_SPEED_M_S = (
    float(np.linalg.norm(_TRUE_PROTON_VELOCITY_RTN)) + _TRUE_DELTA_V_KM_S
) * 1.0e3
_ALPHA_PEAK_VOLTAGE = (
    _ALPHA_TRUTH_SPEED_M_S**2
    * ALPHA_MASS_PER_CHARGE_M_P_PER_E
    / (2.0 * SWAPI_K_FACTOR * (PROTON_CHARGE_COULOMBS / PROTON_MASS_KG))
)


# ----- helpers --------------------------------------------------------------


def _load_swapi_response_with_warm_cache() -> SwapiResponse:
    """Load `SwapiResponse` from the shipped instrument-team CSVs and warm
    its passband cache for the fixture voltages.

    `create_response_grid` raises if the passband cache isn't warm for the
    requested voltage, so callers building a fit context off this response
    must warm it first."""
    response = SwapiResponse.from_files(
        _AZIMUTHAL_TRANSMISSION_PATH,
        _CENTRAL_EFFECTIVE_AREA_PATH,
        _PASSBAND_FIT_COEFFICIENTS_PATH,
    )
    response.warm_cache(_FIVE_SWEEP_VOLTAGE)
    return response


def _identity_rotation_matrices(n: int = _N_MEAS) -> np.ndarray:
    """Identity SWAPI→RTN rotations so the instrument frame coincides with
    RTN — keeps the wind-direction interpretation straightforward.

    `np.broadcast_to` produces a read-only view; `.copy()` materialises a
    contiguous writeable array, which the JIT'd Stage-2 forward model
    expects."""
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def _build_proton_fit_result(
    *,
    density: float = _TRUE_PROTON_DENSITY_CM3,
    temperature: float = _TRUE_PROTON_TEMPERATURE_K,
    velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
    bad_fit_flag: int = int(SwapiL3Flags.NONE),
) -> ProtonSolarWindFitResult:
    """Build a `ProtonSolarWindFitResult` (Stage-1 output) with small but
    nonzero uncertainties so `bulk_velocity_rtn_covariance` is well-defined
    and Stage-2 has something to add `σ_Δv²·B̂B̂ᵀ` to."""
    return ProtonSolarWindFitResult(
        density=ufloat(density, _STAGE1_PROTON_DENSITY_SIGMA_CM3),
        temperature=ufloat(temperature, _STAGE1_PROTON_TEMPERATURE_SIGMA_K),
        bulk_velocity_rtn=(
            ufloat(velocity_rtn[0], _STAGE1_PROTON_VELOCITY_SIGMA_KM_S),
            ufloat(velocity_rtn[1], _STAGE1_PROTON_VELOCITY_SIGMA_KM_S),
            ufloat(velocity_rtn[2], _STAGE1_PROTON_VELOCITY_SIGMA_KM_S),
        ),
        bad_fit_flag=int(bad_fit_flag),
    )


def _build_empty_alpha_ctx() -> SolarWindFitContext:
    """Build a zero-measurement `SolarWindFitContext` with the alpha mass.

    `build_solar_wind_fit_context` would happily build this since it only
    drops invalid voltages, but we want to also exercise the
    `len(esa_voltage) == 0` short-circuit explicitly."""
    return SolarWindFitContext(
        count_rate=np.array([], dtype=float),
        esa_voltage=np.array([], dtype=float),
        # Empty typed list — element type is irrelevant since no iteration
        # happens before the n_meas==0 short-circuit.
        response_grids=numba.typed.List.empty_list(item_type=numba.types.int64),
        rotation_matrices=np.zeros((0, 3, 3)),
        mass_kg=ALPHA_PARTICLE_MASS_KG,
    )


def _assert_moments_are_nan_filled(test, result) -> None:
    """Assert that every moment field on an `AlphaSolarWindMoments`
    fill-valued result is NaN. Used by the two pre-fit guard branches
    (Stage-1 failed, MAG B̂ NaN) which both short-circuit to fill values."""
    test.assertTrue(np.isnan(result.density.nominal_value))
    test.assertTrue(np.isnan(result.temperature.nominal_value))
    test.assertTrue(np.isnan(result.delta_v.nominal_value))
    for component in result.bulk_velocity_rtn:
        test.assertTrue(np.isnan(component.nominal_value))


def _synthesize_proton_plus_alpha_count_rate(
    *,
    response: SwapiResponse,
    voltage: np.ndarray,
    rotation_matrices: np.ndarray,
    proton_density: float = _TRUE_PROTON_DENSITY_CM3,
    proton_temperature: float = _TRUE_PROTON_TEMPERATURE_K,
    proton_velocity_rtn: np.ndarray = _TRUE_PROTON_VELOCITY_RTN,
    alpha_density: float = _TRUE_ALPHA_DENSITY_CM3,
    alpha_temperature: float = _TRUE_ALPHA_TEMPERATURE_K,
    delta_v: float = _TRUE_DELTA_V_KM_S,
    b_hat: np.ndarray = _B_HAT_RTN,
):
    """Forward-model deadtime-applied count rates for a (proton + alpha)
    spectrum on the fixture voltage axis. Pass `alpha_density=0.0` for a
    proton-only spectrum.

    Returns the observed count rate, the (pre-deadtime) proton-only true
    rate, and the alpha velocity v_α = v_p + Δv·B̂. The proton true rate is
    the same Stage-2 calls "frozen proton model rate" — useful for
    `_alpha_initial_guess` tests as well."""
    proton_params = SolarWindParams(
        density=proton_density,
        bulk_velocity_rtn=np.asarray(proton_velocity_rtn, dtype=float),
        temperature=proton_temperature,
        mass=PROTON_MASS_KG,
    )
    alpha_velocity = np.asarray(proton_velocity_rtn, dtype=float) + delta_v * b_hat
    alpha_params = SolarWindParams(
        density=alpha_density,
        bulk_velocity_rtn=alpha_velocity,
        temperature=alpha_temperature,
        mass=ALPHA_PARTICLE_MASS_KG,
    )
    proton_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltage)),
        esa_voltage=voltage,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    proton_true, _ = model_solar_wind_ideal_coincidence_rates(proton_params, proton_ctx)
    if alpha_density > 0.0:
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=np.zeros(len(voltage)),
            esa_voltage=voltage,
            swapi_response=response,
            central_effective_area_scale=1.0,
            rotation_matrices=rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        alpha_true, _ = model_solar_wind_ideal_coincidence_rates(alpha_params, alpha_ctx)
        total_true = proton_true + alpha_true
    else:
        total_true = proton_true
    observed = total_true * deadtime_factor(total_true)
    return observed, proton_true, alpha_velocity


class _SyntheticAlphaSpectrumFixture:
    """Cached forward-model fixture: builds the SwapiResponse, warm-caches
    its passband, and synthesizes a (proton + alpha) count-rate spectrum
    once for all subclass tests. Class-level so JIT compile + the pandas
    pivot inside `_build_passband_array` are paid one time."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache()
        cls.voltage = _FIVE_SWEEP_VOLTAGE
        cls.rotation_matrices = _identity_rotation_matrices()
        observed, proton_true, alpha_v = _synthesize_proton_plus_alpha_count_rate(
            response=cls.response,
            voltage=cls.voltage,
            rotation_matrices=cls.rotation_matrices,
        )
        cls.observed_count_rate = observed
        cls.proton_true_rate = proton_true
        cls.alpha_velocity_rtn = alpha_v


# ----- _infer_sweep_layout --------------------------------------------------


class TestInferSweepLayout(unittest.TestCase):
    """`_infer_sweep_layout(esa_voltage)` is the heuristic used by Stage-2 to
    reshape flat per-measurement arrays into `(n_sweeps, n_bins)` for the
    sweep-averaged peak-finder."""

    def test_clean_5_sweep_pattern_returns_5_and_per_sweep_bin_count(self):
        # 4 unique voltages tiled 5 times — the pipeline's canonical Stage-2
        # layout. `5` is tried first so this is the expected return.
        one_sweep = np.array([100.0, 200.0, 300.0, 400.0])
        five_sweeps = np.tile(one_sweep, 5)
        n_sweeps, n_bins = _infer_sweep_layout(five_sweeps)
        self.assertEqual(n_sweeps, 5)
        self.assertEqual(n_bins, 4)

    def test_non_5_sweep_pattern_returns_single_sweep(self):
        # For any non-5-tiled array, the search hits `n_sweeps=1` before any
        # other divisor, since 1 is checked second and always passes the
        # trivial reshape-equality test. So a genuine 2-sweep tile is
        # reported as `(1, n_meas)`, not `(2, n_meas/2)`.
        two_sweeps = np.array([100.0, 200.0, 300.0, 400.0, 100.0, 200.0, 300.0, 400.0])
        n_sweeps, n_bins = _infer_sweep_layout(two_sweeps)
        self.assertEqual(n_sweeps, 1)
        self.assertEqual(n_bins, 8)

    def test_non_periodic_input_returns_single_sweep(self):
        # `n_sweeps=1, n_bins=n_meas` trivially satisfies the
        # reshape-equality check, so non-periodic inputs land on the
        # single-sweep layout.
        irregular = np.array([100.0, 200.0, 300.0, 110.0, 210.0, 305.0, 99.0])
        n_sweeps, n_bins = _infer_sweep_layout(irregular)
        self.assertEqual(n_sweeps, 1)
        self.assertEqual(n_bins, 7)

    def test_5_sweep_search_order_beats_1(self):
        # Both `5×N` and `1×5N` would satisfy the check on a 5-tiled axis.
        # The function tries 5 first, so the canonical Stage-2 axis returns
        # 5 sweeps even though 1 would also match.
        five_sweeps = np.tile(np.array([100.0, 200.0, 300.0]), 5)
        n_sweeps, n_bins = _infer_sweep_layout(five_sweeps)
        self.assertEqual(n_sweeps, 5)
        self.assertEqual(n_bins, 3)


# ----- _alpha_initial_guess -------------------------------------------------


class TestAlphaInitialGuessRecoversAlphaBump(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """`_alpha_initial_guess` should locate the alpha bump on a clean
    (proton + alpha) spectrum and return a (n_α, T_α, Δv=0, peak_indices)
    seed that brackets the truth.

    Doc § Alpha § Initial guess — the residual is `max(0, mean_obs − 2·deadtime(proton_true))`
    averaged across sweeps, the peak is found by `get_alpha_peak_indices`,
    and density is scaled from a unit-density alpha forward model at Δv=0.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Build an alpha-only fit context and inject the *observed* count
        # rate so `_alpha_initial_guess` can read sweep-averaged counts off
        # `alpha_ctx.count_rate`.
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=cls.observed_count_rate,
            esa_voltage=cls.voltage,
            swapi_response=cls.response,
            central_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        cls.alpha_ctx = alpha_ctx
        cls.guess = _alpha_initial_guess(
            proton_true_rate=cls.proton_true_rate,
            proton_temperature=_TRUE_PROTON_TEMPERATURE_K,
            alpha_ctx=alpha_ctx,
            proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
        )

    def test_returns_seed_when_spectrum_has_alpha_bump(self):
        self.assertEqual(len(self.guess or ()), 4)

    def test_density_seed_brackets_truth(self):
        # The seed is `mean(residual_peak) / mean(unit_density_alpha_peak)`
        # averaged across sweeps; the doc only requires it as a starting
        # point for LM, not high accuracy. The factor-of-2 tolerance below
        # is a smoke test, not a contract — empirically it lands within
        # ±50% of truth on a clean spectrum.
        n_alpha_seed, _, _, _ = self.guess
        ratio = n_alpha_seed / _TRUE_ALPHA_DENSITY_CM3
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_temperature_seed_equals_proton_temperature(self):
        # Doc § Alpha § Initial guess step 5: the alpha thermal width is fit
        # by LM in Stage 2 — the seed is just T_α = T_p*.
        _, t_alpha_seed, _, _ = self.guess
        self.assertEqual(t_alpha_seed, _TRUE_PROTON_TEMPERATURE_K)

    def test_delta_v_seed_is_zero(self):
        # Doc § Alpha § Initial guess step 7: Δv=0 starts the optimizer and
        # the wrong-basin flip handles sign ambiguity afterward.
        _, _, dv_seed, _ = self.guess
        self.assertEqual(dv_seed, 0.0)

    def test_peak_indices_have_at_least_three_bins(self):
        # Doc § Alpha § Initial guess step 4: guard requires ≥3 bins in the
        # detected peak.
        _, _, _, peak_indices = self.guess
        self.assertGreaterEqual(len(peak_indices), 3)

    def test_peak_indices_bracket_true_alpha_voltage(self):
        # The alpha bump should appear at the ESA voltage `_ALPHA_PEAK_VOLTAGE`
        # (≈ 1264 V) derived from the proton truth velocity, m/q ratio, and
        # SWAPI k-factor — verify the peak window encloses it.
        _, _, _, peak_indices = self.guess
        peak_voltage = _ONE_SWEEP_VOLTAGE[peak_indices]
        # Voltages are decreasing in `peak_indices` (which is increasing)
        # — high V at low index, low V at high index.
        self.assertGreater(peak_voltage.max(), _ALPHA_PEAK_VOLTAGE)
        self.assertLess(peak_voltage.min(), _ALPHA_PEAK_VOLTAGE)


class TestAlphaInitialGuessFailureBranches(unittest.TestCase):
    """Failure paths in `_alpha_initial_guess` — every guard must return
    `None` (not raise) so the alpha fitter can downgrade cleanly to
    `FIT_FAILED` instead of crashing the chunk."""

    def test_returns_none_when_alpha_ctx_is_empty(self):
        # `_alpha_initial_guess` short-circuits when there are zero
        # measurements.
        empty_ctx = _build_empty_alpha_ctx()
        guess = _alpha_initial_guess(
            proton_true_rate=np.array([]),
            proton_temperature=1.0e5,
            alpha_ctx=empty_ctx,
            proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
        )
        self.assertIsNone(guess)

    def test_returns_none_when_residual_has_no_alpha_bump(self):
        # If the spectrum is proton-only (no alpha bump), the residual
        # `max(0, mean_obs − 2·proton_obs)` collapses to ~0 across every
        # bin, `get_alpha_peak_indices` cannot find a local minimum that
        # bounds the alpha peak on the proton side, and the function
        # returns `None` (the underlying `Exception` is caught).
        response = _load_swapi_response_with_warm_cache()
        rotations = _identity_rotation_matrices()
        proton_only_obs, proton_true, _ = _synthesize_proton_plus_alpha_count_rate(
            response=response,
            voltage=_FIVE_SWEEP_VOLTAGE,
            rotation_matrices=rotations,
            alpha_density=0.0,
        )
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=proton_only_obs,
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            swapi_response=response,
            central_effective_area_scale=1.0,
            rotation_matrices=rotations,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        guess = _alpha_initial_guess(
            proton_true_rate=proton_true,
            proton_temperature=_TRUE_PROTON_TEMPERATURE_K,
            alpha_ctx=alpha_ctx,
            proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
        )
        self.assertIsNone(guess)


# ----- fit_solar_wind_alpha_moments — guard conditions ---------------------


class TestFitAlphaMomentsGuardBranches(unittest.TestCase):
    """`fit_solar_wind_alpha_moments` has two pre-fit guards (Stage-1
    failure and missing/non-finite MAG B̂) that must short-circuit before
    any forward-model evaluation, returning a NaN-filled
    `AlphaSolarWindMoments` with the correct quality flag set."""

    def test_stale_proton_flag_when_stage1_proton_fit_failed(self):
        # Doc § Alpha § Quality flags: STALE_PROTON (=32) is set when the
        # Stage-1 proton fit failed (proton `bad_fit_flag != NONE`).
        proton_moments = _build_proton_fit_result(
            bad_fit_flag=int(SwapiL3Flags.FIT_FAILED)
        )
        result = fit_solar_wind_alpha_moments(
            count_rate=np.zeros(_N_MEAS),
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            measurement_time=np.zeros(_N_MEAS),
            # `swapi_response` should not be touched on this branch — but
            # pass a real one so an unexpected dereference would not crash.
            swapi_response=_load_swapi_response_with_warm_cache(),
            proton_moments=proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=_identity_rotation_matrices(),
        )
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.STALE_PROTON))

    def test_stale_proton_returns_nan_filled_moments(self):
        proton_moments = _build_proton_fit_result(
            bad_fit_flag=int(SwapiL3Flags.FIT_FAILED)
        )
        result = fit_solar_wind_alpha_moments(
            count_rate=np.zeros(_N_MEAS),
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=_load_swapi_response_with_warm_cache(),
            proton_moments=proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=_identity_rotation_matrices(),
        )
        _assert_moments_are_nan_filled(self, result)

    def test_fit_failed_flag_when_magnetic_field_direction_has_nans(self):
        # Doc § Alpha § Quality flags / Two-stage strategy: MAG is required;
        # per-chunk gaps (NaN B̂) must short-circuit to a fill-valued
        # result. The source uses `FIT_FAILED` for this branch (the
        # `MAG_GAP` flag is set higher up in chunk_fits.py before this
        # function is called).
        proton_moments = _build_proton_fit_result()
        nan_b_hat = np.array([np.nan, 0.0, 0.0])
        result = fit_solar_wind_alpha_moments(
            count_rate=np.zeros(_N_MEAS),
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=_load_swapi_response_with_warm_cache(),
            proton_moments=proton_moments,
            magnetic_field_direction=nan_b_hat,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=_identity_rotation_matrices(),
        )
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.FIT_FAILED))

    def test_nan_magnetic_field_direction_returns_nan_filled_moments(self):
        proton_moments = _build_proton_fit_result()
        nan_b_hat = np.array([np.nan, 0.0, 0.0])
        result = fit_solar_wind_alpha_moments(
            count_rate=np.zeros(_N_MEAS),
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=_load_swapi_response_with_warm_cache(),
            proton_moments=proton_moments,
            magnetic_field_direction=nan_b_hat,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=_identity_rotation_matrices(),
        )
        _assert_moments_are_nan_filled(self, result)


# ----- fit_solar_wind_alpha_moments — context construction -----------------


class TestFitAlphaMomentsContextConstruction(unittest.TestCase):
    """`fit_solar_wind_alpha_moments` must build *two* fit contexts: one
    for the frozen proton background (carrying the proton mass-per-charge
    and proton effective-area scale) and one for the alpha bump (carrying
    the alpha mass-per-charge and alpha effective-area scale).

    Doc § Alpha § Species-dependent pieces requires `mass_per_charge =
    ALPHA_MASS_PER_CHARGE_M_P_PER_E` for the alpha grid; doc § Effective-
    area scaling requires the alpha CEM-efficiency scale to flow through
    to `central_effective_area_scale` for the alpha context (and the
    proton scale for the proton context)."""

    def setUp(self):
        self.proton_moments = _build_proton_fit_result()
        self.swapi_response = _load_swapi_response_with_warm_cache()
        self.proton_eff_scale = 0.987
        self.alpha_eff_scale = 0.731
        self.rotation_matrices = _identity_rotation_matrices()

        build_ctx_patcher = patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.alpha."
            "calculate_alpha_solar_wind_moments.build_solar_wind_fit_context"
        )
        initial_guess_patcher = patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.alpha."
            "calculate_alpha_solar_wind_moments._alpha_initial_guess",
            return_value=None,
        )
        forward_model_patcher = patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.alpha."
            "calculate_alpha_solar_wind_moments.model_solar_wind_ideal_coincidence_rates",
            return_value=(np.zeros(_N_MEAS), np.zeros((_N_MEAS, 5))),
        )
        self.addCleanup(build_ctx_patcher.stop)
        self.addCleanup(initial_guess_patcher.stop)
        self.addCleanup(forward_model_patcher.stop)
        self.mock_build_ctx = build_ctx_patcher.start()
        initial_guess_patcher.start()
        forward_model_patcher.start()
        self.mock_build_ctx.return_value = MagicMock()

        fit_solar_wind_alpha_moments(
            count_rate=np.zeros(_N_MEAS),
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=self.swapi_response,
            proton_moments=self.proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
            alpha_effective_area_scale=self.alpha_eff_scale,
            proton_effective_area_scale=self.proton_eff_scale,
            rotation_matrices=self.rotation_matrices,
        )
        self.proton_call_kwargs = self.mock_build_ctx.call_args_list[0].kwargs
        self.alpha_call_kwargs = self.mock_build_ctx.call_args_list[1].kwargs

    def test_two_contexts_are_built(self):
        # One call for the proton context, one for the alpha context.
        self.assertEqual(self.mock_build_ctx.call_count, 2)

    def test_proton_context_uses_proton_mass_per_charge(self):
        self.assertEqual(
            self.proton_call_kwargs["mass_per_charge_m_p_per_e"],
            PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )

    def test_alpha_context_uses_alpha_mass_per_charge(self):
        # Doc § Alpha § Species-dependent pieces — the alpha grid must use
        # mass_per_charge = ALPHA_MASS_PER_CHARGE_M_P_PER_E (≈2 m_p/e), not
        # the proton 1.0.
        self.assertEqual(
            self.alpha_call_kwargs["mass_per_charge_m_p_per_e"],
            ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )

    def test_proton_context_uses_proton_mass_kg(self):
        self.assertEqual(self.proton_call_kwargs["mass_kg"], PROTON_MASS_KG)

    def test_alpha_context_uses_alpha_mass_kg(self):
        self.assertEqual(self.alpha_call_kwargs["mass_kg"], ALPHA_PARTICLE_MASS_KG)

    def test_proton_context_uses_proton_effective_area_scale(self):
        # Doc § Alpha § Effective-area scaling — proton EA scale flows to
        # the proton context's `central_effective_area_scale`.
        self.assertEqual(
            self.proton_call_kwargs["central_effective_area_scale"],
            self.proton_eff_scale,
        )

    def test_alpha_context_uses_alpha_effective_area_scale(self):
        # Doc § Alpha § Effective-area scaling — alpha EA scale (which folds
        # the alpha species correction A_α/A_p with alpha time drift into a
        # single ratio) flows to the alpha context's
        # `central_effective_area_scale`.
        self.assertEqual(
            self.alpha_call_kwargs["central_effective_area_scale"],
            self.alpha_eff_scale,
        )

    def test_alpha_and_proton_effective_area_scales_are_distinct_values(self):
        # Sanity: the two species use different scales — guards against a
        # regression that wires both contexts to the same EA scale.
        self.assertNotEqual(
            self.alpha_call_kwargs["central_effective_area_scale"],
            self.proton_call_kwargs["central_effective_area_scale"],
        )


# ----- fit_solar_wind_alpha_moments — end-to-end recovery -----------------


class TestFitAlphaMomentsRecoversTruth(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """End-to-end Stage-2 fit: synthesize a proton+alpha spectrum from
    known truth, hand the fitter the *exact* proton moments (so Stage-2's
    "frozen proton" assumption matches the synthesis), and verify the
    fitter recovers the alpha moments and Δv.

    This pins:
    * Two-stage strategy (frozen proton, Stage-2 fits only alpha).
    * Field-aligned alpha drift constraint v_α = v_p + Δv·B̂.
    * Wrong-basin flip — exact recovery of |Δv|, sign included.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.proton_moments = _build_proton_fit_result()
        cls.result = fit_solar_wind_alpha_moments(
            count_rate=cls.observed_count_rate,
            esa_voltage=cls.voltage,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=cls.response,
            proton_moments=cls.proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
        )

    def test_fit_succeeds_with_no_quality_flags(self):
        self.assertEqual(self.result.bad_fit_flag, int(SwapiL3Flags.NONE))

    def test_recovers_alpha_density(self):
        # Forward-model synthesis means the fit is exact to LM tolerance.
        # Allow ~1% to absorb rounding through deadtime + LM termination.
        np.testing.assert_allclose(
            self.result.density.nominal_value,
            _TRUE_ALPHA_DENSITY_CM3,
            rtol=1e-2,
        )

    def test_recovers_alpha_temperature(self):
        np.testing.assert_allclose(
            self.result.temperature.nominal_value,
            _TRUE_ALPHA_TEMPERATURE_K,
            rtol=1e-2,
        )

    def test_recovers_signed_delta_v(self):
        # The wrong-basin flip means a positive truth Δv can converge from
        # either basin — this checks both magnitude and sign are recovered.
        # `atol=0.5 km/s` (~1.7% of the 30 km/s truth) is a loose bound on
        # LM termination + deadtime rounding; the empirical residual is
        # well below this.
        np.testing.assert_allclose(
            self.result.delta_v.nominal_value, _TRUE_DELTA_V_KM_S, atol=0.5
        )

    def test_alpha_velocity_equals_proton_velocity_plus_delta_v_along_bhat(self):
        # Doc § Alpha: v_α = v_p* + Δv·B̂. The dataclass stores `bulk_velocity_rtn`
        # as the post-fit alpha velocity in RTN; verify it matches the
        # field-aligned constraint exactly (not approximately — this is an
        # algebraic identity, not a fit residual).
        v_alpha = self.result.bulk_velocity_rtn_nominal()
        expected = (
            _TRUE_PROTON_VELOCITY_RTN
            + self.result.delta_v.nominal_value * _B_HAT_RTN
        )
        np.testing.assert_allclose(v_alpha, expected, atol=1e-9)

    def test_alpha_velocity_recovers_truth(self):
        # Combination of the above two: the alpha velocity vector matches
        # the synthesis truth to LM tolerance. `atol=1.0 km/s` (~0.2% of
        # the 480 km/s alpha speed) is the same scale as the Δv tolerance
        # above; tighter than the per-component contribution from
        # `_STAGE1_PROTON_VELOCITY_SIGMA_KM_S`.
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn_nominal(),
            self.alpha_velocity_rtn,
            atol=1.0,
        )


class TestFitAlphaMomentsAlphaVelocityFollowsBHat(unittest.TestCase):
    """The field-aligned drift constraint v_α = v_p + Δv·B̂ must hold for any
    B̂ direction, not just B̂ ∥ -R̂. Synthesize a fit using a tilted B̂ and
    verify the recovered alpha velocity sits on the line through v_p along
    that B̂."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache()
        cls.rotation_matrices = _identity_rotation_matrices()
        cls.proton_velocity_rtn = _TRUE_PROTON_VELOCITY_RTN
        # B̂ tilted away from -R̂ by `tilt_deg` in the R-T plane.
        tilt_deg = 10.0
        tilt_rad = np.deg2rad(tilt_deg)
        cls.b_hat = np.array([-np.cos(tilt_rad), np.sin(tilt_rad), 0.0])
        observed, _proton_true, alpha_v = _synthesize_proton_plus_alpha_count_rate(
            response=cls.response,
            voltage=_FIVE_SWEEP_VOLTAGE,
            rotation_matrices=cls.rotation_matrices,
            proton_velocity_rtn=cls.proton_velocity_rtn,
            delta_v=_TRUE_DELTA_V_KM_S,
            b_hat=cls.b_hat,
        )
        cls.alpha_velocity_truth = alpha_v
        cls.proton_moments = _build_proton_fit_result(
            velocity_rtn=cls.proton_velocity_rtn
        )
        cls.result = fit_solar_wind_alpha_moments(
            count_rate=observed,
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=cls.response,
            proton_moments=cls.proton_moments,
            magnetic_field_direction=cls.b_hat,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
        )

    def test_alpha_velocity_minus_proton_velocity_is_parallel_to_bhat(self):
        # `v_α − v_p` must lie along ±B̂, so `(v_α − v_p) × B̂` must be
        # zero up to numerical noise.
        delta = (
            self.result.bulk_velocity_rtn_nominal() - self.proton_velocity_rtn
        )
        np.testing.assert_allclose(np.cross(delta, self.b_hat), 0.0, atol=1e-9)

    def test_recovered_delta_v_matches_dot_product_of_velocity_offset(self):
        # `Δv = (v_α − v_p)·B̂` since B̂ is unit-normed.
        delta_along = float(
            np.dot(
                self.result.bulk_velocity_rtn_nominal() - self.proton_velocity_rtn,
                self.b_hat,
            )
        )
        np.testing.assert_allclose(
            self.result.delta_v.nominal_value, delta_along, atol=1e-9
        )

    def test_alpha_velocity_recovers_truth_under_tilted_bhat(self):
        np.testing.assert_allclose(
            self.result.bulk_velocity_rtn_nominal(),
            self.alpha_velocity_truth,
            atol=1.0,
        )


# ----- AlphaSolarWindMoments accessors --------------------------------------


class TestAlphaSolarWindMomentsAccessors(unittest.TestCase):
    """`AlphaSolarWindMoments.bulk_velocity_rtn_nominal` and
    `bulk_velocity_rtn_covariance` extract the nominal vector and covariance
    matrix from the correlated UFloat triple stored in `bulk_velocity_rtn`.
    Mirrors the equivalent accessors on `ProtonSolarWindFitResult`."""

    def setUp(self):
        # Build moments from a known correlated velocity covariance so the
        # accessor outputs are predictable. Using `make_correlated_velocity`
        # would require reconstructing the same covariance the alpha
        # pipeline produces; building the UFloat triple directly is enough
        # to exercise the accessors.
        from imap_l3_processing.swapi.l3a.science.solar_wind.proton.uncertainties import (
            make_correlated_velocity,
        )

        self.expected_nominal = np.array([-450.0, 10.0, -5.0])
        # Symmetric, positive-definite — `correlated_values` requires PSD.
        self.expected_covariance = np.array(
            [
                [4.0, 1.0, 0.5],
                [1.0, 9.0, 0.25],
                [0.5, 0.25, 16.0],
            ]
        )
        velocity_triple = make_correlated_velocity(
            self.expected_nominal, self.expected_covariance
        )
        self.moments = AlphaSolarWindMoments(
            density=ufloat(0.2, 0.01),
            temperature=ufloat(4.0e5, 1.0e3),
            bulk_velocity_rtn=velocity_triple,
            delta_v=ufloat(30.0, 1.0),
            bad_fit_flag=int(SwapiL3Flags.NONE),
        )

    def test_bulk_velocity_rtn_nominal_returns_per_component_nominals(self):
        np.testing.assert_array_equal(
            self.moments.bulk_velocity_rtn_nominal(), self.expected_nominal
        )

    def test_bulk_velocity_rtn_covariance_is_three_by_three(self):
        covariance = self.moments.bulk_velocity_rtn_covariance()
        self.assertEqual(covariance.shape, (3, 3))

    def test_bulk_velocity_rtn_covariance_matches_input_covariance(self):
        # `correlated_values` round-trips the covariance through the UFloat
        # triple, so the accessor returns the same matrix that was passed
        # to `make_correlated_velocity`.
        np.testing.assert_allclose(
            self.moments.bulk_velocity_rtn_covariance(),
            self.expected_covariance,
            atol=1e-12,
        )


# ----- fit_solar_wind_alpha_moments — geometry fallback --------------------


class TestFitAlphaMomentsGeometryFallback(unittest.TestCase):
    """When `rotation_matrices` is `None`, `fit_solar_wind_alpha_moments`
    must resolve SWAPI→RTN geometry from `measurement_time` via SPICE
    (`get_swapi_geometry`). The Stage-1 caller normally precomputes
    geometry and shares it across stages — the `None` branch is the
    standalone-callable fallback (used by tests and ad-hoc scripts)."""

    def test_fetches_geometry_from_measurement_time_when_rotation_matrices_omitted(
        self,
    ):
        # Mock `get_swapi_geometry` so the test does not require SPICE
        # kernels. Patch by string to match the lazy-import inside the
        # function body. The downstream forward-model and initial-guess
        # paths are also mocked so the test isolates only the geometry
        # branch.
        proton_moments = _build_proton_fit_result()
        rotation_matrices_from_geometry = _identity_rotation_matrices()
        measurement_time = np.arange(_N_MEAS, dtype=float)

        with patch(
            "imap_l3_processing.swapi.l3a.utils.get_swapi_geometry",
            return_value=rotation_matrices_from_geometry,
        ) as mock_get_geometry, patch.object(
            alpha_module,
            "_alpha_initial_guess",
            return_value=None,
        ):
            result = fit_solar_wind_alpha_moments(
                count_rate=np.zeros(_N_MEAS),
                esa_voltage=_FIVE_SWEEP_VOLTAGE,
                measurement_time=measurement_time,
                swapi_response=_load_swapi_response_with_warm_cache(),
                proton_moments=proton_moments,
                magnetic_field_direction=_B_HAT_RTN,
                alpha_effective_area_scale=1.0,
                proton_effective_area_scale=1.0,
                rotation_matrices=None,
            )

        # `get_swapi_geometry` was called with the measurement time array.
        mock_get_geometry.assert_called_once()
        np.testing.assert_array_equal(
            mock_get_geometry.call_args.args[0], measurement_time
        )
        # The mocked `_alpha_initial_guess` returned `None`, so the fitter
        # short-circuited to a FIT_FAILED moments result — proving it
        # reached the initial-guess step (i.e. did not crash on geometry).
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.FIT_FAILED))


# ----- fit_solar_wind_alpha_moments — peak-bin filtering -------------------


class TestFitAlphaMomentsPeakBinFiltering(
    _SyntheticAlphaSpectrumFixture, unittest.TestCase
):
    """The Stage-2 fit subsets per-measurement arrays to the alpha-peak
    indices repeated across sweeps. Bins with `count_rate <= 0` (e.g. a
    drop-out flagged at L2) inside that subset must be filtered out so the
    LM residual axis only includes physically meaningful bins. Otherwise,
    deadtime correction can divide by zero on a 0-count bin."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Zero out a single peak-region bin in every sweep — `_alpha_initial_guess`
        # will still find a peak (the rest of the bump is intact), but the
        # zeroed bins must be dropped from the LM residual axis.
        # Use `_alpha_initial_guess` to identify which bins are in the peak,
        # then zero one of them.
        alpha_ctx_full = build_solar_wind_fit_context(
            count_rate=cls.observed_count_rate,
            esa_voltage=cls.voltage,
            swapi_response=cls.response,
            central_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        guess = _alpha_initial_guess(
            proton_true_rate=cls.proton_true_rate,
            proton_temperature=_TRUE_PROTON_TEMPERATURE_K,
            alpha_ctx=alpha_ctx_full,
            proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
            magnetic_field_direction=_B_HAT_RTN,
        )
        _, _, _, peak_bin_idx = guess
        # Zero out the central peak bin in every sweep on a copy of the
        # observed count-rate array.
        zeroed_observed = cls.observed_count_rate.copy()
        target_bin = int(peak_bin_idx[len(peak_bin_idx) // 2])
        for s in range(_N_SWEEPS):
            zeroed_observed[s * _N_BINS_PER_SWEEP + target_bin] = 0.0
        cls.zeroed_observed_count_rate = zeroed_observed
        cls.target_bin = target_bin

        cls.proton_moments = _build_proton_fit_result()
        cls.result = fit_solar_wind_alpha_moments(
            count_rate=zeroed_observed,
            esa_voltage=cls.voltage,
            measurement_time=np.zeros(_N_MEAS),
            swapi_response=cls.response,
            proton_moments=cls.proton_moments,
            magnetic_field_direction=_B_HAT_RTN,
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=cls.rotation_matrices,
        )

    def test_fit_completes_when_some_peak_bins_are_zeroed(self):
        # The pipeline filters zero-count bins out of the LM residual axis
        # rather than crashing on them. With most of the alpha bump still
        # intact, the fit should still flag NONE.
        self.assertEqual(self.result.bad_fit_flag, int(SwapiL3Flags.NONE))

    def test_recovers_alpha_density_within_loose_bound_after_zeroing(self):
        # Zeroing one bin per sweep removes ~5/(5·n_peak_bins) of the
        # signal — recovery should still be within ~10% of truth.
        np.testing.assert_allclose(
            self.result.density.nominal_value,
            _TRUE_ALPHA_DENSITY_CM3,
            rtol=0.1,
        )


# ----- fit_solar_wind_alpha_moments — wrong-basin flip ---------------------


class TestFitAlphaMomentsWrongBasinFlip(unittest.TestCase):
    """The `Δv` parameter has a 1-DOF basin ambiguity along B̂: LM can
    converge to a local minimum at the sign-flipped Δv if the initial
    guess is biased toward the wrong basin. After LM-1, the fitter
    evaluates the residual at `−Δv` and runs LM-2 from the flipped seed
    if the flipped MSE is strictly smaller. The final result is whichever
    converged."""

    def test_lm_runs_a_second_time_when_flipped_residual_is_lower(self):
        # Mock `least_squares` to drive the wrong-basin path:
        # * LM-1 returns a converged result with Δv = +30, but a fake
        #   residual vector designed so the flipped Δv = -30 evaluation
        #   has a smaller squared norm.
        # * LM-2 returns a result at Δv = -30 with a smaller residual.
        # The dummy `_alpha_residuals_njit` is also patched so the flipped
        # MSE (the test's controlled trigger) is below the LM-1 MSE.
        proton_moments = _build_proton_fit_result()

        # LM-1 result: Δv = +30, residual norm² / N = 100.0.
        lm1_x = np.array([np.log(0.2), np.log(4.0e5), 30.0])
        lm1_fun = np.full(_N_MEAS, np.sqrt(100.0))
        lm1_jac = np.zeros((_N_MEAS, 3))
        lm1_jac[0, 0] = 1.0
        lm1_jac[0, 1] = 1.0
        lm1_jac[0, 2] = 1.0
        lm1_result = MagicMock()
        lm1_result.x = lm1_x
        lm1_result.fun = lm1_fun
        lm1_result.jac = lm1_jac
        lm1_result.success = True

        # LM-2 result: Δv = -30, residual norm² / N = 1.0.
        lm2_x = np.array([np.log(0.2), np.log(4.0e5), -30.0])
        lm2_fun = np.full(_N_MEAS, 1.0)
        lm2_jac = np.zeros((_N_MEAS, 3))
        lm2_jac[0, 0] = 1.0
        lm2_jac[0, 1] = 1.0
        lm2_jac[0, 2] = 1.0
        lm2_result = MagicMock()
        lm2_result.x = lm2_x
        lm2_result.fun = lm2_fun
        lm2_result.jac = lm2_jac
        lm2_result.success = True

        # Patch the dummy residual function so the flipped-residual MSE
        # check (called between LM-1 and LM-2) returns a smaller MSE than
        # LM-1's own residual norm.
        # The function under test computes `mse_flipped = mean(residuals(x_flipped)**2)`.
        # We control this via a residual function that returns 1.0 for any
        # call (so squared-mean is 1.0 < 100.0).
        with patch.object(
            alpha_module,
            "_alpha_initial_guess",
            return_value=(0.2, 4.0e5, 0.0, np.array([10, 11, 12])),
        ), patch.object(
            alpha_module,
            "_alpha_residuals_njit",
            return_value=np.full(_N_MEAS, 1.0),
        ), patch.object(
            alpha_module.scipy.optimize,
            "least_squares",
            side_effect=[lm1_result, lm2_result],
        ) as mock_lm:
            result = fit_solar_wind_alpha_moments(
                count_rate=np.full(_N_MEAS, 100.0),
                esa_voltage=_FIVE_SWEEP_VOLTAGE,
                measurement_time=np.zeros(_N_MEAS),
                swapi_response=_load_swapi_response_with_warm_cache(),
                proton_moments=proton_moments,
                magnetic_field_direction=_B_HAT_RTN,
                alpha_effective_area_scale=1.0,
                proton_effective_area_scale=1.0,
                rotation_matrices=_identity_rotation_matrices(),
            )

        # LM ran twice — once for LM-1, once from the flipped seed.
        self.assertEqual(mock_lm.call_count, 2)
        # The second call's `x0` is the LM-1 result with Δv sign flipped.
        lm2_x0 = mock_lm.call_args_list[1].args[1]
        self.assertEqual(lm2_x0[2], -30.0)
        # The final Δv comes from LM-2 (which was given the flipped seed).
        self.assertAlmostEqual(result.delta_v.nominal_value, -30.0, places=10)


# ----- fit_solar_wind_alpha_moments — non-converged LM ---------------------


class TestFitAlphaMomentsLMFailureFlag(unittest.TestCase):
    """When the Stage-2 LM run does not converge, `result.success=False`
    must add `FIT_FAILED` to the moments quality flag — the moments
    themselves still carry whatever LM produced (so downstream consumers
    can inspect them), but the flag indicates the fit was not trusted."""

    def test_fit_failed_flag_set_when_least_squares_does_not_converge(self):
        proton_moments = _build_proton_fit_result()

        # Build an LM result that doesn't converge (`success=False`) but is
        # otherwise structurally valid so the rest of the function runs.
        x_fit = np.array([np.log(0.2), np.log(4.0e5), 30.0])
        residual_norm = np.full(_N_MEAS, 1.0)
        jac = np.zeros((_N_MEAS, 3))
        jac[0, 0] = 1.0
        jac[0, 1] = 1.0
        jac[0, 2] = 1.0
        non_converged = MagicMock()
        non_converged.x = x_fit
        non_converged.fun = residual_norm
        non_converged.jac = jac
        non_converged.success = False

        # Force the wrong-basin check to *not* trigger (mse_flipped >= mse)
        # so only one LM call happens and the result is the non-converged one.
        # Returning identical residuals everywhere makes mse == mse_flipped,
        # so the `<` check fails.
        with patch.object(
            alpha_module,
            "_alpha_initial_guess",
            return_value=(0.2, 4.0e5, 0.0, np.array([10, 11, 12])),
        ), patch.object(
            alpha_module,
            "_alpha_residuals_njit",
            return_value=np.full(_N_MEAS, 1.0),
        ), patch.object(
            alpha_module.scipy.optimize,
            "least_squares",
            return_value=non_converged,
        ):
            result = fit_solar_wind_alpha_moments(
                count_rate=np.full(_N_MEAS, 100.0),
                esa_voltage=_FIVE_SWEEP_VOLTAGE,
                measurement_time=np.zeros(_N_MEAS),
                swapi_response=_load_swapi_response_with_warm_cache(),
                proton_moments=proton_moments,
                magnetic_field_direction=_B_HAT_RTN,
                alpha_effective_area_scale=1.0,
                proton_effective_area_scale=1.0,
                rotation_matrices=_identity_rotation_matrices(),
            )

        # `FIT_FAILED` set; non-converged LM moments still populated (not NaN).
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.FIT_FAILED))
        self.assertFalse(np.isnan(result.density.nominal_value))


# ----- _alpha_initial_guess — additional failure branches ------------------


class TestAlphaInitialGuessAdditionalFailureBranches(unittest.TestCase):
    """Failure paths in `_alpha_initial_guess` reached by mocking the
    helpers it delegates to. Each must return `None` so the fitter can
    downgrade to `FIT_FAILED` rather than crashing."""

    def _build_alpha_ctx_for_synthetic_spectrum(self):
        response = _load_swapi_response_with_warm_cache()
        rotations = _identity_rotation_matrices()
        observed, proton_true, _ = _synthesize_proton_plus_alpha_count_rate(
            response=response,
            voltage=_FIVE_SWEEP_VOLTAGE,
            rotation_matrices=rotations,
        )
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=observed,
            esa_voltage=_FIVE_SWEEP_VOLTAGE,
            swapi_response=response,
            central_effective_area_scale=1.0,
            rotation_matrices=rotations,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        return alpha_ctx, proton_true

    def test_returns_none_when_alpha_peak_window_is_too_narrow(self):
        # If `get_alpha_peak_indices` returns a slice of length < 3, the
        # peak window is too narrow to fit a Gaussian seed and the helper
        # returns `None`. Patch `get_alpha_peak_indices` to a 2-bin slice.
        alpha_ctx, proton_true = self._build_alpha_ctx_for_synthetic_spectrum()
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.alpha."
            "calculate_alpha_solar_wind_moments.get_alpha_peak_indices",
            return_value=slice(20, 22),
        ):
            guess = _alpha_initial_guess(
                proton_true_rate=proton_true,
                proton_temperature=_TRUE_PROTON_TEMPERATURE_K,
                alpha_ctx=alpha_ctx,
                proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
                magnetic_field_direction=_B_HAT_RTN,
            )
        self.assertIsNone(guess)

    def test_returns_none_when_residual_at_peak_indices_is_all_nonpositive(self):
        # If `get_alpha_peak_indices` returns indices in a region where the
        # `max(0, mean_obs − 2·proton_obs)` residual is identically zero,
        # the alpha-density seed step (which divides by the mean residual)
        # would be 0 → LM seed `n_alpha=0`. The helper guards by returning
        # `None` whenever the residual at peak indices is not strictly
        # positive anywhere.
        alpha_ctx, proton_true = self._build_alpha_ctx_for_synthetic_spectrum()
        # Choose indices in the proton-dominated region where the residual
        # is clamped to zero by `np.maximum(0, ...)`.
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.alpha."
            "calculate_alpha_solar_wind_moments.get_alpha_peak_indices",
            return_value=slice(_N_BINS_PER_SWEEP - 5, _N_BINS_PER_SWEEP - 1),
        ):
            guess = _alpha_initial_guess(
                proton_true_rate=proton_true,
                proton_temperature=_TRUE_PROTON_TEMPERATURE_K,
                alpha_ctx=alpha_ctx,
                proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
                magnetic_field_direction=_B_HAT_RTN,
            )
        self.assertIsNone(guess)

    def test_returns_none_when_unit_density_alpha_rates_are_zero(self):
        # The density seed is `mean(residual_peak) / mean(unit_alpha_peak)`.
        # If the unit-density alpha forward model evaluates to zero at the
        # peak bins (e.g. the peak window falls outside any bin where
        # alpha rate is nonzero), `denom == 0` and the helper returns
        # `None` rather than dividing by zero.
        alpha_ctx, proton_true = self._build_alpha_ctx_for_synthetic_spectrum()
        n_meas = len(alpha_ctx.esa_voltage)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.alpha."
            "calculate_alpha_solar_wind_moments.model_solar_wind_ideal_coincidence_rates",
            return_value=(np.zeros(n_meas), np.zeros((n_meas, 5))),
        ):
            guess = _alpha_initial_guess(
                proton_true_rate=proton_true,
                proton_temperature=_TRUE_PROTON_TEMPERATURE_K,
                alpha_ctx=alpha_ctx,
                proton_bulk_velocity_rtn=_TRUE_PROTON_VELOCITY_RTN,
                magnetic_field_direction=_B_HAT_RTN,
            )
        self.assertIsNone(guess)


# Doc contracts intentionally not pinned here: upstream-set quality flags
# (`PRELIMINARY_MAG`, `MAG_GAP`, `EPHEMERIS_GAP`) belong in chunk-fitter /
# processor tests, and the LM-jacobian-derived sigmas (`σ_n_α`, `σ_T_α`,
# `σ_Δv`) plus the `Σ_v_α = Σ_v_p + σ_Δv²·B̂B̂ᵀ` covariance update would
# require an independent numerical reference to pin without coupling to
# scipy version drift.
#
# Defensive branches in `_infer_sweep_layout` (line 297, `return None, None`)
# and the matching `if n_sweeps is None: return None` guard in
# `_alpha_initial_guess` (line 244) are unreachable in practice. The loop
# always passes when `n_sweeps=1` because `n_meas % 1 == 0` and
# `np.allclose(arr.reshape(1, n_meas), arr[:n_meas])` is trivially True
# for any array (including the empty array). These lines remain as
# safety rails but cannot be hit by any real input.


if __name__ == "__main__":
    unittest.main()
