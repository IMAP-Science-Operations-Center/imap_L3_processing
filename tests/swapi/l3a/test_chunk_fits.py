"""Unit tests for `imap_l3_processing.swapi.l3a.chunk_fits`.

Behavior spec lives in `docs/swapi/solar-wind-moments.md`. The module owns
three `ChunkFitter` subclasses (`ProtonChunkFitter`, `AlphaChunkFitter`,
`PuiProtonChunkFitter`) plus a fork-pool dispatcher (`ParallelChunkRunner`).

Each fitter splits work into a parent-side `precompute_geometry` (SPICE/MAG
queries) and a worker-side `fit_chunk` consuming the precomputed tuple plus
shared resources from `chunk_fits._shared`. These tests:

* exercise both methods directly (no fork pool) by populating `_shared` in
  the same process;
* mock the SPICE-backed helpers (`get_swapi_geometry`,
  `get_spacecraft_velocity_rtn`, `rotate_rtn_to_dps`) so the unit tests do
  not depend on furnished SPICE kernels;
* drive `fit_chunk` with forward-modelled count rates synthesized from a
  known `SolarWindParams` so the happy path actually exercises the fitter.

`SwapiResponse` is loaded from the shipped instrument-team CSVs once per
test class — `_build_passband_array` (the pandas pivot inside
`SWAPIResponse.create_passband_grid`) is the dominant per-call cost and is
amortized across all subclass tests via `setUpClass`.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from uncertainties import ufloat

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.models import MagData
from imap_l3_processing.swapi.l3a import chunk_fits
from imap_l3_processing.swapi.l3a.chunk_fits import (
    AlphaChunkFitter,
    ChunkFitter,
    ParallelChunkRunner,
    ProtonChunkFitter,
    PuiProtonChunkFitter,
    _eff_scale,
    _init_worker,
    _run_one,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_alpha_solar_wind_moments import (
    AlphaSolarWindMoments,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams
from imap_l3_processing.swapi.l3a.utils import chunk_l2_data
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


# ----- module-level fixtures ------------------------------------------------

# Shipped calibration CSVs. Loading the full SwapiResponse triggers the
# same code path the production pipeline uses.
_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)

# 5 sweeps × 72 bins/sweep — matches `chunk_l2_data(..., chunk_size=5)`
# in the production processor and the SWAPI 12 s / 72-bin sweep layout.
_N_SWEEPS = 5
_N_BINS_PER_SWEEP = 72

# Coarse-only voltage axis (62 bins/sweep). SWAPI L2 voltages are descending;
# `_alpha_initial_guess` requires strictly decreasing energies, but the exact
# endpoints are not load-bearing for these tests.
_COARSE_ONE_SWEEP_VOLTAGE = np.logspace(np.log10(3500.0), np.log10(140.0), 62)
# Full 71-bin science axis: bin 0 (a hardware artifact) is dropped before
# this point. We pad the leading bin with a placeholder voltage that the
# fitter never reaches because `SWAPI_SCIENCE_BINS = slice(1, 72)`.
_FINE_VOLTAGES = np.logspace(np.log10(120.0), np.log10(50.0), 9)
_SCIENCE_ONE_SWEEP_VOLTAGE = np.concatenate(
    [_COARSE_ONE_SWEEP_VOLTAGE, _FINE_VOLTAGES]
)
# Full 72-bin axis stored on `SwapiL2Data.energy`, with a placeholder
# leading value (bin 0). The pipeline divides by `SWAPI_L2_K_FACTOR`
# before passing voltages to the fitter, so we multiply here so the
# fitter sees the same physical volts as the synthesized truth.
_FULL_ONE_SWEEP_ENERGY = np.concatenate(
    [[1.0e4], _SCIENCE_ONE_SWEEP_VOLTAGE]
) * SWAPI_L2_K_FACTOR

# Anchor SWAPI→RTN rotation matrix near 2026-01-01, matching the synthesis
# pattern in `tests/.../proton/test_fit_model.py`. The literal block is in
# RTN→SWAPI orientation; transposing makes it SWAPI→RTN. Per-bin matrices
# are produced by spinning this anchor about its own +Y column (the spin
# axis in RTN) at the SWAPI spin period.
_ANCHOR_ROTATION_MATRIX = np.array(
    [
        [+0.0705, +0.9157, +0.3955],
        [-0.9968, +0.0792, -0.0057],
        [-0.0365, -0.3939, +0.9184],
    ]
).T
_SWEEP_DURATION_S = 12.0
_SAMPLE_TIME_PER_BIN_S = _SWEEP_DURATION_S / 72
_SPIN_PERIOD_S = 15.13
_ANCHOR_TIME_S = 0.5 * _SWEEP_DURATION_S
_SPIN_OMEGA_RAD_S = -2.0 * np.pi / _SPIN_PERIOD_S

# Truth solar-wind parameters. The bulk velocity is anti-parallel to the
# synthetic spin axis (the +Y column of `_ANCHOR_ROTATION_MATRIX`, which
# lies near -R̂_RTN in this fixture). This puts the wind *into* the SWAPI
# aperture; using arbitrary RTN-frame components would point the wind at
# the back of the instrument and zero the synthetic count rates.
_TRUE_DENSITY_CM3 = 5.0
_TRUE_TEMPERATURE_K = 1.0e5
_TRUE_BULK_SPEED_KM_S = 450.0
_TRUE_BULK_VELOCITY_RTN_KM_S = (
    -_TRUE_BULK_SPEED_KM_S * _ANCHOR_ROTATION_MATRIX[:, 1]
)

# Synthetic spacecraft velocity in RTN (km/s) — small relative to the wind
# so `v_sun = v_sc-frame + v_sc` differs from `v_sc-frame` enough for the
# offset to be observable, but doesn't change the qualitative geometry.
_SC_VELOCITY_RTN_KM_S = np.array([0.0, 30.0, 0.0])

# Stage-1 proton fit uncertainties used to seed `ProtonSolarWindFitResult`.
# Small but nonzero so `bulk_velocity_rtn_covariance` is positive-definite.
_PROTON_SIGMA_DENSITY = 0.05
_PROTON_SIGMA_TEMPERATURE = 1.0e3
_PROTON_SIGMA_VELOCITY = 1.0

# Magnetic field direction along the truth bulk-velocity direction
# (anti-sunward in this fixture). Positive Δv pushes alphas along +B̂ —
# the standard alpha drift sign.
_B_HAT_RTN = _TRUE_BULK_VELOCITY_RTN_KM_S / np.linalg.norm(
    _TRUE_BULK_VELOCITY_RTN_KM_S
)

# Fixture epoch (TT2000 ns). `derive_velocity_angles` is patched, so the
# value only matters insofar as it propagates through `chunk_epoch` — pick
# a round number near the start of the IMAP era.
_EPOCH_TT2000_NS = 800_000_000_000_000_000

# ----- helpers --------------------------------------------------------------


def _load_swapi_response_with_warm_cache(voltages: np.ndarray) -> SwapiResponse:
    """Load `SwapiResponse` from the shipped CSVs and warm its passband
    cache for the fixture voltages.

    `create_response_grid` raises if the cache isn't warm for the requested
    voltage, so callers building a fit context off this response must warm
    it first.
    """
    response = SwapiResponse.from_files(
        _AZIMUTHAL_TRANSMISSION_PATH,
        _CENTRAL_EFFECTIVE_AREA_PATH,
        _PASSBAND_FIT_COEFFICIENTS_PATH,
    )
    response.warm_cache(voltages)
    return response


def _identity_rotations(n: int) -> np.ndarray:
    """`n` identity SWAPI→RTN rotation matrices (writeable contiguous
    array; the JIT'd forward model rejects read-only views).

    Used for tests where the rotation values themselves are not load-bearing
    (e.g. `precompute_geometry` mocking, abstract-class checks).
    """
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def _per_bin_rotation_matrices(n_bins_per_sweep: int) -> np.ndarray:
    """Synthesize plausible per-bin SWAPI→RTN matrices for `_N_SWEEPS`
    sweeps and `n_bins_per_sweep` bins per sweep — matches the synthesis
    pattern used by `tests/.../proton/test_fit_model.py`. Identity-rotation
    matrices give a degenerate geometry where the LM sees no inter-bin
    angular spread and converges into spin-axis-mirror basins; spinning
    the anchor over the sweep removes that degeneracy.
    """
    sweep_index = np.repeat(np.arange(_N_SWEEPS), n_bins_per_sweep)
    bin_index_in_sweep = np.tile(
        np.arange(1, n_bins_per_sweep + 1), _N_SWEEPS
    )
    sample_times_s = (
        sweep_index * _SWEEP_DURATION_S
        + bin_index_in_sweep * _SAMPLE_TIME_PER_BIN_S
    )

    spin_axis = _ANCHOR_ROTATION_MATRIX[:, 1] / np.linalg.norm(
        _ANCHOR_ROTATION_MATRIX[:, 1]
    )
    delta_phi = _SPIN_OMEGA_RAD_S * (sample_times_s - _ANCHOR_TIME_S)

    ax, ay, az = spin_axis
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    sin_dp = np.sin(delta_phi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(delta_phi))[:, None, None]
    rot = np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)
    return rot @ _ANCHOR_ROTATION_MATRIX


def _make_efficiency_table_mock(
    *,
    proton_efficiency: float = 0.05,
    alpha_efficiency: float = 0.05,
    eps_p_lab: float = 0.05,
) -> MagicMock:
    """Stand-in for `EfficiencyCalibrationTable`. The chunk fitters only
    consume `eps_p_lab`, `get_proton_efficiency_for(epoch)`, and
    `get_alpha_efficiency_for(epoch)` — nothing else off the table is
    accessed inside `_fit_proton` or `_eff_scale`.

    Defaults to lab-equal efficiencies so `_eff_scale = 1.0` for both
    species, decoupling the synthetic-rate path from any efficiency
    rescaling.
    """
    table = MagicMock()
    table.eps_p_lab = eps_p_lab
    table.get_proton_efficiency_for.return_value = proton_efficiency
    table.get_alpha_efficiency_for.return_value = alpha_efficiency
    return table


def _populate_shared(swapi_response, efficiency_table) -> None:
    """Populate the module-level `_shared` dict used by worker functions
    in lieu of running `_init_worker` inside a fork pool."""
    chunk_fits._shared["swapi_response"] = swapi_response
    chunk_fits._shared["efficiency_table"] = efficiency_table


def _clear_shared() -> None:
    chunk_fits._shared.clear()


def _build_proton_fit_result(
    *,
    density: float = _TRUE_DENSITY_CM3,
    temperature: float = _TRUE_TEMPERATURE_K,
    velocity_rtn: np.ndarray = _TRUE_BULK_VELOCITY_RTN_KM_S,
    bad_fit_flag: int = int(SwapiL3Flags.NONE),
) -> ProtonSolarWindFitResult:
    return ProtonSolarWindFitResult(
        density=ufloat(density, _PROTON_SIGMA_DENSITY),
        temperature=ufloat(temperature, _PROTON_SIGMA_TEMPERATURE),
        bulk_velocity_rtn=(
            ufloat(velocity_rtn[0], _PROTON_SIGMA_VELOCITY),
            ufloat(velocity_rtn[1], _PROTON_SIGMA_VELOCITY),
            ufloat(velocity_rtn[2], _PROTON_SIGMA_VELOCITY),
        ),
        bad_fit_flag=int(bad_fit_flag),
    )


def _identity_rotate_rtn_to_dps(vector_rtn, _epoch_tt2000_ns):
    """Stand-in for the SPICE-driven RTN→DPS rotation: returns the input
    unchanged. Lets `derive_velocity_angles` run without furnished SPICE."""
    return vector_rtn


def _synthesize_proton_only_count_rate(
    *,
    response: SwapiResponse,
    voltage: np.ndarray,
    rotation_matrices: np.ndarray,
    truth_params: SolarWindParams,
) -> np.ndarray:
    """Forward-model deadtime-applied proton-only count rates on the
    fixture voltage axis at the truth params — feeds the happy-path fit
    tests below. No Poisson noise; we exercise orchestration, not the
    noise budget."""
    ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltage)),
        esa_voltage=voltage,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    ideal, _ = model_solar_wind_ideal_coincidence_rates(truth_params, ctx)
    return ideal * deadtime_factor(ideal)


def _build_synthetic_proton_chunk(
    *,
    response: SwapiResponse,
    science_only: bool,
) -> tuple[SwapiL2Data, np.ndarray]:
    """Build a 5-sweep `SwapiL2Data` chunk with proton-only count rates
    forward-modelled at `_TRUE_BULK_VELOCITY_RTN_KM_S`.

    `science_only=True` synthesizes for the 71-bin (`SWAPI_SCIENCE_BINS`)
    axis used by proton-sw and pui-he. `science_only=False` synthesizes
    for the 62-bin (`SWAPI_COARSE_SWEEP_BINS`) axis used by alpha-sw.
    Either way the returned `SwapiL2Data.energy` carries the full 72-bin
    L2 axis (the leading bin 0 is a placeholder, never read by the
    fitters).

    Returns `(chunk, rotation_matrices)`. `rotation_matrices` is the
    flattened per-bin SWAPI→RTN array sized to match the slice the
    fitter will use — the production pipeline builds this in
    `precompute_geometry`.
    """
    bin_slice = SWAPI_SCIENCE_BINS if science_only else SWAPI_COARSE_SWEEP_BINS
    n_bins_per_sweep_used = bin_slice.stop - bin_slice.start
    one_sweep_v = (
        _SCIENCE_ONE_SWEEP_VOLTAGE if science_only else _COARSE_ONE_SWEEP_VOLTAGE
    )
    # Tile per-sweep voltages to the full 5-sweep flat axis the fitter sees.
    flat_voltages = np.tile(one_sweep_v, _N_SWEEPS)

    rotation_matrices = _per_bin_rotation_matrices(n_bins_per_sweep_used)

    truth_params = SolarWindParams(
        density=_TRUE_DENSITY_CM3,
        bulk_velocity_rtn=_TRUE_BULK_VELOCITY_RTN_KM_S.copy(),
        temperature=_TRUE_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    flat_rates = _synthesize_proton_only_count_rate(
        response=response,
        voltage=flat_voltages,
        rotation_matrices=rotation_matrices,
        truth_params=truth_params,
    )

    # Embed the synthesized rates back into the full 72-bin L2 axis. Bins
    # outside the science slice get fill values that never reach the fit
    # path because the fitters slice with `SWAPI_SCIENCE_BINS` /
    # `SWAPI_COARSE_SWEEP_BINS` before flattening.
    sweep_rates = flat_rates.reshape(_N_SWEEPS, n_bins_per_sweep_used)
    full_rates = np.zeros((_N_SWEEPS, _N_BINS_PER_SWEEP), dtype=float)
    full_rates[:, bin_slice] = sweep_rates

    sci_start_time = _EPOCH_TT2000_NS + np.arange(_N_SWEEPS, dtype=np.int64) * (
        12 * 1_000_000_000
    )
    energy = np.tile(_FULL_ONE_SWEEP_ENERGY, (_N_SWEEPS, 1))
    chunk = SwapiL2Data(
        sci_start_time=sci_start_time,
        energy=energy,
        coincidence_count_rate=full_rates,
        coincidence_count_rate_uncertainty=np.full_like(full_rates, 0.1),
    )
    return chunk, rotation_matrices


# ----- chunk_l2_data trailing-discard semantics -----------------------------


class TestChunkL2DataTrailingDiscard(unittest.TestCase):
    """`chunk_l2_data` yields whole `chunk_size` chunks only — the trailing
    `n % chunk_size` sweeps are dropped. `tests/swapi/l3a/test_utils.py`
    only exercises the exactly-divisible path; this class verifies the
    discard.
    """

    def test_partial_trailing_chunk_is_dropped(self):
        n_sweeps = 7  # `7 % 5 == 2` trailing sweeps that must be dropped.
        chunk_size = 5
        epoch = np.arange(n_sweeps, dtype=np.int64)
        rates = np.arange(n_sweeps * 4, dtype=float).reshape(n_sweeps, 4)
        data = SwapiL2Data(
            sci_start_time=epoch,
            energy=rates.copy(),
            coincidence_count_rate=rates.copy(),
            coincidence_count_rate_uncertainty=rates.copy(),
        )

        chunks = list(chunk_l2_data(data, chunk_size))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].sci_start_time), chunk_size)
        np.testing.assert_array_equal(
            chunks[0].sci_start_time, np.arange(chunk_size)
        )

    def test_n_less_than_chunk_size_yields_no_chunks(self):
        # Strictly fewer than `chunk_size` measurements ⇒ no whole chunk.
        data = SwapiL2Data(
            sci_start_time=np.array([0, 1, 2], dtype=np.int64),
            energy=np.zeros((3, 4)),
            coincidence_count_rate=np.zeros((3, 4)),
            coincidence_count_rate_uncertainty=np.zeros((3, 4)),
        )
        self.assertEqual(list(chunk_l2_data(data, 5)), [])


# ----- _eff_scale -----------------------------------------------------------


class TestEffScale(unittest.TestCase):
    """`_eff_scale` divides the species efficiency by `eps_p_lab`. Used
    inside `_fit_proton` and `AlphaChunkFitter.fit_chunk` to rescale the
    central effective area to the in-flight efficiency."""

    def test_proton_scale_is_proton_efficiency_over_lab(self):
        table = _make_efficiency_table_mock(
            proton_efficiency=0.04, eps_p_lab=0.05
        )
        scale = _eff_scale(table, _EPOCH_TT2000_NS, "proton")
        self.assertAlmostEqual(scale, 0.04 / 0.05)

    def test_alpha_scale_is_alpha_efficiency_over_lab(self):
        table = _make_efficiency_table_mock(
            alpha_efficiency=0.06, eps_p_lab=0.05
        )
        scale = _eff_scale(table, _EPOCH_TT2000_NS, "alpha")
        self.assertAlmostEqual(scale, 0.06 / 0.05)

    def test_unit_efficiencies_yield_unit_scale(self):
        # Lab-equal efficiencies ⇒ scale=1.0 (the regime exercised by the
        # synthetic-rate fits below, which use `_make_efficiency_table_mock`
        # with default lab-equal values).
        table = _make_efficiency_table_mock()
        self.assertAlmostEqual(_eff_scale(table, 0, "proton"), 1.0)
        self.assertAlmostEqual(_eff_scale(table, 0, "alpha"), 1.0)


# ----- ChunkFitter abstract base -------------------------------------------


class TestChunkFitterIsAbstract(unittest.TestCase):
    """`ChunkFitter` declares `precompute_geometry` and `fit_chunk` abstract;
    instantiating it directly must fail."""

    def test_cannot_instantiate_chunk_fitter_directly(self):
        with self.assertRaises(TypeError):
            ChunkFitter()  # type: ignore[abstract]


# ----- ProtonChunkFitter.precompute_geometry -------------------------------


class TestProtonChunkFitterPrecomputeGeometry(unittest.TestCase):
    """`precompute_geometry` returns `(epoch, rotation_matrices, sc_velocity)`
    on success and `(epoch, None, None)` on any SPICE failure. The doc
    (`§Quality Flags`, line 659) defines `EPHEMERIS_GAP` as the chunk-level
    quality flag set when SPICE cannot provide rotation matrices.
    """

    def setUp(self):
        # 1-sweep chunk is sufficient — `precompute_geometry` does not
        # inspect the per-sweep arrays beyond `chunk.sci_start_time`.
        self.chunk = SwapiL2Data(
            sci_start_time=np.array([_EPOCH_TT2000_NS], dtype=np.int64),
            energy=np.zeros((1, _N_BINS_PER_SWEEP)),
            coincidence_count_rate=np.zeros((1, _N_BINS_PER_SWEEP)),
            coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS_PER_SWEEP)),
        )

    def test_returns_epoch_rotation_matrices_and_sc_velocity_on_success(self):
        rm = _identity_rotations(_N_BINS_PER_SWEEP - 1)  # SWAPI_SCIENCE_BINS
        sc_vel = _SC_VELOCITY_RTN_KM_S.copy()
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            return_value=rm,
        ), patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn",
            return_value=sc_vel,
        ):
            epoch, rotation_matrices, sc_velocity = (
                ProtonChunkFitter().precompute_geometry(self.chunk)
            )

        self.assertEqual(
            epoch, _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS
        )
        np.testing.assert_array_equal(rotation_matrices, rm)
        np.testing.assert_array_equal(sc_velocity, sc_vel)

    def test_swapi_geometry_failure_returns_none_for_rm_and_sc_velocity(self):
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=RuntimeError("SPICE gap"),
        ), patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn",
        ) as mock_sc_velocity:
            epoch, rotation_matrices, sc_velocity = (
                ProtonChunkFitter().precompute_geometry(self.chunk)
            )

        self.assertEqual(
            epoch, _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS
        )
        self.assertIsNone(rotation_matrices)
        self.assertIsNone(sc_velocity)
        # Once `get_swapi_geometry` raises, the spacecraft-velocity SPICE
        # call is short-circuited by the same except clause.
        mock_sc_velocity.assert_not_called()

    def test_sc_velocity_failure_returns_none_for_both(self):
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            return_value=_identity_rotations(_N_BINS_PER_SWEEP - 1),
        ), patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn",
            side_effect=RuntimeError("SPICE gap"),
        ):
            _, rotation_matrices, sc_velocity = (
                ProtonChunkFitter().precompute_geometry(self.chunk)
            )

        # Both sentinels must be `None` regardless of which SPICE call
        # failed — the production path uses a single boolean check on
        # `rotation_matrices` to decide whether to NaN-fill.
        self.assertIsNone(rotation_matrices)
        self.assertIsNone(sc_velocity)


# ----- ProtonChunkFitter.fit_chunk happy path -------------------------------


class _ProtonHappyPathFixture(unittest.TestCase):
    """Class-level fixture: builds the SwapiResponse, populates `_shared`,
    forward-models a 5-sweep chunk at the truth params, and runs
    `ProtonChunkFitter.fit_chunk` once. The pandas pivot inside
    `_build_passband_array` and the JIT compile of `calculate_integral`
    are paid one time across all subclass tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache(
            np.tile(_SCIENCE_ONE_SWEEP_VOLTAGE, _N_SWEEPS)
        )
        cls.efficiency_table = _make_efficiency_table_mock()
        _populate_shared(cls.response, cls.efficiency_table)

        cls.chunk, cls.rotation_matrices = _build_synthetic_proton_chunk(
            response=cls.response, science_only=True
        )
        cls.epoch = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS

        with patch(
            "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
            side_effect=_identity_rotate_rtn_to_dps,
        ):
            cls.result = ProtonChunkFitter().fit_chunk(
                cls.chunk,
                cls.epoch,
                cls.rotation_matrices,
                _SC_VELOCITY_RTN_KM_S.copy(),
            )

    @classmethod
    def tearDownClass(cls):
        _clear_shared()


class TestProtonChunkFitterHappyPath(_ProtonHappyPathFixture):
    """With valid rotation matrices, valid SC velocity, and forward-modelled
    proton-only count rates, `fit_chunk` returns a populated dict that
    recovers the truth moments to within the WIND/SWE-validated accuracy
    in `docs/swapi/solar-wind-moments.md` § Fitting Algorithm Validation.
    """

    def test_quality_flag_is_none_on_clean_synthetic_input(self):
        # No SPICE gaps, no fill values, LM converges ⇒ no flag bits set.
        self.assertEqual(self.result["quality_flags"], SwapiL3Flags.NONE)

    def test_recovers_truth_density(self):
        self.assertAlmostEqual(
            self.result["proton_sw_density"],
            _TRUE_DENSITY_CM3,
            delta=0.05 * _TRUE_DENSITY_CM3,
        )

    def test_recovers_truth_temperature(self):
        self.assertAlmostEqual(
            self.result["proton_sw_temperature"],
            _TRUE_TEMPERATURE_K,
            delta=0.05 * _TRUE_TEMPERATURE_K,
        )

    def test_bulk_velocity_rtn_sc_recovers_truth(self):
        np.testing.assert_allclose(
            self.result["proton_sw_bulk_velocity_rtn_sc"],
            _TRUE_BULK_VELOCITY_RTN_KM_S,
            atol=5.0,
        )

    def test_bulk_velocity_rtn_sun_is_sc_frame_plus_sc_velocity(self):
        # The doc-level contract: `v_sun = v_sc-frame + v_sc`. Use the
        # actual nominal SC-frame bulk velocity from this fit (not the
        # truth) — the assertion is on the offset, not on the fit.
        np.testing.assert_allclose(
            self.result["proton_sw_bulk_velocity_rtn_sun"],
            self.result["proton_sw_bulk_velocity_rtn_sc"] + _SC_VELOCITY_RTN_KM_S,
            atol=1e-9,
        )

    def test_velocity_covariance_is_three_by_three(self):
        self.assertEqual(
            self.result["proton_sw_bulk_velocity_rtn_sc_covariance"].shape, (3, 3)
        )
        self.assertEqual(
            self.result["proton_sw_bulk_velocity_rtn_sun_covariance"].shape, (3, 3)
        )

    def test_speed_uncertainty_is_finite(self):
        self.assertTrue(np.isfinite(self.result["proton_sw_speed"]))
        self.assertTrue(np.isfinite(self.result["proton_sw_speed_uncert"]))

    def test_speed_sun_is_finite(self):
        self.assertTrue(np.isfinite(self.result["proton_sw_speed_sun"]))
        self.assertTrue(np.isfinite(self.result["proton_sw_speed_sun_uncert"]))

    def test_clock_and_deflection_angles_are_finite(self):
        self.assertTrue(np.isfinite(self.result["proton_sw_clock_angle"]))
        self.assertTrue(np.isfinite(self.result["proton_sw_clock_angle_uncert"]))
        self.assertTrue(np.isfinite(self.result["proton_sw_deflection_angle"]))
        self.assertTrue(
            np.isfinite(self.result["proton_sw_deflection_angle_uncert"])
        )

    def test_epoch_is_passed_through_unchanged(self):
        self.assertEqual(self.result["epoch"], self.epoch)


# ----- ProtonChunkFitter.fit_chunk fill-value branches ---------------------


class TestProtonChunkFitterFillValueBranches(unittest.TestCase):
    """Per `docs/swapi/solar-wind-moments.md` §Quality Flags (line 659),
    `EPHEMERIS_GAP` is set when SPICE could not provide rotation matrices
    (or SC velocity), and the chunk is fill-valued without attempting a
    fit. Fill values in the input data also short-circuit to NaN moments
    (no specific flag — moments are NaN-filled, quality_flags is NONE).
    """

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache(
            np.tile(_SCIENCE_ONE_SWEEP_VOLTAGE, _N_SWEEPS)
        )
        cls.efficiency_table = _make_efficiency_table_mock()
        _populate_shared(cls.response, cls.efficiency_table)
        cls.chunk, cls.rotation_matrices = _build_synthetic_proton_chunk(
            response=cls.response, science_only=True
        )
        cls.epoch = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def _assert_moments_are_nan_filled(self, result):
        self.assertTrue(np.isnan(result["proton_sw_density"]))
        self.assertTrue(np.isnan(result["proton_sw_temperature"]))
        self.assertTrue(np.isnan(result["proton_sw_speed"]))
        self.assertTrue(
            np.all(np.isnan(result["proton_sw_bulk_velocity_rtn_sc"]))
        )
        self.assertTrue(
            np.all(np.isnan(result["proton_sw_bulk_velocity_rtn_sun"]))
        )

    def test_missing_rotation_matrices_sets_ephemeris_gap_flag(self):
        result = ProtonChunkFitter().fit_chunk(
            self.chunk, self.epoch, None, _SC_VELOCITY_RTN_KM_S
        )
        self.assertTrue(
            int(result["quality_flags"]) & int(SwapiL3Flags.EPHEMERIS_GAP)
        )
        self._assert_moments_are_nan_filled(result)

    def test_missing_sc_velocity_sets_ephemeris_gap_flag(self):
        # With non-None rotation matrices but None SC velocity, the same
        # `EPHEMERIS_GAP` flag must be set: the doc treats both SPICE
        # outputs as a single ephemeris-availability concept.
        result = ProtonChunkFitter().fit_chunk(
            self.chunk, self.epoch, self.rotation_matrices, None
        )
        self.assertTrue(
            int(result["quality_flags"]) & int(SwapiL3Flags.EPHEMERIS_GAP)
        )
        self._assert_moments_are_nan_filled(result)

    def test_nan_in_count_rate_short_circuits_without_setting_flag(self):
        # NaN fill values in the input bins ⇒ moments NaN-filled but no
        # quality bit is set (this is treated as bad-input, not a SPICE
        # gap or fit failure).
        bad_chunk = SwapiL2Data(
            sci_start_time=self.chunk.sci_start_time,
            energy=self.chunk.energy,
            coincidence_count_rate=np.where(
                np.arange(self.chunk.coincidence_count_rate.size).reshape(
                    self.chunk.coincidence_count_rate.shape
                )
                == 5,
                np.nan,
                self.chunk.coincidence_count_rate,
            ),
            coincidence_count_rate_uncertainty=self.chunk.coincidence_count_rate_uncertainty,
        )
        result = ProtonChunkFitter().fit_chunk(
            bad_chunk, self.epoch, self.rotation_matrices, _SC_VELOCITY_RTN_KM_S
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self._assert_moments_are_nan_filled(result)


# ----- AlphaChunkFitter.precompute_geometry --------------------------------


class TestAlphaChunkFitterPrecomputeGeometry(unittest.TestCase):
    """`AlphaChunkFitter.precompute_geometry` queries SPICE for rotation
    matrices and computes the chunk-mean B̂ from the supplied MAG data.
    On SPICE failure it returns `(epoch, None, b_hat)`; on missing MAG
    coverage `compute_direction_of_mean_magnetic_field_over_chunk`
    returns NaN, which Stage-2 detects via `MAG_GAP` (see
    `docs/swapi/solar-wind-moments.md` line 92, line 660).
    """

    def setUp(self):
        self.chunk = SwapiL2Data(
            sci_start_time=np.array([_EPOCH_TT2000_NS], dtype=np.int64),
            energy=np.zeros((1, _N_BINS_PER_SWEEP)),
            coincidence_count_rate=np.zeros((1, _N_BINS_PER_SWEEP)),
            coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS_PER_SWEEP)),
        )

    def _make_mag_data_covering(self, epoch_ns: int) -> MagData:
        # Three samples within ±30 s of `epoch_ns`, all -R̂. Mean is -R̂.
        offsets = np.array(
            [-1_000_000_000, 0, 1_000_000_000], dtype=np.int64
        )
        return MagData(
            epoch=epoch_ns + offsets,
            mag_data=np.tile(_B_HAT_RTN, (3, 1)),
        )

    def test_returns_rotation_matrices_and_b_hat_on_success(self):
        epoch_center = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS
        mag = self._make_mag_data_covering(epoch_center)
        rm = _identity_rotations(_N_SWEEPS * 62)
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            return_value=rm,
        ):
            epoch, rotation_matrices, b_hat = AlphaChunkFitter(
                mag
            ).precompute_geometry(self.chunk)

        self.assertEqual(epoch, epoch_center)
        np.testing.assert_array_equal(rotation_matrices, rm)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_spice_failure_yields_none_rotation_but_keeps_b_hat(self):
        # Per `precompute_geometry`, a SPICE failure NaN-fills the chunk
        # via `rotation_matrices=None`; the MAG query is independent and
        # still runs.
        epoch_center = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS
        mag = self._make_mag_data_covering(epoch_center)
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=RuntimeError("SPICE gap"),
        ):
            _, rotation_matrices, b_hat = AlphaChunkFitter(
                mag
            ).precompute_geometry(self.chunk)

        self.assertIsNone(rotation_matrices)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_empty_mag_window_produces_nan_b_hat(self):
        # MAG epochs entirely outside the ±30 s chunk window — the helper
        # returns NaN, which Stage-2 turns into `MAG_GAP`.
        far_future_epoch = _EPOCH_TT2000_NS + 10**18
        mag = self._make_mag_data_covering(far_future_epoch)
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            return_value=_identity_rotations(1),
        ):
            _, _, b_hat = AlphaChunkFitter(mag).precompute_geometry(self.chunk)

        self.assertTrue(np.all(np.isnan(b_hat)))


# ----- AlphaChunkFitter.fit_chunk flag branches ----------------------------


class TestAlphaChunkFitterFlagBranches(unittest.TestCase):
    """`AlphaChunkFitter.fit_chunk` has three short-circuit branches in
    addition to the happy path:

    * `rotation_matrices is None`        ⇒ `bad_fit_flag = EPHEMERIS_GAP`
    * `magnetic_field_direction` non-finite ⇒ `bad_fit_flag = MAG_GAP`
    * NaN in the input count rates       ⇒ `bad_fit_flag = FIT_FAILED`
      (initial value, no flag rewrite happens after the fill-values
       exception)

    These match `docs/swapi/solar-wind-moments.md` § Quality Flags
    (lines 657–660). `PRELIMINARY_MAG` is set higher up by the
    SwapiProcessor on every chunk in the run when MAG L1D was used —
    it is not a chunk-fits responsibility, so it is not tested here.
    """

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache(
            np.tile(_COARSE_ONE_SWEEP_VOLTAGE, _N_SWEEPS)
        )
        cls.efficiency_table = _make_efficiency_table_mock()
        _populate_shared(cls.response, cls.efficiency_table)
        cls.chunk, cls.rotation_matrices = _build_synthetic_proton_chunk(
            response=cls.response, science_only=False
        )
        cls.epoch = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS
        cls.fitter = AlphaChunkFitter(mag_data=None)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def _assert_moments_are_nan_filled(self, result):
        self.assertTrue(np.isnan(result["alpha_sw_density"]))
        self.assertTrue(np.isnan(result["alpha_sw_temperature"]))
        self.assertTrue(np.isnan(result["alpha_sw_delta_v"]))
        self.assertTrue(np.all(np.isnan(result["alpha_sw_velocity_rtn"])))
        self.assertTrue(
            np.all(np.isnan(result["alpha_sw_reference_proton_velocity_rtn"]))
        )
        self.assertTrue(np.all(np.isnan(result["alpha_sw_b_hat_rtn"])))

    def test_missing_rotation_matrices_sets_ephemeris_gap_flag(self):
        result = self.fitter.fit_chunk(
            self.chunk, self.epoch, None, _B_HAT_RTN
        )
        self.assertEqual(
            int(result["bad_fit_flag"]), int(SwapiL3Flags.EPHEMERIS_GAP)
        )
        self._assert_moments_are_nan_filled(result)

    def test_nan_b_hat_sets_mag_gap_flag(self):
        result = self.fitter.fit_chunk(
            self.chunk,
            self.epoch,
            self.rotation_matrices,
            np.full(3, np.nan),
        )
        self.assertEqual(int(result["bad_fit_flag"]), int(SwapiL3Flags.MAG_GAP))
        self._assert_moments_are_nan_filled(result)

    def test_none_b_hat_sets_mag_gap_flag(self):
        # The same gate also catches `magnetic_field_direction is None`;
        # `precompute_geometry` does not return None for B̂ today, but the
        # `fit_chunk` guard handles it defensively.
        result = self.fitter.fit_chunk(
            self.chunk, self.epoch, self.rotation_matrices, None
        )
        self.assertEqual(int(result["bad_fit_flag"]), int(SwapiL3Flags.MAG_GAP))
        self._assert_moments_are_nan_filled(result)

    def test_nan_count_rate_keeps_initial_fit_failed_flag(self):
        # The fill-values branch raises before Stage-1 runs, so the
        # initial `bad_fit_flag = FIT_FAILED` value is preserved.
        bad_chunk = SwapiL2Data(
            sci_start_time=self.chunk.sci_start_time,
            energy=self.chunk.energy,
            coincidence_count_rate=np.where(
                np.arange(self.chunk.coincidence_count_rate.size).reshape(
                    self.chunk.coincidence_count_rate.shape
                )
                == 5,
                np.nan,
                self.chunk.coincidence_count_rate,
            ),
            coincidence_count_rate_uncertainty=self.chunk.coincidence_count_rate_uncertainty,
        )
        result = self.fitter.fit_chunk(
            bad_chunk, self.epoch, self.rotation_matrices, _B_HAT_RTN
        )
        self.assertEqual(int(result["bad_fit_flag"]), int(SwapiL3Flags.FIT_FAILED))
        self._assert_moments_are_nan_filled(result)


# ----- AlphaChunkFitter.fit_chunk Stage-1 → Stage-2 ordering ---------------


class TestAlphaChunkFitterStageOrdering(unittest.TestCase):
    """Stage 1 fits proton moments on coarse-only bins; Stage 2 calls
    `fit_solar_wind_alpha_moments` with the Stage-1 result frozen.
    Verify the call order with patches. Nominal alpha output is taken from
    the patched Stage 2 — we do not exercise the alpha numerics here (that
    is covered in `tests/swapi/l3a/science/solar_wind/alpha/`).
    """

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache(
            np.tile(_COARSE_ONE_SWEEP_VOLTAGE, _N_SWEEPS)
        )
        cls.efficiency_table = _make_efficiency_table_mock()
        _populate_shared(cls.response, cls.efficiency_table)
        cls.chunk, cls.rotation_matrices = _build_synthetic_proton_chunk(
            response=cls.response, science_only=False
        )
        cls.epoch = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def test_stage_2_is_called_with_stage_1_proton_result(self):
        # Stub Stage-1 so we know exactly what Stage-2 must receive.
        stage_1_result = _build_proton_fit_result()
        nan = ufloat(np.nan, np.nan)
        stage_2_result = AlphaSolarWindMoments(
            density=ufloat(0.2, 0.01),
            temperature=ufloat(4.0e5, 1.0e3),
            bulk_velocity_rtn=(
                ufloat(-480.0, 1.0),
                ufloat(0.0, 1.0),
                ufloat(0.0, 1.0),
            ),
            delta_v=ufloat(30.0, 1.0),
            bad_fit_flag=int(SwapiL3Flags.NONE),
        )

        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits._fit_proton",
            return_value=stage_1_result,
        ) as mock_stage_1, patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.fit_solar_wind_alpha_moments",
            return_value=stage_2_result,
        ) as mock_stage_2:
            result = AlphaChunkFitter(mag_data=None).fit_chunk(
                self.chunk,
                self.epoch,
                self.rotation_matrices,
                _B_HAT_RTN,
            )

        # Stage 1 must run before Stage 2 — `mock_stage_1` was called
        # first by virtue of `_fit_proton` being invoked inside the
        # try-block before `fit_solar_wind_alpha_moments`.
        mock_stage_1.assert_called_once()
        mock_stage_2.assert_called_once()
        # Stage 2's `proton_moments` argument must be the Stage-1 result.
        # `fit_solar_wind_alpha_moments` takes positional args; we look
        # up by position 4 (after count_rate, voltages, times, response).
        passed_proton = mock_stage_2.call_args.args[4]
        self.assertIs(passed_proton, stage_1_result)

        # Nominal alpha output gets unpacked into the dict.
        self.assertAlmostEqual(result["alpha_sw_density"], 0.2)
        self.assertAlmostEqual(result["alpha_sw_delta_v"], 30.0)
        # Reference proton fields propagate from Stage 1.
        self.assertAlmostEqual(
            result["alpha_sw_reference_proton_density"], _TRUE_DENSITY_CM3
        )
        self.assertAlmostEqual(
            result["alpha_sw_reference_proton_temperature"], _TRUE_TEMPERATURE_K
        )
        np.testing.assert_array_equal(
            result["alpha_sw_reference_proton_velocity_rtn"],
            _TRUE_BULK_VELOCITY_RTN_KM_S,
        )
        # `bad_fit_flag` from Stage 2 is propagated unchanged.
        self.assertEqual(int(result["bad_fit_flag"]), int(SwapiL3Flags.NONE))


# ----- PuiProtonChunkFitter -------------------------------------------------


class TestPuiProtonChunkFitterPrecomputeGeometry(unittest.TestCase):
    """`PuiProtonChunkFitter.precompute_geometry` returns
    `(epoch, rotation_matrices)` on success and `(epoch, None)` on SPICE
    failure. PUI chunks do not carry SC velocity through to `fit_chunk` —
    the pui-he pipeline only needs the speed/clock/deflection angles, not
    the inertial-frame velocity.
    """

    def setUp(self):
        self.chunk = SwapiL2Data(
            sci_start_time=np.array([_EPOCH_TT2000_NS], dtype=np.int64),
            energy=np.zeros((1, _N_BINS_PER_SWEEP)),
            coincidence_count_rate=np.zeros((1, _N_BINS_PER_SWEEP)),
            coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS_PER_SWEEP)),
        )

    def test_returns_epoch_and_rotation_matrices_on_success(self):
        rm = _identity_rotations(_N_BINS_PER_SWEEP - 1)
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            return_value=rm,
        ):
            geom = PuiProtonChunkFitter().precompute_geometry(self.chunk)

        self.assertEqual(len(geom), 2)
        self.assertEqual(
            geom[0], _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS
        )
        np.testing.assert_array_equal(geom[1], rm)

    def test_spice_failure_returns_none_rotation_matrices(self):
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=RuntimeError("SPICE gap"),
        ):
            _, rotation_matrices = (
                PuiProtonChunkFitter().precompute_geometry(self.chunk)
            )
        self.assertIsNone(rotation_matrices)


class TestPuiProtonChunkFitterFitChunk(unittest.TestCase):
    """`PuiProtonChunkFitter.fit_chunk` returns a dict of correlated
    `UFloat` angles plus a quality flag. Mirror the proton-sw fitter's
    behavior: clean input ⇒ NONE flag; missing rotation matrices ⇒
    `EPHEMERIS_GAP`.
    """

    @classmethod
    def setUpClass(cls):
        cls.response = _load_swapi_response_with_warm_cache(
            np.tile(_SCIENCE_ONE_SWEEP_VOLTAGE, _N_SWEEPS)
        )
        cls.efficiency_table = _make_efficiency_table_mock()
        _populate_shared(cls.response, cls.efficiency_table)
        cls.chunk, cls.rotation_matrices = _build_synthetic_proton_chunk(
            response=cls.response, science_only=True
        )
        cls.epoch = _EPOCH_TT2000_NS + THIRTY_SECONDS_IN_NANOSECONDS

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def test_happy_path_returns_finite_angles_and_none_flag(self):
        with patch(
            "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
            side_effect=_identity_rotate_rtn_to_dps,
        ):
            result = PuiProtonChunkFitter().fit_chunk(
                self.chunk, self.epoch, self.rotation_matrices
            )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self.assertTrue(np.isfinite(result["proton_sw_speed"].nominal_value))
        self.assertTrue(
            np.isfinite(result["proton_sw_clock_angle"].nominal_value)
        )
        self.assertTrue(
            np.isfinite(result["proton_sw_deflection_angle"].nominal_value)
        )

    def test_missing_rotation_matrices_sets_ephemeris_gap_and_nan_angles(self):
        result = PuiProtonChunkFitter().fit_chunk(self.chunk, self.epoch, None)
        self.assertTrue(
            int(result["quality_flags"]) & int(SwapiL3Flags.EPHEMERIS_GAP)
        )
        self.assertTrue(np.isnan(result["proton_sw_speed"].nominal_value))
        self.assertTrue(np.isnan(result["proton_sw_clock_angle"].nominal_value))
        self.assertTrue(
            np.isnan(result["proton_sw_deflection_angle"].nominal_value)
        )

    def test_nan_count_rate_short_circuits_with_nan_angles(self):
        bad_chunk = SwapiL2Data(
            sci_start_time=self.chunk.sci_start_time,
            energy=self.chunk.energy,
            coincidence_count_rate=np.where(
                np.arange(self.chunk.coincidence_count_rate.size).reshape(
                    self.chunk.coincidence_count_rate.shape
                )
                == 5,
                np.nan,
                self.chunk.coincidence_count_rate,
            ),
            coincidence_count_rate_uncertainty=self.chunk.coincidence_count_rate_uncertainty,
        )
        result = PuiProtonChunkFitter().fit_chunk(
            bad_chunk, self.epoch, self.rotation_matrices
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self.assertTrue(np.isnan(result["proton_sw_speed"].nominal_value))


# ----- ParallelChunkRunner worker init / dispatch --------------------------


class TestParallelChunkRunnerWorkerHelpers(unittest.TestCase):
    """`_init_worker` and `_run_one` are the worker-side hooks the fork
    pool runs. `_init_worker` populates `_shared`; `_run_one` reads
    `_shared['fitter']` and dispatches `fitter.fit_chunk`. The fork pool
    itself is platform-dependent and exercised only via integration."""

    def tearDown(self):
        _clear_shared()

    def test_init_worker_populates_shared_state(self):
        response = MagicMock(name="response")
        table = MagicMock(name="table")
        fitter = MagicMock(spec=ChunkFitter)

        _init_worker(response, table, fitter)

        self.assertIs(chunk_fits._shared["swapi_response"], response)
        self.assertIs(chunk_fits._shared["efficiency_table"], table)
        self.assertIs(chunk_fits._shared["fitter"], fitter)

    def test_run_one_dispatches_to_shared_fitter_fit_chunk(self):
        # `_run_one(chunk, geom)` should unpack `geom` as positional args
        # to `fitter.fit_chunk(chunk, *geom)`.
        sentinel_chunk = MagicMock(name="chunk")
        sentinel_result = {"epoch": 12345}

        fitter = MagicMock(spec=ChunkFitter)
        fitter.fit_chunk.return_value = sentinel_result

        _init_worker(MagicMock(), MagicMock(), fitter)

        result = _run_one(sentinel_chunk, ("epoch_arg", "rm_arg", "extra"))

        fitter.fit_chunk.assert_called_once_with(
            sentinel_chunk, "epoch_arg", "rm_arg", "extra"
        )
        self.assertIs(result, sentinel_result)


class TestParallelChunkRunnerOrchestration(unittest.TestCase):
    """`ParallelChunkRunner.run` precomputes geometry per chunk in the
    parent, dispatches to a fork pool, and stacks per-key dict results
    into per-key numpy arrays. The fork pool is replaced with an inline
    runner so the orchestration is testable in a single process — the
    pool internals (fork semantics, OS-level FD inheritance) are
    platform-dependent and exercised only via integration tests.
    """

    def test_run_dispatches_fit_chunk_per_geometry_and_stacks_dict_results(self):
        runner = ParallelChunkRunner(
            swapi_response=MagicMock(),
            efficiency_table=MagicMock(),
        )

        fitter = MagicMock(spec=ChunkFitter)
        # Two chunks, two geometries, two fit results that must stack.
        fitter.precompute_geometry.side_effect = [
            ("g0",),
            ("g1",),
        ]
        fitter.fit_chunk.side_effect = [
            {"a": 1.0, "b": np.array([2.0, 3.0])},
            {"a": 4.0, "b": np.array([5.0, 6.0])},
        ]
        chunks = [MagicMock(name="chunk0"), MagicMock(name="chunk1")]

        # Replace the fork-pool block: precomputation in the parent runs
        # for real; the dispatch step is patched to run sequentially in
        # the same process so the parent-side orchestration (stacking
        # per-key into arrays) is the only behavior under test.
        class _InlinePool:
            def __init__(self, *_args, initializer=None, initargs=(), **_kwargs):
                if initializer is not None:
                    initializer(*initargs)

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def starmap(self, func, iterable):
                return [func(*args) for args in iterable]

        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.multiprocessing"
        ) as mock_mp:
            mock_mp.get_context.return_value.Pool = _InlinePool
            stacked = runner.run(chunks, fitter)

        # Geometry was precomputed once per chunk.
        self.assertEqual(fitter.precompute_geometry.call_count, 2)
        # `fit_chunk` was dispatched once per chunk with the right geom.
        self.assertEqual(fitter.fit_chunk.call_count, 2)

        # Per-key stacking: each output key is a numpy array indexed by chunk.
        np.testing.assert_array_equal(stacked["a"], np.array([1.0, 4.0]))
        np.testing.assert_array_equal(
            stacked["b"], np.array([[2.0, 3.0], [5.0, 6.0]])
        )

    def tearDown(self):
        _clear_shared()


if __name__ == "__main__":
    unittest.main()
