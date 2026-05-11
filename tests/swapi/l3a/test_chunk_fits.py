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
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


_N_SWEEPS = 5
_N_BINS = 72
_COARSE_V = np.logspace(np.log10(3500.0), np.log10(140.0), 62)
_FINE_V = np.logspace(np.log10(120.0), np.log10(50.0), 9)
_SCIENCE_V = np.concatenate([_COARSE_V, _FINE_V])
_FULL_ENERGY = np.concatenate([[1.0e4], _SCIENCE_V]) * SWAPI_L2_K_FACTOR

# RTN→SWAPI orientation from proton fit-model tests; transposed = SWAPI→RTN.
# +Y column is the spin axis (≈ -R̂_RTN).
_ANCHOR_ROT = np.array([
    [+0.0705, +0.9157, +0.3955],
    [-0.9968, +0.0792, -0.0057],
    [-0.0365, -0.3939, +0.9184],
]).T
_SPIN_OMEGA = -2.0 * np.pi / 15.13

_TRUE_DENSITY = 5.0
_TRUE_TEMPERATURE_K = 1.0e5
_TRUE_BULK_SPEED = 450.0
# Anti-parallel to spin axis ⇒ wind enters the aperture.
_TRUE_VELOCITY_RTN = -_TRUE_BULK_SPEED * _ANCHOR_ROT[:, 1]
_B_HAT_RTN = _TRUE_VELOCITY_RTN / np.linalg.norm(_TRUE_VELOCITY_RTN)
_SC_VELOCITY_RTN = np.array([0.0, 30.0, 0.0])
_EPOCH_TT2000 = 800_000_000_000_000_000
_CHUNK_EPOCH = _EPOCH_TT2000 + THIRTY_SECONDS_IN_NANOSECONDS

# Every non-flag, non-epoch field must NaN-fill on short-circuit branches.
_PROTON_SCALAR_KEYS = [
    "proton_sw_speed", "proton_sw_speed_uncert",
    "proton_sw_speed_sun", "proton_sw_speed_sun_uncert",
    "proton_sw_temperature", "proton_sw_temperature_uncert",
    "proton_sw_density", "proton_sw_density_uncert",
    "proton_sw_clock_angle", "proton_sw_clock_angle_uncert",
    "proton_sw_deflection_angle", "proton_sw_deflection_angle_uncert",
]
_PROTON_ARRAY_KEYS = [
    "proton_sw_bulk_velocity_rtn_sun", "proton_sw_bulk_velocity_rtn_sun_covariance",
    "proton_sw_bulk_velocity_rtn_sc", "proton_sw_bulk_velocity_rtn_sc_covariance",
]
_ALPHA_SCALAR_KEYS = [
    "alpha_sw_density", "alpha_sw_density_uncert",
    "alpha_sw_temperature", "alpha_sw_temperature_uncert",
    "alpha_sw_delta_v", "alpha_sw_delta_v_uncert",
    "alpha_sw_reference_proton_density", "alpha_sw_reference_proton_temperature",
]
_ALPHA_ARRAY_KEYS = [
    "alpha_sw_velocity_rtn", "alpha_sw_velocity_covariance_rtn",
    "alpha_sw_b_hat_rtn", "alpha_sw_reference_proton_velocity_rtn",
]


def _swapi_response_with_warm_cache(voltages):
    resp = SwapiResponse.from_files(
        get_test_instrument_team_data_path("swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"),
        get_test_instrument_team_data_path("swapi/imap_swapi_central-effective-area_20260425_v001.csv"),
        get_test_instrument_team_data_path("swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"),
    )
    resp.warm_cache(voltages)
    return resp


def _identity_rotations(n):
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def _per_bin_rotations(n_bins_per_sweep):
    """SWAPI→RTN matrices spun from `_ANCHOR_ROT` about its +Y axis at the
    SWAPI spin rate. Identity rotations would collapse the LM into a
    spin-axis-mirror basin; spinning resolves the angular spread."""
    sweep = np.repeat(np.arange(_N_SWEEPS), n_bins_per_sweep)
    bin_in_sweep = np.tile(np.arange(1, n_bins_per_sweep + 1), _N_SWEEPS)
    t = sweep * 12.0 + bin_in_sweep * (12.0 / 72)
    axis = _ANCHOR_ROT[:, 1] / np.linalg.norm(_ANCHOR_ROT[:, 1])
    dphi = _SPIN_OMEGA * (t - 6.0)
    ax, ay, az = axis
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    sin_dp = np.sin(dphi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(dphi))[:, None, None]
    return (np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)) @ _ANCHOR_ROT


def _efficiency_table_mock(*, proton=0.05, alpha=0.05, lab=0.05):
    table = MagicMock()
    table.eps_p_lab = lab
    table.get_proton_efficiency_for.return_value = proton
    table.get_alpha_efficiency_for.return_value = alpha
    return table


def _populate_shared(response, table):
    chunk_fits._shared.update(swapi_response=response, efficiency_table=table)


def _clear_shared():
    chunk_fits._shared.clear()


def _identity_rotate_rtn_to_dps(v_rtn, _epoch):
    return v_rtn


def _proton_fit_result():
    return ProtonSolarWindFitResult(
        density=ufloat(_TRUE_DENSITY, 0.05),
        temperature=ufloat(_TRUE_TEMPERATURE_K, 1.0e3),
        bulk_velocity_rtn=tuple(ufloat(v, 1.0) for v in _TRUE_VELOCITY_RTN),
        bad_fit_flag=int(SwapiL3Flags.NONE),
    )


def _synthesize_chunk(*, response, science_only):
    """Forward-model a 5-sweep proton-only chunk at the truth params.

    `science_only=True` → 71-bin science axis (proton-sw, pui).
    `science_only=False` → 62-bin coarse axis (alpha stage 1).
    """
    bin_slice = SWAPI_SCIENCE_BINS if science_only else SWAPI_COARSE_SWEEP_BINS
    n = bin_slice.stop - bin_slice.start
    voltage_one_sweep = _SCIENCE_V if science_only else _COARSE_V
    voltages = np.tile(voltage_one_sweep, _N_SWEEPS)
    rotations = _per_bin_rotations(n)

    ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltages)),
        esa_voltage=voltages,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=rotations,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    truth = SolarWindParams(
        density=_TRUE_DENSITY,
        bulk_velocity_rtn=_TRUE_VELOCITY_RTN.copy(),
        temperature=_TRUE_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    ideal, _ = model_solar_wind_ideal_coincidence_rates(truth, ctx)
    flat_rates = ideal * deadtime_factor(ideal)
    full_rates = np.zeros((_N_SWEEPS, _N_BINS))
    full_rates[:, bin_slice] = flat_rates.reshape(_N_SWEEPS, n)
    chunk = SwapiL2Data(
        sci_start_time=_EPOCH_TT2000 + np.arange(_N_SWEEPS, dtype=np.int64) * 12_000_000_000,
        energy=np.tile(_FULL_ENERGY, (_N_SWEEPS, 1)),
        coincidence_count_rate=full_rates,
        coincidence_count_rate_uncertainty=np.full_like(full_rates, 0.1),
    )
    return chunk, rotations


def _with_nan_at(chunk, sweep, bin_):
    bad = chunk.coincidence_count_rate.copy()
    bad[sweep, bin_] = np.nan
    return SwapiL2Data(
        sci_start_time=chunk.sci_start_time,
        energy=chunk.energy,
        coincidence_count_rate=bad,
        coincidence_count_rate_uncertainty=chunk.coincidence_count_rate_uncertainty,
    )


def _assert_all_nan(tc, result, scalar_keys, array_keys):
    for key in scalar_keys:
        tc.assertTrue(np.isnan(result[key]), msg=f"{key} not NaN")
    for key in array_keys:
        tc.assertTrue(np.all(np.isnan(result[key])), msg=f"{key} not all-NaN")


def _zero_chunk():
    return SwapiL2Data(
        sci_start_time=np.array([_EPOCH_TT2000], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


# ----- chunk_l2_data --------------------------------------------------------


class TestChunkL2DataTrailingDiscard(unittest.TestCase):
    """`chunk_l2_data` yields whole `chunk_size` chunks only — trailing
    `n % chunk_size` sweeps are dropped."""

    def test_partial_trailing_chunk_is_dropped(self):
        rates = np.arange(7 * 4, dtype=float).reshape(7, 4)
        data = SwapiL2Data(
            sci_start_time=np.arange(7, dtype=np.int64),
            energy=rates.copy(),
            coincidence_count_rate=rates.copy(),
            coincidence_count_rate_uncertainty=rates.copy(),
        )
        chunks = list(chunk_l2_data(data, 5))
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0].sci_start_time, np.arange(5))

    def test_n_less_than_chunk_size_yields_no_chunks(self):
        data = SwapiL2Data(
            sci_start_time=np.array([0, 1, 2], dtype=np.int64),
            energy=np.zeros((3, 4)),
            coincidence_count_rate=np.zeros((3, 4)),
            coincidence_count_rate_uncertainty=np.zeros((3, 4)),
        )
        self.assertEqual(list(chunk_l2_data(data, 5)), [])


# ----- _eff_scale and abstract base ----------------------------------------


class TestModuleHelpers(unittest.TestCase):
    def test_proton_eff_scale_is_proton_efficiency_over_lab(self):
        table = _efficiency_table_mock(proton=0.04, lab=0.05)
        self.assertAlmostEqual(_eff_scale(table, _EPOCH_TT2000, "proton"), 0.04 / 0.05)

    def test_alpha_eff_scale_is_alpha_efficiency_over_lab(self):
        table = _efficiency_table_mock(alpha=0.06, lab=0.05)
        self.assertAlmostEqual(_eff_scale(table, _EPOCH_TT2000, "alpha"), 0.06 / 0.05)

    def test_chunk_fitter_is_abstract(self):
        with self.assertRaises(TypeError):
            ChunkFitter()  # type: ignore[abstract]


# ----- ProtonChunkFitter ----------------------------------------------------


class TestProtonChunkFitterPrecomputeGeometry(unittest.TestCase):
    def test_success_returns_epoch_rotations_and_sc_velocity(self):
        rm = _identity_rotations(_N_BINS - 1)
        sc = _SC_VELOCITY_RTN.copy()
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry", return_value=rm
        ), patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn", return_value=sc
        ):
            epoch, rotations, sc_vel = ProtonChunkFitter().precompute_geometry(_zero_chunk())

        self.assertEqual(epoch, _CHUNK_EPOCH)
        np.testing.assert_array_equal(rotations, rm)
        np.testing.assert_array_equal(sc_vel, sc)

    def test_spice_failure_yields_none_for_both_outputs(self):
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=RuntimeError("SPICE gap"),
        ), patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_spacecraft_velocity_rtn"
        ) as sc_mock:
            _, rotations, sc_vel = ProtonChunkFitter().precompute_geometry(_zero_chunk())

        self.assertIsNone(rotations)
        self.assertIsNone(sc_vel)
        sc_mock.assert_not_called()


class TestProtonChunkFitterHappyPath(unittest.TestCase):
    """End-to-end fit on a forward-modelled 5-sweep chunk. Truth recovery
    tolerances per `docs/swapi/solar-wind-moments.md` § Fitting Algorithm
    Validation. The pandas pivot in `_build_passband_array` and the JIT
    compile of `calculate_integral` are amortized across these tests."""

    @classmethod
    def setUpClass(cls):
        cls.response = _swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS))
        _populate_shared(cls.response, _efficiency_table_mock())
        cls.chunk, cls.rotations = _synthesize_chunk(response=cls.response, science_only=True)
        with patch(
            "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
            side_effect=_identity_rotate_rtn_to_dps,
        ):
            cls.result = ProtonChunkFitter().fit_chunk(
                cls.chunk, _CHUNK_EPOCH, cls.rotations, _SC_VELOCITY_RTN.copy()
            )

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def test_quality_flag_none_and_epoch_passthrough(self):
        self.assertEqual(self.result["quality_flags"], SwapiL3Flags.NONE)
        self.assertEqual(self.result["epoch"], _CHUNK_EPOCH)

    def test_recovers_truth_moments(self):
        self.assertAlmostEqual(
            self.result["proton_sw_density"], _TRUE_DENSITY, delta=0.05 * _TRUE_DENSITY
        )
        self.assertAlmostEqual(
            self.result["proton_sw_temperature"],
            _TRUE_TEMPERATURE_K,
            delta=0.05 * _TRUE_TEMPERATURE_K,
        )
        np.testing.assert_allclose(
            self.result["proton_sw_bulk_velocity_rtn_sc"], _TRUE_VELOCITY_RTN, atol=5.0
        )

    def test_uncertainties_are_strictly_positive(self):
        # Bit-exact zero would mean the LM Jacobian degenerated.
        for key in _PROTON_SCALAR_KEYS:
            if key.endswith("_uncert"):
                with self.subTest(key=key):
                    self.assertGreater(self.result[key], 0.0)

    def test_speed_is_norm_of_sc_frame_velocity(self):
        # With identity RTN→DPS, derive_velocity_angles defines speed = ‖v_dps‖.
        np.testing.assert_allclose(
            self.result["proton_sw_speed"],
            np.linalg.norm(self.result["proton_sw_bulk_velocity_rtn_sc"]),
            rtol=1e-9,
        )

    def test_sun_frame_velocity_is_sc_frame_plus_sc_velocity(self):
        np.testing.assert_allclose(
            self.result["proton_sw_bulk_velocity_rtn_sun"],
            self.result["proton_sw_bulk_velocity_rtn_sc"] + _SC_VELOCITY_RTN,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            self.result["proton_sw_speed_sun"],
            np.linalg.norm(self.result["proton_sw_bulk_velocity_rtn_sun"]),
            rtol=1e-9,
        )

    def test_clock_and_deflection_match_velocity_components(self):
        v = self.result["proton_sw_bulk_velocity_rtn_sc"]
        expected_clock = np.degrees(np.arctan2(-v[1], -v[0])) % 360
        expected_defl = np.degrees(np.arccos(-v[2] / np.linalg.norm(v)))
        self.assertAlmostEqual(self.result["proton_sw_clock_angle"], expected_clock, places=6)
        self.assertAlmostEqual(self.result["proton_sw_deflection_angle"], expected_defl, places=6)

    def test_velocity_covariance_is_symmetric_psd_and_sc_equals_sun(self):
        cov_sc = self.result["proton_sw_bulk_velocity_rtn_sc_covariance"]
        cov_sun = self.result["proton_sw_bulk_velocity_rtn_sun_covariance"]
        self.assertEqual(cov_sc.shape, (3, 3))
        np.testing.assert_allclose(cov_sc, cov_sc.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov_sc)[0], 0.0)
        # Source returns the same matrix for both keys.
        np.testing.assert_array_equal(cov_sc, cov_sun)


class TestProtonChunkFitterFillValueBranches(unittest.TestCase):
    """`EPHEMERIS_GAP` per `docs/swapi/solar-wind-moments.md` § Quality Flags
    when SPICE cannot provide rotations or SC velocity. NaN in count rate
    short-circuits without setting a flag."""

    @classmethod
    def setUpClass(cls):
        cls.response = _swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS))
        _populate_shared(cls.response, _efficiency_table_mock())
        cls.chunk, cls.rotations = _synthesize_chunk(response=cls.response, science_only=True)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def test_missing_rotation_matrices_sets_ephemeris_gap(self):
        result = ProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, None, _SC_VELOCITY_RTN)
        self.assertEqual(result["quality_flags"], SwapiL3Flags.EPHEMERIS_GAP)
        _assert_all_nan(self, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)

    def test_missing_sc_velocity_sets_ephemeris_gap(self):
        result = ProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, self.rotations, None)
        self.assertEqual(result["quality_flags"], SwapiL3Flags.EPHEMERIS_GAP)
        _assert_all_nan(self, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)

    def test_nan_in_count_rate_short_circuits_without_setting_flag(self):
        result = ProtonChunkFitter().fit_chunk(
            _with_nan_at(self.chunk, 0, 5), _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        _assert_all_nan(self, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)


# ----- AlphaChunkFitter -----------------------------------------------------


class TestAlphaChunkFitterPrecomputeGeometry(unittest.TestCase):
    def _mag_centered_on(self, epoch_ns):
        offsets = np.array([-1_000_000_000, 0, 1_000_000_000], dtype=np.int64)
        return MagData(epoch=epoch_ns + offsets, mag_data=np.tile(_B_HAT_RTN, (3, 1)))

    def test_success_returns_rotations_and_b_hat(self):
        rm = _identity_rotations(_N_SWEEPS * 62)
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry", return_value=rm
        ):
            epoch, rotations, b_hat = AlphaChunkFitter(
                self._mag_centered_on(_CHUNK_EPOCH)
            ).precompute_geometry(_zero_chunk())

        self.assertEqual(epoch, _CHUNK_EPOCH)
        np.testing.assert_array_equal(rotations, rm)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_spice_failure_yields_none_rotations_but_keeps_b_hat(self):
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=RuntimeError("SPICE gap"),
        ):
            _, rotations, b_hat = AlphaChunkFitter(
                self._mag_centered_on(_CHUNK_EPOCH)
            ).precompute_geometry(_zero_chunk())
        self.assertIsNone(rotations)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_empty_mag_window_yields_nan_b_hat(self):
        far_future = _EPOCH_TT2000 + 10**18
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            return_value=_identity_rotations(1),
        ):
            _, _, b_hat = AlphaChunkFitter(
                self._mag_centered_on(far_future)
            ).precompute_geometry(_zero_chunk())
        self.assertTrue(np.all(np.isnan(b_hat)))


class TestAlphaChunkFitterFlagBranches(unittest.TestCase):
    """`bad_fit_flag` per `docs/swapi/solar-wind-moments.md` § Quality Flags:
    rotation_matrices is None ⇒ EPHEMERIS_GAP; b_hat is None / non-finite ⇒
    MAG_GAP; NaN in count rates ⇒ FIT_FAILED (initial value preserved)."""

    @classmethod
    def setUpClass(cls):
        cls.response = _swapi_response_with_warm_cache(np.tile(_COARSE_V, _N_SWEEPS))
        _populate_shared(cls.response, _efficiency_table_mock())
        cls.chunk, cls.rotations = _synthesize_chunk(response=cls.response, science_only=False)
        cls.fitter = AlphaChunkFitter(mag_data=None)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def _assert_flag_and_all_nan(self, result, flag):
        self.assertEqual(int(result["bad_fit_flag"]), int(flag))
        _assert_all_nan(self, result, _ALPHA_SCALAR_KEYS, _ALPHA_ARRAY_KEYS)

    def test_missing_rotations_sets_ephemeris_gap(self):
        result = self.fitter.fit_chunk(self.chunk, _CHUNK_EPOCH, None, _B_HAT_RTN)
        self._assert_flag_and_all_nan(result, SwapiL3Flags.EPHEMERIS_GAP)

    def test_nan_b_hat_sets_mag_gap(self):
        result = self.fitter.fit_chunk(
            self.chunk, _CHUNK_EPOCH, self.rotations, np.full(3, np.nan)
        )
        self._assert_flag_and_all_nan(result, SwapiL3Flags.MAG_GAP)

    def test_none_b_hat_sets_mag_gap(self):
        result = self.fitter.fit_chunk(self.chunk, _CHUNK_EPOCH, self.rotations, None)
        self._assert_flag_and_all_nan(result, SwapiL3Flags.MAG_GAP)

    def test_nan_count_rate_keeps_initial_fit_failed_flag(self):
        result = self.fitter.fit_chunk(
            _with_nan_at(self.chunk, 0, 5), _CHUNK_EPOCH, self.rotations, _B_HAT_RTN
        )
        self._assert_flag_and_all_nan(result, SwapiL3Flags.FIT_FAILED)


class TestAlphaChunkFitterStageOrdering(unittest.TestCase):
    """Stage 1 (proton fit on coarse bins) feeds Stage 2 (alpha fit with
    frozen proton moments). Patches verify the contract; alpha numerics
    are covered under `tests/.../solar_wind/alpha/`."""

    @classmethod
    def setUpClass(cls):
        cls.response = _swapi_response_with_warm_cache(np.tile(_COARSE_V, _N_SWEEPS))
        _populate_shared(cls.response, _efficiency_table_mock())
        cls.chunk, cls.rotations = _synthesize_chunk(response=cls.response, science_only=False)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def test_stage_2_receives_stage_1_result_and_outputs_propagate(self):
        stage_1 = _proton_fit_result()
        stage_2 = AlphaSolarWindMoments(
            density=ufloat(0.2, 0.01),
            temperature=ufloat(4.0e5, 1.0e3),
            bulk_velocity_rtn=(ufloat(-480.0, 1.5), ufloat(0.0, 1.5), ufloat(0.0, 1.5)),
            delta_v=ufloat(30.0, 1.0),
            bad_fit_flag=int(SwapiL3Flags.NONE),
        )

        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits._fit_proton", return_value=stage_1
        ) as mock_stage_1, patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.fit_solar_wind_alpha_moments",
            return_value=stage_2,
        ) as mock_stage_2:
            result = AlphaChunkFitter(mag_data=None).fit_chunk(
                self.chunk, _CHUNK_EPOCH, self.rotations, _B_HAT_RTN
            )

        mock_stage_1.assert_called_once()
        # `proton_moments` is the 5th positional arg of fit_solar_wind_alpha_moments.
        self.assertIs(mock_stage_2.call_args.args[4], stage_1)

        self.assertAlmostEqual(result["alpha_sw_density"], 0.2)
        self.assertAlmostEqual(result["alpha_sw_density_uncert"], 0.01)
        self.assertAlmostEqual(result["alpha_sw_temperature"], 4.0e5)
        self.assertAlmostEqual(result["alpha_sw_delta_v"], 30.0)
        self.assertAlmostEqual(result["alpha_sw_delta_v_uncert"], 1.0)
        np.testing.assert_allclose(result["alpha_sw_velocity_rtn"], [-480.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["alpha_sw_b_hat_rtn"], _B_HAT_RTN)
        # Reference proton fields propagate from Stage 1.
        self.assertAlmostEqual(result["alpha_sw_reference_proton_density"], _TRUE_DENSITY)
        self.assertAlmostEqual(
            result["alpha_sw_reference_proton_temperature"], _TRUE_TEMPERATURE_K
        )
        np.testing.assert_array_equal(
            result["alpha_sw_reference_proton_velocity_rtn"], _TRUE_VELOCITY_RTN
        )
        self.assertEqual(int(result["bad_fit_flag"]), int(SwapiL3Flags.NONE))


# ----- PuiProtonChunkFitter -------------------------------------------------


class TestPuiProtonChunkFitterPrecomputeGeometry(unittest.TestCase):
    def test_success_returns_epoch_and_rotations(self):
        rm = _identity_rotations(_N_BINS - 1)
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry", return_value=rm
        ):
            epoch, rotations = PuiProtonChunkFitter().precompute_geometry(_zero_chunk())
        self.assertEqual(epoch, _CHUNK_EPOCH)
        np.testing.assert_array_equal(rotations, rm)

    def test_spice_failure_returns_none_rotations(self):
        with patch(
            "imap_l3_processing.swapi.l3a.chunk_fits.get_swapi_geometry",
            side_effect=RuntimeError("SPICE gap"),
        ):
            _, rotations = PuiProtonChunkFitter().precompute_geometry(_zero_chunk())
        self.assertIsNone(rotations)


class TestPuiProtonChunkFitterFitChunk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response = _swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS))
        _populate_shared(cls.response, _efficiency_table_mock())
        cls.chunk, cls.rotations = _synthesize_chunk(response=cls.response, science_only=True)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()

    def _assert_all_ufloats_nan(self, result):
        for key in ["proton_sw_speed", "proton_sw_clock_angle", "proton_sw_deflection_angle"]:
            with self.subTest(key=key):
                self.assertTrue(np.isnan(result[key].nominal_value))

    def test_happy_path_recovers_truth_with_correct_angles_and_finite_uncertainty(self):
        with patch(
            "imap_l3_processing.swapi.l3a.utils.rotate_rtn_to_dps",
            side_effect=_identity_rotate_rtn_to_dps,
        ):
            result = PuiProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, self.rotations)

        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self.assertAlmostEqual(
            result["proton_sw_speed"].nominal_value, _TRUE_BULK_SPEED, delta=5.0
        )
        # With identity DPS rotation, closed forms from derive_velocity_angles.
        expected_clock = (
            np.degrees(np.arctan2(-_TRUE_VELOCITY_RTN[1], -_TRUE_VELOCITY_RTN[0])) % 360
        )
        expected_defl = np.degrees(np.arccos(-_TRUE_VELOCITY_RTN[2] / _TRUE_BULK_SPEED))
        self.assertAlmostEqual(
            result["proton_sw_clock_angle"].nominal_value, expected_clock, delta=1.0
        )
        self.assertAlmostEqual(
            result["proton_sw_deflection_angle"].nominal_value, expected_defl, delta=1.0
        )
        self.assertGreater(result["proton_sw_speed"].std_dev, 0.0)

    def test_missing_rotations_sets_ephemeris_gap_and_nan_ufloats(self):
        result = PuiProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, None)
        self.assertEqual(int(result["quality_flags"]), int(SwapiL3Flags.EPHEMERIS_GAP))
        self._assert_all_ufloats_nan(result)

    def test_nan_count_rate_short_circuits_with_nan_ufloats_and_none_flag(self):
        result = PuiProtonChunkFitter().fit_chunk(
            _with_nan_at(self.chunk, 0, 5), _CHUNK_EPOCH, self.rotations
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self._assert_all_ufloats_nan(result)


# ----- ParallelChunkRunner --------------------------------------------------


class TestParallelChunkRunnerWorkerHelpers(unittest.TestCase):
    def tearDown(self):
        _clear_shared()

    def test_init_worker_populates_shared_state(self):
        response, table, fitter = MagicMock(), MagicMock(), MagicMock(spec=ChunkFitter)
        _init_worker(response, table, fitter)
        self.assertIs(chunk_fits._shared["swapi_response"], response)
        self.assertIs(chunk_fits._shared["efficiency_table"], table)
        self.assertIs(chunk_fits._shared["fitter"], fitter)

    def test_run_one_unpacks_geom_into_fit_chunk(self):
        fitter = MagicMock(spec=ChunkFitter)
        fitter.fit_chunk.return_value = {"epoch": 12345}
        _init_worker(MagicMock(), MagicMock(), fitter)

        chunk = MagicMock()
        result = _run_one(chunk, ("epoch_arg", "rm_arg", "extra"))

        fitter.fit_chunk.assert_called_once_with(chunk, "epoch_arg", "rm_arg", "extra")
        self.assertEqual(result, {"epoch": 12345})


class TestParallelChunkRunnerOrchestration(unittest.TestCase):
    """`run` precomputes geometry parent-side, dispatches across a fork pool,
    and stacks per-key dict results into per-key arrays. The fork pool is
    replaced with an inline pool — pool internals (fork semantics, FD
    inheritance) are exercised only via integration tests."""

    def tearDown(self):
        _clear_shared()

    def test_run_dispatches_per_chunk_and_stacks_outputs_in_chunk_order(self):
        fitter = MagicMock(spec=ChunkFitter)
        fitter.precompute_geometry.side_effect = [("g0",), ("g1",)]
        fitter.fit_chunk.side_effect = [
            {"a": 1.0, "b": np.array([2.0, 3.0])},
            {"a": 4.0, "b": np.array([5.0, 6.0])},
        ]
        chunks = [MagicMock(name="chunk0"), MagicMock(name="chunk1")]

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

        runner = ParallelChunkRunner(swapi_response=MagicMock(), efficiency_table=MagicMock())
        with patch("imap_l3_processing.swapi.l3a.chunk_fits.multiprocessing") as mock_mp:
            mock_mp.get_context.return_value.Pool = _InlinePool
            stacked = runner.run(chunks, fitter)

        self.assertEqual(fitter.precompute_geometry.call_count, 2)
        self.assertEqual(fitter.fit_chunk.call_count, 2)
        np.testing.assert_array_equal(stacked["a"], np.array([1.0, 4.0]))
        np.testing.assert_array_equal(stacked["b"], np.array([[2.0, 3.0], [5.0, 6.0]]))


if __name__ == "__main__":
    unittest.main()
