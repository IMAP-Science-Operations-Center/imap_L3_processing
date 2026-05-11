import unittest

import numpy as np

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    ONE_SECOND_IN_NANOSECONDS,
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
)
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import (
    EfficiencyCalibrationTable,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams
from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry, rotate_rtn_to_dps
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.spice_test_case import SpiceTestCase
from tests.test_helpers import get_test_instrument_team_data_path


_N_SWEEPS = 5
_N_BINS = 72
_COARSE_V = np.logspace(np.log10(10000.0), np.log10(50.0), 62)
_FINE_V = np.logspace(np.log10(1000.0), np.log10(500.0), 9)
_SCIENCE_V = np.concatenate([_COARSE_V, _FINE_V])
_FULL_ENERGY = np.concatenate([[1.0e4], _SCIENCE_V]) * SWAPI_L2_K_FACTOR

_TRUE_DENSITY = 5.0
_TRUE_TEMPERATURE_K = 1.0e5
_TRUE_BULK_SPEED = 450.0
# Sunward Parker spiral, off-nominal: 55° from -R toward +T (vs. nominal
# 45° from +R toward -T), tilted 10° out of the ecliptic toward +N.
_B_HAT_RTN = np.array([
    -np.cos(np.radians(55.0)) * np.cos(np.radians(10.0)),
    np.sin(np.radians(55.0)) * np.cos(np.radians(10.0)),
    np.sin(np.radians(10.0)),
])
_TRUE_ALPHA_DENSITY = 0.2
_TRUE_ALPHA_TEMPERATURE_K = 4.0e5
_TRUE_DELTA_V_KM_S = 30.0
_SC_VELOCITY_RTN = np.array([0.0, 30.0, 0.0])
_EPOCH_TT2000 = 800_000_000_000_000_000
_CHUNK_EPOCH = _EPOCH_TT2000 + THIRTY_SECONDS_IN_NANOSECONDS
_SCI_START_TIME = _EPOCH_TT2000 + np.arange(_N_SWEEPS, dtype=np.int64) * 12_000_000_000

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


def _spice_rotations(bin_slice):
    """SPICE-derived SWAPI→RTN rotations at the synthetic chunk's measurement
    times over `bin_slice`."""
    bin_indices = np.arange(bin_slice.start, bin_slice.stop)
    measurement_times = (
        _SCI_START_TIME[:, np.newaxis]
        + bin_indices * (12 / 72 * ONE_SECOND_IN_NANOSECONDS)
    ).flatten()
    return get_swapi_geometry(measurement_times)


def _truth_velocity_rtn(rotations):
    """Truth wind vector: `_TRUE_BULK_SPEED` km/s anti-parallel-ish to the SWAPI
    boresight (column 1 of the SWAPI→RTN rotation at the chunk's first bin),
    tilted 5° toward a stable in-plane direction. The deflection lifts the
    wind off the spin axis so clock-angle assertions are non-degenerate."""
    spin_axis_rtn = rotations[0, :, 1]
    perpendicular = np.cross(spin_axis_rtn, [1.0, 0.0, 0.0])
    perpendicular /= np.linalg.norm(perpendicular)
    deflection = np.radians(5.0)
    direction = -spin_axis_rtn * np.cos(deflection) + perpendicular * np.sin(deflection)
    return _TRUE_BULK_SPEED * direction


def _efficiency_table():
    """Synthetic `EfficiencyCalibrationTable` with realistic in-flight proton
    (0.12) and alpha (0.15) efficiencies and a lab-cal proton efficiency of 0.12."""
    table = EfficiencyCalibrationTable.__new__(EfficiencyCalibrationTable)
    table.data = np.array(
        [
            (np.datetime64("2024-01-01", "ns"), 0, 0.12, 0.15),
            (np.datetime64("2025-11-01", "ns"), 0, 0.12, 0.15),
        ],
        dtype=[
            ("time", "M8[ns]"),
            ("MET", "i8"),
            ("proton efficiency", "f8"),
            ("alpha efficiency", "f8"),
        ],
    )
    return table


def _populate_shared(response, table):
    chunk_fits._shared.update(swapi_response=response, efficiency_table=table)


def _clear_shared():
    chunk_fits._shared.clear()


def _synthesize_chunk(*, response, rotations, proton_velocity_rtn, alpha_velocity_rtn, efficiency_table):
    """Forward-model a 5-sweep proton + alpha chunk at the truth params over
    the full 71-bin science axis. Per-species effective-area scales come from
    `efficiency_table` so synthesis and the fitter share the same calibration."""
    n = SWAPI_SCIENCE_BINS.stop - SWAPI_SCIENCE_BINS.start
    voltages = np.tile(_SCIENCE_V, _N_SWEEPS)

    proton_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltages)),
        esa_voltage=voltages,
        swapi_response=response,
        central_effective_area_scale=chunk_fits._eff_scale(efficiency_table, _CHUNK_EPOCH, "proton"),
        rotation_matrices=rotations,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    alpha_ctx = build_solar_wind_fit_context(
        count_rate=np.zeros(len(voltages)),
        esa_voltage=voltages,
        swapi_response=response,
        central_effective_area_scale=chunk_fits._eff_scale(efficiency_table, _CHUNK_EPOCH, "alpha"),
        rotation_matrices=rotations,
        mass_kg=ALPHA_PARTICLE_MASS_KG,
        mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    )
    proton_truth = SolarWindParams(
        density=_TRUE_DENSITY,
        bulk_velocity_rtn=proton_velocity_rtn.copy(),
        temperature=_TRUE_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    alpha_truth = SolarWindParams(
        density=_TRUE_ALPHA_DENSITY,
        bulk_velocity_rtn=alpha_velocity_rtn.copy(),
        temperature=_TRUE_ALPHA_TEMPERATURE_K,
        mass=ALPHA_PARTICLE_MASS_KG,
    )
    proton_ideal, _ = model_solar_wind_ideal_coincidence_rates(proton_truth, proton_ctx)
    alpha_ideal, _ = model_solar_wind_ideal_coincidence_rates(alpha_truth, alpha_ctx)
    ideal = proton_ideal + alpha_ideal
    flat_rates = ideal * deadtime_factor(ideal)
    full_rates = np.zeros((_N_SWEEPS, _N_BINS))
    full_rates[:, SWAPI_SCIENCE_BINS] = flat_rates.reshape(_N_SWEEPS, n)
    chunk = SwapiL2Data(
        sci_start_time=_SCI_START_TIME.copy(),
        energy=np.tile(_FULL_ENERGY, (_N_SWEEPS, 1)),
        coincidence_count_rate=full_rates,
        coincidence_count_rate_uncertainty=np.full_like(full_rates, 0.1),
    )
    return chunk


def _build_truth_chunk(response, efficiency_table):
    """Forward-model a clean proton+alpha chunk from `response` over the science
    bin range, returning the chunk plus the rotations and truth velocities used
    to synthesize it. The alpha truth velocity is constructed here so all three
    fitter test classes share the same proton-to-alpha relationship."""
    science_rotations = _spice_rotations(SWAPI_SCIENCE_BINS)
    proton_velocity_rtn = _truth_velocity_rtn(science_rotations)
    alpha_velocity_rtn = proton_velocity_rtn + _TRUE_DELTA_V_KM_S * _B_HAT_RTN
    chunk = _synthesize_chunk(
        response=response,
        rotations=science_rotations,
        proton_velocity_rtn=proton_velocity_rtn,
        alpha_velocity_rtn=alpha_velocity_rtn,
        efficiency_table=efficiency_table,
    )
    return chunk, science_rotations, proton_velocity_rtn, alpha_velocity_rtn


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


# 10**18 ns ≈ 31.7 years past `_EPOCH_TT2000` — beyond every kernel in
# `spice_kernels/`, so SPICE raises `SpiceSPKINSUFFDATA` for this chunk.
_OUT_OF_COVERAGE_START_TIME = _EPOCH_TT2000 + 10**18


def _out_of_coverage_chunk():
    return SwapiL2Data(
        sci_start_time=np.array([_OUT_OF_COVERAGE_START_TIME], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


# ----- ProtonChunkFitter ----------------------------------------------------


class TestProtonChunkFitterPrecomputeGeometry(SpiceTestCase):
    """Tests for `ProtonChunkFitter.precompute_geometry` with real SPICE kernels."""

    def test_success_returns_epoch_rotations_and_sc_velocity(self):
        """At an in-coverage chunk, precompute_geometry returns the chunk midpoint epoch, a per-bin rotation array of the right shape, and a 3-vector spacecraft velocity."""
        epoch, rotation_matrices, spacecraft_velocity = ProtonChunkFitter().precompute_geometry(_zero_chunk())

        self.assertEqual(epoch, _CHUNK_EPOCH)
        assert rotation_matrices is not None and spacecraft_velocity is not None
        self.assertEqual(rotation_matrices.shape, (_N_BINS - 1, 3, 3))
        self.assertEqual(spacecraft_velocity.shape, (3,))

    def test_spice_failure_yields_none_for_both_outputs(self):
        """If the chunk falls outside SPICE coverage, the proton fitter returns None for both rotations and spacecraft velocity."""
        _, rotation_matrices, spacecraft_velocity = ProtonChunkFitter().precompute_geometry(_out_of_coverage_chunk())

        self.assertIsNone(rotation_matrices)
        self.assertIsNone(spacecraft_velocity)


class TestProtonChunkFitterFitChunk(SpiceTestCase):
    """Tests for `ProtonChunkFitter.fit_chunk` — end-to-end proton fit plus
    fill-value branches on a forward-modelled chunk. Uses real SPICE for
    `rotate_rtn_to_dps` inside `derive_velocity_angles`."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, cls.rotations, cls.true_proton_velocity_rtn, _ = _build_truth_chunk(cls.response, efficiency_table)
        cls.result = ProtonChunkFitter().fit_chunk(
            cls.chunk, _CHUNK_EPOCH, cls.rotations, _SC_VELOCITY_RTN.copy()
        )

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def test_quality_flag_none_and_epoch_passthrough(self):
        """A clean synthetic chunk produces a NONE quality flag and passes the chunk epoch straight through to the result."""
        self.assertEqual(self.result["quality_flags"], SwapiL3Flags.NONE)
        self.assertEqual(self.result["epoch"], _CHUNK_EPOCH)

    def test_recovers_truth_moments(self):
        """Fitting a forward-modeled chunk back recovers the true density, temperature, and bulk velocity within a few percent."""
        self.assertAlmostEqual(
            self.result["proton_sw_density"], _TRUE_DENSITY, delta=0.05 * _TRUE_DENSITY
        )
        self.assertAlmostEqual(
            self.result["proton_sw_temperature"],
            _TRUE_TEMPERATURE_K,
            delta=0.05 * _TRUE_TEMPERATURE_K,
        )
        np.testing.assert_allclose(
            self.result["proton_sw_bulk_velocity_rtn_sc"], self.true_proton_velocity_rtn, atol=5.0
        )

    def test_uncertainties_are_strictly_positive(self):
        """Every reported scalar uncertainty is strictly positive, so the LM Jacobian did not degenerate to zero."""
        # Bit-exact zero would mean the LM Jacobian degenerated.
        for key in _PROTON_SCALAR_KEYS:
            if key.endswith("_uncert"):
                with self.subTest(key=key):
                    self.assertGreater(self.result[key], 0.0)

    def test_speed_is_norm_of_sc_frame_velocity(self):
        """Reported proton speed equals the magnitude of the SC-frame bulk velocity (rotation to DPS is norm-preserving)."""
        np.testing.assert_allclose(
            self.result["proton_sw_speed"],
            np.linalg.norm(self.result["proton_sw_bulk_velocity_rtn_sc"]),
            rtol=1e-9,
        )

    def test_sun_frame_velocity_is_sc_frame_plus_sc_velocity(self):
        """Sun-frame velocity is the SC-frame velocity plus the SC orbital velocity, and Sun-frame speed is its magnitude."""
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
        """Clock and deflection angles match the closed-form arctan2 / arccos of the DPS-frame velocity (RTN velocity rotated through real SPICE)."""
        velocity_dps = rotate_rtn_to_dps(
            self.result["proton_sw_bulk_velocity_rtn_sc"], _CHUNK_EPOCH
        )
        expected_clock = np.degrees(np.arctan2(-velocity_dps[1], -velocity_dps[0])) % 360
        expected_deflection = np.degrees(np.arccos(-velocity_dps[2] / np.linalg.norm(velocity_dps)))
        self.assertAlmostEqual(self.result["proton_sw_clock_angle"], expected_clock, places=6)
        self.assertAlmostEqual(self.result["proton_sw_deflection_angle"], expected_deflection, places=6)

    def test_velocity_covariance_is_symmetric_psd_and_sc_equals_sun(self):
        """The 3x3 velocity covariance is symmetric and positive-semidefinite, and the same matrix is reported under both the spacecraft-frame and Sun-frame keys."""
        covariance_spacecraft_frame = self.result["proton_sw_bulk_velocity_rtn_sc_covariance"]
        covariance_sun_frame = self.result["proton_sw_bulk_velocity_rtn_sun_covariance"]
        self.assertEqual(covariance_spacecraft_frame.shape, (3, 3))
        np.testing.assert_allclose(covariance_spacecraft_frame, covariance_spacecraft_frame.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance_spacecraft_frame)[0], 0.0)
        # Source returns the same matrix for both keys.
        np.testing.assert_array_equal(covariance_spacecraft_frame, covariance_sun_frame)

    def test_missing_rotation_matrices_sets_ephemeris_gap(self):
        """Calling fit_chunk with no rotation matrices flags EPHEMERIS_GAP and NaN-fills every science field."""
        result = ProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, None, _SC_VELOCITY_RTN)
        self.assertEqual(result["quality_flags"], SwapiL3Flags.EPHEMERIS_GAP)
        _assert_all_nan(self, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)

    def test_missing_sc_velocity_sets_ephemeris_gap(self):
        """Calling fit_chunk with no SC velocity flags EPHEMERIS_GAP and NaN-fills every science field."""
        result = ProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, self.rotations, None)
        self.assertEqual(result["quality_flags"], SwapiL3Flags.EPHEMERIS_GAP)
        _assert_all_nan(self, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)

    def test_nan_in_count_rate_short_circuits_without_setting_flag(self):
        """A NaN in the count rate short-circuits the fit to NaN outputs but leaves the quality flag as NONE (an upstream L2 flag, not a fit failure)."""
        result = ProtonChunkFitter().fit_chunk(
            _with_nan_at(self.chunk, 0, 5), _CHUNK_EPOCH, self.rotations, _SC_VELOCITY_RTN
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        _assert_all_nan(self, result, _PROTON_SCALAR_KEYS, _PROTON_ARRAY_KEYS)


# ----- AlphaChunkFitter -----------------------------------------------------


class TestAlphaChunkFitterPrecomputeGeometry(SpiceTestCase):
    """Tests for `AlphaChunkFitter.precompute_geometry` with real SPICE kernels."""

    def _mag_centered_on(self, epoch_ns):
        offsets = np.array([-1_000_000_000, 0, 1_000_000_000], dtype=np.int64)
        return MagData(epoch=epoch_ns + offsets, mag_data=np.tile(_B_HAT_RTN, (3, 1)))

    def test_success_returns_rotations_and_b_hat(self):
        """With both SPICE and MAG available, alpha precompute_geometry returns the chunk epoch, a coarse-bin rotation array of the right shape, and the median B̂ in the chunk window."""
        epoch, rotation_matrices, b_hat = AlphaChunkFitter(
            self._mag_centered_on(_CHUNK_EPOCH)
        ).precompute_geometry(_zero_chunk())

        self.assertEqual(epoch, _CHUNK_EPOCH)
        assert rotation_matrices is not None
        self.assertEqual(
            rotation_matrices.shape,
            (SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start, 3, 3),
        )
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_spice_failure_yields_none_rotations_but_keeps_b_hat(self):
        """When the chunk falls outside SPICE coverage, alpha precompute returns None rotations but B̂ is still computed from MAG since that path is independent."""
        out_of_coverage_chunk_epoch = _OUT_OF_COVERAGE_START_TIME + THIRTY_SECONDS_IN_NANOSECONDS
        _, rotation_matrices, b_hat = AlphaChunkFitter(
            self._mag_centered_on(out_of_coverage_chunk_epoch)
        ).precompute_geometry(_out_of_coverage_chunk())
        self.assertIsNone(rotation_matrices)
        np.testing.assert_allclose(b_hat, _B_HAT_RTN)

    def test_empty_mag_window_yields_nan_b_hat(self):
        """When no MAG samples fall inside the chunk window, B̂ comes back as NaN even though rotations were successfully computed."""
        far_future = _EPOCH_TT2000 + 10**18
        _, _, b_hat = AlphaChunkFitter(
            self._mag_centered_on(far_future)
        ).precompute_geometry(_zero_chunk())
        self.assertTrue(np.all(np.isnan(b_hat)))


class TestAlphaChunkFitterFitChunk(SpiceTestCase):
    """Tests for `AlphaChunkFitter.fit_chunk` — stage ordering plus fill-value branches in the alpha fit."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, _, cls.true_proton_velocity_rtn, cls.true_alpha_velocity_rtn = _build_truth_chunk(cls.response, efficiency_table)
        # AlphaChunkFitter slices `SWAPI_COARSE_SWEEP_BINS` from the chunk, so
        # the rotations passed in must be coarse-bin-aligned.
        cls.rotations = _spice_rotations(SWAPI_COARSE_SWEEP_BINS)
        cls.fitter = AlphaChunkFitter(mag_data=None)
        cls.happy_result = cls.fitter.fit_chunk(
            cls.chunk, _CHUNK_EPOCH, cls.rotations, _B_HAT_RTN
        )

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def _assert_flag_and_all_nan(self, result, flag):
        self.assertEqual(int(result["bad_fit_flag"]), int(flag))
        _assert_all_nan(self, result, _ALPHA_SCALAR_KEYS, _ALPHA_ARRAY_KEYS)

    def test_recovers_alpha_truth_moments(self):
        """Fitting a forward-modeled proton+alpha chunk recovers the true alpha density, temperature, delta-v, and bulk velocity within a few percent."""
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_density"],
            _TRUE_ALPHA_DENSITY,
            delta=0.10 * _TRUE_ALPHA_DENSITY,
        )
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_temperature"],
            _TRUE_ALPHA_TEMPERATURE_K,
            delta=0.10 * _TRUE_ALPHA_TEMPERATURE_K,
        )
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_delta_v"], _TRUE_DELTA_V_KM_S, delta=2.0
        )
        np.testing.assert_allclose(
            self.happy_result["alpha_sw_velocity_rtn"], self.true_alpha_velocity_rtn, atol=5.0
        )

    def test_reference_proton_moments_propagate_from_stage_1(self):
        """Stage-1 proton density, temperature, and velocity are surfaced under the alpha result's reference-proton fields."""
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_reference_proton_density"],
            _TRUE_DENSITY,
            delta=0.05 * _TRUE_DENSITY,
        )
        self.assertAlmostEqual(
            self.happy_result["alpha_sw_reference_proton_temperature"],
            _TRUE_TEMPERATURE_K,
            delta=0.05 * _TRUE_TEMPERATURE_K,
        )
        np.testing.assert_allclose(
            self.happy_result["alpha_sw_reference_proton_velocity_rtn"],
            self.true_proton_velocity_rtn,
            atol=5.0,
        )

    def test_b_hat_passes_through_to_result(self):
        """The B̂ argument is surfaced unchanged under `alpha_sw_b_hat_rtn`."""
        np.testing.assert_array_equal(self.happy_result["alpha_sw_b_hat_rtn"], _B_HAT_RTN)

    def test_quality_flag_none_on_clean_chunk(self):
        """A clean chunk yields a NONE bad_fit_flag (both Stage 1 and Stage 2 converged)."""
        self.assertEqual(
            int(self.happy_result["bad_fit_flag"]), int(SwapiL3Flags.NONE)
        )

    def test_missing_rotations_sets_ephemeris_gap(self):
        """Calling alpha fit_chunk with no rotations flags EPHEMERIS_GAP and NaN-fills every alpha field."""
        result = self.fitter.fit_chunk(self.chunk, _CHUNK_EPOCH, None, _B_HAT_RTN)
        self._assert_flag_and_all_nan(result, SwapiL3Flags.EPHEMERIS_GAP)

    def test_nan_b_hat_sets_mag_gap(self):
        """A NaN-valued B̂ flags MAG_GAP and NaN-fills every alpha field, since the field-aligned drift constraint cannot be evaluated."""
        result = self.fitter.fit_chunk(
            self.chunk, _CHUNK_EPOCH, self.rotations, np.full(3, np.nan)
        )
        self._assert_flag_and_all_nan(result, SwapiL3Flags.MAG_GAP)

    def test_none_b_hat_sets_mag_gap(self):
        """Passing None for B̂ flags MAG_GAP and NaN-fills every alpha field, mirroring the NaN case."""
        result = self.fitter.fit_chunk(self.chunk, _CHUNK_EPOCH, self.rotations, None)
        self._assert_flag_and_all_nan(result, SwapiL3Flags.MAG_GAP)

    def test_nan_count_rate_keeps_initial_fit_failed_flag(self):
        """A NaN in the alpha count rate causes the Stage-1 proton fit to fail, so the chunk gets FIT_FAILED and NaN-filled alpha fields."""
        result = self.fitter.fit_chunk(
            _with_nan_at(self.chunk, 0, 5), _CHUNK_EPOCH, self.rotations, _B_HAT_RTN
        )
        self._assert_flag_and_all_nan(result, SwapiL3Flags.FIT_FAILED)


# ----- PuiProtonChunkFitter -------------------------------------------------


class TestPuiProtonChunkFitterPrecomputeGeometry(SpiceTestCase):
    """Tests for `PuiProtonChunkFitter.precompute_geometry` with real SPICE kernels."""

    def test_success_returns_epoch_and_rotations(self):
        """At an in-coverage chunk, PUI precompute_geometry returns the chunk midpoint epoch and a per-bin rotation array of the right shape (no spacecraft velocity for PUI)."""
        epoch, rotation_matrices = PuiProtonChunkFitter().precompute_geometry(_zero_chunk())
        self.assertEqual(epoch, _CHUNK_EPOCH)
        assert rotation_matrices is not None
        self.assertEqual(rotation_matrices.shape, (_N_BINS - 1, 3, 3))

    def test_spice_failure_returns_none_rotations(self):
        """When the chunk falls outside SPICE coverage, PUI precompute_geometry returns None for rotations."""
        _, rotation_matrices = PuiProtonChunkFitter().precompute_geometry(_out_of_coverage_chunk())
        self.assertIsNone(rotation_matrices)


class TestPuiProtonChunkFitterFitChunk(SpiceTestCase):
    """Tests for `PuiProtonChunkFitter.fit_chunk` — uses real SPICE so that
    `rotate_rtn_to_dps` inside `derive_velocity_angles` runs unmocked."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.response = _swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS))
        efficiency_table = _efficiency_table()
        _populate_shared(cls.response, efficiency_table)
        cls.chunk, cls.rotations, cls.true_proton_velocity_rtn, _ = _build_truth_chunk(cls.response, efficiency_table)

    @classmethod
    def tearDownClass(cls):
        _clear_shared()
        super().tearDownClass()

    def _assert_all_ufloats_nan(self, result):
        for key in ["proton_sw_speed", "proton_sw_clock_angle", "proton_sw_deflection_angle"]:
            with self.subTest(key=key):
                self.assertTrue(np.isnan(result[key].nominal_value))

    def test_happy_path_recovers_truth_with_correct_angles_and_finite_uncertainty(self):
        """A clean PUI fit recovers the true bulk speed, matches the closed-form clock and deflection angles computed in the DPS frame (from the true RTN velocity rotated through real SPICE), and reports a finite positive speed uncertainty."""
        result = PuiProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, self.rotations)

        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self.assertAlmostEqual(
            result["proton_sw_speed"].nominal_value, _TRUE_BULK_SPEED, delta=5.0
        )
        velocity_dps_truth = rotate_rtn_to_dps(self.true_proton_velocity_rtn, _CHUNK_EPOCH)
        expected_clock = (
            np.degrees(np.arctan2(-velocity_dps_truth[1], -velocity_dps_truth[0])) % 360
        )
        expected_defl = np.degrees(np.arccos(-velocity_dps_truth[2] / _TRUE_BULK_SPEED))
        self.assertAlmostEqual(
            result["proton_sw_clock_angle"].nominal_value, expected_clock, delta=1.0
        )
        self.assertAlmostEqual(
            result["proton_sw_deflection_angle"].nominal_value, expected_defl, delta=1.0
        )
        self.assertGreater(result["proton_sw_speed"].std_dev, 0.0)

    def test_missing_rotations_sets_ephemeris_gap_and_nan_ufloats(self):
        """Missing rotations flag the PUI chunk EPHEMERIS_GAP and produce NaN UFloat outputs for speed, clock, and deflection."""
        result = PuiProtonChunkFitter().fit_chunk(self.chunk, _CHUNK_EPOCH, None)
        self.assertEqual(int(result["quality_flags"]), int(SwapiL3Flags.EPHEMERIS_GAP))
        self._assert_all_ufloats_nan(result)

    def test_nan_count_rate_short_circuits_with_nan_ufloats_and_none_flag(self):
        """A NaN in the PUI count rate short-circuits to NaN UFloat outputs but leaves the quality flag as NONE (the NaN propagates from L2, not from a fit failure)."""
        result = PuiProtonChunkFitter().fit_chunk(
            _with_nan_at(self.chunk, 0, 5), _CHUNK_EPOCH, self.rotations
        )
        self.assertEqual(result["quality_flags"], SwapiL3Flags.NONE)
        self._assert_all_ufloats_nan(result)


# ----- ParallelChunkRunner --------------------------------------------------


class _RecordingChunkFitter(ChunkFitter):
    """A real `ChunkFitter` that echoes the chunk's start time. Module-level
    so fork-spawned workers can locate it via inherited memory."""

    def precompute_geometry(self, chunk):
        return (int(chunk.sci_start_time[0]),)

    def fit_chunk(self, chunk, epoch):
        return {
            "epoch": epoch,
            "first_start_time": int(chunk.sci_start_time[0]),
        }


class _WarmCacheProbeChunkFitter(ChunkFitter):
    """A real `ChunkFitter` whose `fit_chunk` reports the `_passband_grid_cache` size
    seen by the worker process. Module-level for fork inheritance."""

    def precompute_geometry(self, chunk):
        return (int(chunk.sci_start_time[0]),)

    def fit_chunk(self, chunk, epoch):
        worker_response = chunk_fits._shared["swapi_response"]
        return {
            "epoch": epoch,
            "worker_cache_size": len(worker_response._passband_grid_cache),
        }


def _make_chunk_with_start_time(start_time):
    return SwapiL2Data(
        sci_start_time=np.array([start_time], dtype=np.int64),
        energy=np.zeros((1, _N_BINS)),
        coincidence_count_rate=np.zeros((1, _N_BINS)),
        coincidence_count_rate_uncertainty=np.zeros((1, _N_BINS)),
    )


class TestParallelChunkRunnerOrchestration(unittest.TestCase):
    """Tests for `ParallelChunkRunner.run` against a real fork-based Pool."""

    def tearDown(self):
        _clear_shared()

    def test_dispatches_per_chunk_and_stacks_outputs_in_chunk_order(self):
        """Two chunks with distinct start times produce a stacked output where each per-key array preserves chunk order across the real fork pool."""
        chunks = [
            _make_chunk_with_start_time(_EPOCH_TT2000),
            _make_chunk_with_start_time(_EPOCH_TT2000 + 12_000_000_000),
        ]
        runner = ParallelChunkRunner(
            swapi_response=_swapi_response_with_warm_cache(np.tile(_SCIENCE_V, _N_SWEEPS)),
            efficiency_table=_efficiency_table(),
        )

        result = runner.run(chunks, _RecordingChunkFitter())

        expected_epochs = np.array(
            [int(chunks[0].sci_start_time[0]), int(chunks[1].sci_start_time[0])]
        )
        np.testing.assert_array_equal(result["epoch"], expected_epochs)
        np.testing.assert_array_equal(result["first_start_time"], expected_epochs)

    def test_workers_see_parent_warm_cache_under_fork(self):
        """A passband grid cache populated in the parent before `runner.run` is visible inside each fork-spawned worker at the same size."""
        voltages = np.array([10.0, 50.0, 100.0])
        response = _swapi_response_with_warm_cache(voltages)
        parent_cache_size = len(response._passband_grid_cache)
        self.assertEqual(parent_cache_size, len(voltages))

        runner = ParallelChunkRunner(
            swapi_response=response, efficiency_table=_efficiency_table()
        )

        result = runner.run(
            [_make_chunk_with_start_time(_EPOCH_TT2000)], _WarmCacheProbeChunkFitter()
        )


        np.testing.assert_array_equal(
            result["worker_cache_size"], np.array([parent_cache_size])
        )


if __name__ == "__main__":
    unittest.main()
