"""Tests for `solar_wind.integration_limits` — the per-region quadrature
builders that bound the (elevation, azimuth, speed) integration windows fed
to the forward model.

The behavior under test is described in `docs/swapi/solar-wind-moments.md`
under "Integration Method" — specifically the "Angular limits" and
"Speed limits" subsections. The two big rules are:

1. The angular window is the rectangular envelope of the Maxwellian's
   angular falloff (from on-axis to the EPSILON cutoff), expressed in the
   instrument frame around the bulk direction, then clamped to each
   region's elevation range and azimuth band:
     SG    : az ∈ [-20°, +20°]
     OA-   : az ∈ [-150°, -20°]
     OA+   : az ∈ [+20°, +150°]
   If either dimension collapses to zero width the region is skipped.

2. The speed window is the intersection of `bulk_speed ± k·vth`
   (with k = SPEED_HALF_WIDTH_VTH = 6) and the per-elevation passband
   speed range `[r_min(θ)·v0, r_max(θ)·v0]`. For cold plasma the
   first interval narrows the window inside the passband; for warm
   plasma it's the passband bounds that bind."""

import math
import unittest

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.azimuthal_regions import (
    REGION_OPEN_APERTURE_NEG,
    REGION_OPEN_APERTURE_POS,
    REGION_SUNGLASSES,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.integration_limits import (
    EPSILON,
    SPEED_HALF_WIDTH_VTH,
    AngularQuadrature,
    _clamp,
    _clamp_window,
    _maxwellian_angular_extent,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    SolarWindParams,
    bulk_speed,
    thermal_speed,
    thermal_speed_to_temperature,
)
from imap_l3_processing.swapi.response.passband_grid import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path

# --- module-level fixtures: real instrument-team CSVs and one warmed grid ----

# A 1 keV proton ESA voltage gives a comfortably typical central speed
# (~437 km/s) and a passband elevation range that contains 0.
_ESA_VOLTAGE = 1000.0 / SWAPI_K_FACTOR

# Region azimuth bands fixed by the doc.
_SG_AZIMUTH_LO_DEG = -20.0
_SG_AZIMUTH_HI_DEG = +20.0
_OA_AZIMUTH_INNER_DEG = 20.0
_OA_AZIMUTH_OUTER_DEG = 150.0

# Typical solar-wind reference values for `_proton_params`.
_DENSITY_CM3 = 5.0
_TEMPERATURE_K = 100_000.0
_BULK_SPEED_KM_S = 450.0


def _load_response() -> SwapiResponse:
    """Build a `SwapiResponse` from the shipped instrument-team CSVs."""
    return SwapiResponse.from_files(
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


def _proton_params(
    *,
    velocity_rtn=(0.0, -_BULK_SPEED_KM_S, 0.0),
    temperature: float = _TEMPERATURE_K,
    density: float = _DENSITY_CM3,
) -> SolarWindParams:
    """Solar-wind proton state. The default RTN velocity is along -Y_RTN, which
    the identity rotation matrix maps to az=0, el=0 in the instrument frame —
    the boresight direction (SG region).

    Note: this default differs from `tests/.../test_state.py`'s helper, which
    uses a generic velocity vector. We pick the boresight-aligned default here
    because most tests in this file want SG to be the active region."""
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.array(velocity_rtn),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


class TestClamp(unittest.TestCase):
    """`_clamp(x, lower, upper)` returns its input clipped into the bound
    interval `[lower, upper]`."""

    def test_value_inside_bounds_is_returned_unchanged(self):
        self.assertEqual(_clamp(0.5, -1.0, 1.0), 0.5)

    def test_value_below_lower_bound_clamps_to_lower(self):
        self.assertEqual(_clamp(-2.0, -1.0, 1.0), -1.0)

    def test_value_above_upper_bound_clamps_to_upper(self):
        self.assertEqual(_clamp(3.0, -1.0, 1.0), 1.0)


class TestClampWindow(unittest.TestCase):
    """`_clamp_window(center, half_width, lower, upper)` — clamps a symmetric
    interval `[center − half_width, center + half_width]` to a bounding box.
    Used to fold the Maxwellian's rectangular angular extent into each
    region's elevation range and azimuth band."""

    def test_window_fully_inside_bounds_returns_centered_interval(self):
        lo, hi = _clamp_window(0.0, 5.0, -10.0, 10.0)
        self.assertEqual((lo, hi), (-5.0, 5.0))

    def test_lower_edge_clamped_to_bound_when_window_overhangs(self):
        # half_width=5 puts the lower edge at -7, which is outside [-2, 10];
        # it clamps to the lower bound.
        lo, hi = _clamp_window(-2.0, 5.0, -2.0, 10.0)
        self.assertEqual(lo, -2.0)
        self.assertEqual(hi, 3.0)

    def test_window_entirely_outside_upper_bound_collapses_to_zero_width(self):
        # If the window's lower edge is already past the upper bound, both
        # sides clamp to `upper_bound` and the caller sees zero width — this
        # is the trigger `_angular_limits` uses to skip a region.
        lo, hi = _clamp_window(100.0, 5.0, -10.0, 10.0)
        self.assertEqual(lo, 10.0)
        self.assertEqual(hi, 10.0)


class TestAngularQuadratureNamedTuple(unittest.TestCase):
    """The `AngularQuadrature` field order is part of the public contract
    consumed by the integrator's JIT loops."""

    def test_namedtuple_field_order_matches_jit_unpacking_order(self):
        # The JIT integrator unpacks by position (numba does not honor field
        # names at runtime), so reordering this tuple is a silent breakage.
        self.assertEqual(
            AngularQuadrature._fields,
            (
                "elevation_points",
                "elevation_weights",
                "azimuth_points",
                "azimuth_weights",
                "sin_elevation",
                "cos_elevation",
                "sin_azimuth",
                "cos_azimuth",
                "transmission_azimuth",
            ),
        )


class TestMaxwellianAngularExtent(unittest.TestCase):
    """`_maxwellian_angular_extent(sw, central_speed, EPSILON)` returns the
    half-angle Δα at which the Maxwellian's angular falloff at speed v=v0
    has decayed to EPSILON of its on-axis value — i.e. the half-width of
    the angular window outside of which the contribution to the integral
    is negligible. The closed form is
        Δα = arccos(clamp(σ²·ln(ε)/(v0·v_b) + 1, -1, +1)).
    See `docs/swapi/solar-wind-moments.md` §Angular limits."""

    def test_matches_closed_form_for_typical_proton(self):
        sw = _proton_params()
        central_speed = 437.0
        sigma = thermal_speed(sw)
        speed = bulk_speed(sw)
        cos_theta = sigma**2 * math.log(EPSILON) / (central_speed * speed) + 1
        expected_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))
        self.assertAlmostEqual(
            _maxwellian_angular_extent(sw, central_speed, EPSILON),
            expected_deg,
            places=10,
        )

    def test_clamps_to_180_for_very_broad_distribution(self):
        # When σ is large enough that the arccos argument falls below -1, the
        # outer `clamp(..., -1, +1)` saturates the input at -1 and Δα = arccos(-1)
        # = 180°. Push T sky-high to trigger.
        sw = _proton_params(temperature=1e10)
        self.assertAlmostEqual(
            _maxwellian_angular_extent(sw, 437.0, EPSILON), 180.0, places=10
        )

    def test_cold_plasma_gives_narrower_extent_than_hot_plasma(self):
        cold = _proton_params(temperature=10_000.0)
        hot = _proton_params(temperature=1_000_000.0)
        self.assertLess(
            _maxwellian_angular_extent(cold, 437.0, EPSILON),
            _maxwellian_angular_extent(hot, 437.0, EPSILON),
        )


class TestSpeedWindowMissesPassband(unittest.TestCase):
    """`speed_window_misses_passband(sw, response_grid)` is the cheap pre-check
    for whether the entire `bulk_speed ± k·vth` window lies outside the
    on-axis passband speed range. Returns True only when the windows are
    fully disjoint; the caller skips all per-elevation integration in that
    case (see §Speed limits in the doc)."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()
        cls.response.warm_cache([_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(_ESA_VOLTAGE, 1.0)

    def test_returns_false_when_bulk_speed_sits_in_passband(self):
        # 450 km/s ± 6·σ trivially overlaps the on-axis passband ≈ [417, 455]
        # km/s at 1 keV.
        sw = _proton_params()
        self.assertFalse(speed_window_misses_passband(sw, self.response_grid))

    def test_returns_true_when_bulk_speed_above_passband(self):
        # 1500 km/s ± 6·σ is entirely above the ~455 km/s upper edge.
        sw = _proton_params(velocity_rtn=(0.0, -1500.0, 0.0))
        self.assertTrue(speed_window_misses_passband(sw, self.response_grid))

    def test_returns_true_when_bulk_speed_below_passband(self):
        # 100 km/s ± 6·σ is entirely below the ~417 km/s lower edge.
        sw = _proton_params(velocity_rtn=(0.0, -100.0, 0.0))
        self.assertTrue(speed_window_misses_passband(sw, self.response_grid))

    def test_window_overlaps_passband_when_k_sigma_exceeds_offset_above_edge(self):
        # Place bulk 30 km/s above the upper passband edge (30 km/s = ~1.5x the
        # k·σ threshold we'll test, so we can bracket overlap/non-overlap by
        # changing σ alone). With σ chosen so k·σ = 50 km/s > 30 km/s, the
        # window still reaches into the passband ⇒ should NOT report a miss.
        _, ratio_hi = speed_ratio_range_at_elevation(
            self.response_grid.sg_passband, 0.0
        )
        v_passband_hi = self.response_grid.central_speed * ratio_hi
        bulk = v_passband_hi + 30.0
        sigma_overlap = 50.0 / SPEED_HALF_WIDTH_VTH
        T_overlap = thermal_speed_to_temperature(sigma_overlap, PROTON_MASS_KG)
        sw_overlap = _proton_params(
            velocity_rtn=(0.0, -bulk, 0.0), temperature=T_overlap
        )
        self.assertFalse(speed_window_misses_passband(sw_overlap, self.response_grid))

    def test_window_detaches_from_passband_when_k_sigma_falls_below_offset(self):
        # Same geometry as above (bulk 30 km/s above the upper edge), but with
        # σ chosen so k·σ = 5 km/s ≪ 30 km/s. The window now stays above the
        # passband ⇒ should report a miss. This pair, together with the
        # overlap-case above, pins SPEED_HALF_WIDTH_VTH as the actual scaling
        # constant (not some other factor of σ).
        _, ratio_hi = speed_ratio_range_at_elevation(
            self.response_grid.sg_passband, 0.0
        )
        v_passband_hi = self.response_grid.central_speed * ratio_hi
        bulk = v_passband_hi + 30.0
        sigma_detached = 5.0 / SPEED_HALF_WIDTH_VTH
        T_detached = thermal_speed_to_temperature(sigma_detached, PROTON_MASS_KG)
        sw_detached = _proton_params(
            velocity_rtn=(0.0, -bulk, 0.0), temperature=T_detached
        )
        self.assertTrue(speed_window_misses_passband(sw_detached, self.response_grid))


class TestGetAngularQuadratureSunglassesRegion(unittest.TestCase):
    """For the SG region, `get_angular_quadrature` builds a 21×21 GL quadrature
    over the rectangle `bulk ± Δα` (Maxwellian extent) clamped to the SG
    elevation range and `[-20°, +20°]` in azimuth (per the doc table)."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()
        cls.response.warm_cache([_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(_ESA_VOLTAGE, 1.0)
        cls.identity = np.eye(3)

    def test_sg_region_is_active_for_sun_pointed_bulk(self):
        # Bulk along -Y_RTN ⇒ az=0, el=0 (boresight). With T=1e5 K the
        # Maxwellian extent is < 20°, so SG should be active.
        sw = _proton_params()
        skip, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertFalse(skip)
        self.assertIsNotNone(quadrature)

    def test_sg_azimuth_window_is_inside_minus_20_to_plus_20_degrees(self):
        sw = _proton_params()
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        # All Gauss-Legendre nodes must lie strictly inside [-20°, +20°];
        # this is the SG azimuth band per the doc.
        self.assertGreaterEqual(quadrature.azimuth_points.min(), _SG_AZIMUTH_LO_DEG)
        self.assertLessEqual(quadrature.azimuth_points.max(), _SG_AZIMUTH_HI_DEG)

    def test_sg_azimuth_window_clamps_to_plus_minus_20_for_broad_distribution(self):
        # A very hot plasma's angular extent saturates at 180°, so the
        # azimuth window must clamp exactly to the [-20°, +20°] SG band.
        sw = _proton_params(temperature=1e10)
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        # GL nodes lie strictly inside the band.
        np.testing.assert_array_less(
            quadrature.azimuth_points, _SG_AZIMUTH_HI_DEG + 1e-9
        )
        np.testing.assert_array_less(
            _SG_AZIMUTH_LO_DEG - 1e-9, quadrature.azimuth_points
        )
        # The outermost 21-node GL node sits ≈0.125° inside the edge of a 40°
        # window (analytically `20°·(1 - x_GL[-1])` ≈ 0.125°). 1.5° is a loose
        # cushion above that — only meant to catch the case where the window
        # collapses or shifts away from the edge entirely.
        self.assertGreater(quadrature.azimuth_points.max(), _SG_AZIMUTH_HI_DEG - 1.5)
        self.assertLess(quadrature.azimuth_points.min(), _SG_AZIMUTH_LO_DEG + 1.5)

    def test_sg_elevation_window_clamps_into_passband_elevation_range(self):
        sw = _proton_params(temperature=1e10)
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        sg_el_lo, sg_el_hi = self.response_grid.sg_passband.elevation_range
        self.assertGreaterEqual(quadrature.elevation_points.min(), sg_el_lo)
        self.assertLessEqual(quadrature.elevation_points.max(), sg_el_hi)

    def test_sg_quadrature_has_21_nodes_in_each_angular_dimension(self):
        # The doc fixes (Nθ, Nφ, Nv) = (21, 21, 15).
        sw = _proton_params()
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertEqual(quadrature.elevation_points.shape, (21,))
        self.assertEqual(quadrature.azimuth_points.shape, (21,))

    def test_sin_cos_caches_match_their_angle_arrays(self):
        sw = _proton_params()
        _, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        np.testing.assert_allclose(
            quadrature.sin_elevation, np.sin(np.radians(quadrature.elevation_points))
        )
        np.testing.assert_allclose(
            quadrature.cos_elevation, np.cos(np.radians(quadrature.elevation_points))
        )
        np.testing.assert_allclose(
            quadrature.sin_azimuth, np.sin(np.radians(quadrature.azimuth_points))
        )
        np.testing.assert_allclose(
            quadrature.cos_azimuth, np.cos(np.radians(quadrature.azimuth_points))
        )

    def test_sg_region_skipped_when_bulk_far_from_sg_band(self):
        # Bulk at az ≈ 90° (along -X_RTN under identity rotation) is 70° away
        # from the SG ±20° band. With a typical-temperature Maxwellian (extent
        # ≈ a few °), there is no overlap with SG ⇒ skip.
        sw = _proton_params(velocity_rtn=(-_BULK_SPEED_KM_S, 0.0, 0.0))
        skip, quadrature = get_angular_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, self.identity, 0.0
        )
        self.assertTrue(skip)
        self.assertIsNone(quadrature)


class TestGetAngularQuadratureOpenAperturePositive(unittest.TestCase):
    """OA+ covers az ∈ [+20°, +150°] (the doc table). For a bulk pointed at
    az ≈ 90° (along -X_RTN under identity rotation), this region is the
    active one — OA- and SG should both be skipped."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()
        cls.response.warm_cache([_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(_ESA_VOLTAGE, 1.0)
        cls.identity = np.eye(3)
        # Bulk pointed at az=+90°: identity rotation makes v_inst = (-v, 0, 0),
        # giving azimuth = atan2(v, 0) = 90° and elevation = 0°.
        cls.sw_bulk_at_90 = _proton_params(
            velocity_rtn=(-_BULK_SPEED_KM_S, 0.0, 0.0)
        )

    def test_oa_positive_region_is_active_for_bulk_at_90_deg_azimuth(self):
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_90,
            self.response_grid,
            REGION_OPEN_APERTURE_POS,
            self.identity,
            0.0,
        )
        self.assertFalse(skip)
        self.assertIsNotNone(quadrature)

    def test_oa_positive_azimuth_window_is_inside_20_to_150_degrees(self):
        _, quadrature = get_angular_quadrature(
            self.sw_bulk_at_90,
            self.response_grid,
            REGION_OPEN_APERTURE_POS,
            self.identity,
            0.0,
        )
        # All nodes lie strictly on the +OA side, between +20° and +150°.
        self.assertGreaterEqual(
            quadrature.azimuth_points.min(), _OA_AZIMUTH_INNER_DEG
        )
        self.assertLessEqual(
            quadrature.azimuth_points.max(), _OA_AZIMUTH_OUTER_DEG
        )

    def test_oa_negative_region_is_skipped_for_bulk_at_plus_90_deg_azimuth(self):
        # Bulk at +90° is 110° from the nearest edge of the OA- band
        # [-150°, -20°]. The Maxwellian's angular extent at typical T is far
        # smaller than that gap, so there is no overlap with OA- ⇒ skip.
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_90,
            self.response_grid,
            REGION_OPEN_APERTURE_NEG,
            self.identity,
            0.0,
        )
        self.assertTrue(skip)
        self.assertIsNone(quadrature)


class TestGetAngularQuadratureOpenApertureNegative(unittest.TestCase):
    """OA- covers az ∈ [-150°, -20°] (the doc table; the negative-azimuth
    half of the open aperture). For a bulk at az ≈ -90°, OA- is active and
    OA+ should be skipped."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()
        cls.response.warm_cache([_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(_ESA_VOLTAGE, 1.0)
        cls.identity = np.eye(3)
        # Bulk at az=-90°: v_inst = (+v, 0, 0) ⇒ az = atan2(-v, 0) = -90°.
        cls.sw_bulk_at_minus_90 = _proton_params(
            velocity_rtn=(+_BULK_SPEED_KM_S, 0.0, 0.0)
        )

    def test_oa_negative_region_is_active_for_bulk_at_minus_90_deg_azimuth(self):
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_minus_90,
            self.response_grid,
            REGION_OPEN_APERTURE_NEG,
            self.identity,
            0.0,
        )
        self.assertFalse(skip)
        self.assertIsNotNone(quadrature)

    def test_oa_negative_azimuth_window_is_inside_minus_150_to_minus_20(self):
        _, quadrature = get_angular_quadrature(
            self.sw_bulk_at_minus_90,
            self.response_grid,
            REGION_OPEN_APERTURE_NEG,
            self.identity,
            0.0,
        )
        # All nodes lie strictly on the -OA side, between -150° and -20°.
        self.assertGreaterEqual(
            quadrature.azimuth_points.min(), -_OA_AZIMUTH_OUTER_DEG
        )
        self.assertLessEqual(
            quadrature.azimuth_points.max(), -_OA_AZIMUTH_INNER_DEG
        )

    def test_oa_positive_region_is_skipped_for_bulk_at_minus_90_deg_azimuth(self):
        skip, quadrature = get_angular_quadrature(
            self.sw_bulk_at_minus_90,
            self.response_grid,
            REGION_OPEN_APERTURE_POS,
            self.identity,
            0.0,
        )
        self.assertTrue(skip)
        self.assertIsNone(quadrature)


class TestGetSpeedQuadrature(unittest.TestCase):
    """`get_speed_quadrature(sw, response_grid, region, elevation)` returns a
    15-node Gauss-Legendre quadrature for the speed integral (per the doc:
    Nv = 15). The window is the intersection of `bulk_speed ± k·vth` with
    the per-elevation passband range `[r_min(θ)·v0, r_max(θ)·v0]`."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()
        cls.response.warm_cache([_ESA_VOLTAGE])
        cls.response_grid = cls.response.get_response_grid(_ESA_VOLTAGE, 1.0)

    def test_speed_window_lies_inside_per_elevation_passband(self):
        # For warm plasma the passband is the binding constraint at on-axis
        # elevation: every node must lie in `[r_min(θ)·v0, r_max(θ)·v0]`.
        sw = _proton_params()
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        ratio_lo, ratio_hi = speed_ratio_range_at_elevation(
            self.response_grid.sg_passband, 0.0
        )
        passband_lo = self.response_grid.central_speed * ratio_lo
        passband_hi = self.response_grid.central_speed * ratio_hi
        self.assertGreaterEqual(speed_quadrature.points.min(), passband_lo)
        self.assertLessEqual(speed_quadrature.points.max(), passband_hi)

    def test_cold_plasma_window_lies_inside_bulk_plus_minus_k_vth(self):
        # When `bulk_speed ± k·vth` lies entirely *inside* the passband, the
        # rule says the speed window narrows to that interval (see
        # §Speed limits in the doc). At T=480 K, σ ≈ 2 km/s ⇒ k·σ ≈ 12 km/s
        # at bulk=437.7 km/s, well inside the on-axis passband [≈418, 455].
        sw = _proton_params(
            velocity_rtn=(0.0, -self.response_grid.central_speed, 0.0),
            temperature=480.0,
        )
        sigma = thermal_speed(sw)
        speed = bulk_speed(sw)
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
        window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma
        # All GL nodes are interior to `[bulk - k·σ, bulk + k·σ]`.
        self.assertGreaterEqual(speed_quadrature.points.min(), window_lo)
        self.assertLessEqual(speed_quadrature.points.max(), window_hi)

    def test_cold_plasma_outer_gl_nodes_sit_close_to_window_edges(self):
        # Companion to the previous test: pin that the window endpoints
        # actually equal `bulk ± k·σ` (rather than being some smaller subset of
        # it) by checking the outermost GL nodes sit close to those edges.
        # On a 24 km/s window the outermost 15-node GL node is at
        # `12·(1 − x_GL[-1])` ≈ 0.144 km/s from the edge; 2.0 km/s is a loose
        # cushion that catches any drift away from the analytic position.
        sw = _proton_params(
            velocity_rtn=(0.0, -self.response_grid.central_speed, 0.0),
            temperature=480.0,
        )
        sigma = thermal_speed(sw)
        speed = bulk_speed(sw)
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
        window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma
        self.assertLess(window_hi - speed_quadrature.points.max(), 2.0)
        self.assertLess(speed_quadrature.points.min() - window_lo, 2.0)

    def test_speed_quadrature_arrays_are_all_15_long(self):
        sw = _proton_params()
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        self.assertEqual(speed_quadrature.points.shape, (15,))
        self.assertEqual(speed_quadrature.weights.shape, (15,))
        self.assertEqual(speed_quadrature.speed_cubed_times_passband.shape, (15,))

    def test_returns_skip_when_bulk_speed_window_does_not_overlap_passband(self):
        # Bulk speed far above passband upper edge so `bulk - k·σ` already
        # exceeds the passband upper bound — no overlap, skip returned.
        sw = _proton_params(
            velocity_rtn=(0.0, -1500.0, 0.0), temperature=10_000.0
        )
        skip, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        self.assertTrue(skip)
        self.assertIsNone(speed_quadrature)

    def test_uses_region_specific_passband(self):
        # OA and SG passbands have different speed-ratio bounds. With a hot
        # plasma so the passband is the binding constraint in both regions,
        # the OA window should match the OA bounds — distinct from the SG
        # window — at the same elevation.
        sw = _proton_params(temperature=1_000_000.0)
        _, sg_quad = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        _, oa_quad = get_speed_quadrature(
            sw, self.response_grid, REGION_OPEN_APERTURE_POS, 0.0
        )
        sg_ratio_lo, _ = speed_ratio_range_at_elevation(
            self.response_grid.sg_passband, 0.0
        )
        oa_ratio_lo, _ = speed_ratio_range_at_elevation(
            self.response_grid.oa_passband, 0.0
        )
        sg_lo = self.response_grid.central_speed * sg_ratio_lo
        oa_lo = self.response_grid.central_speed * oa_ratio_lo
        # Each quadrature's lower edge sits at its own region's passband edge.
        self.assertGreaterEqual(sg_quad.points.min(), sg_lo)
        self.assertGreaterEqual(oa_quad.points.min(), oa_lo)
        # And the two windows are actually distinct — the SG and OA passbands
        # have different speed-ratio bounds, so a region mix-up would be
        # silently caught here.
        self.assertNotEqual(sg_quad.points.min(), oa_quad.points.min())

    def test_speed_cubed_times_passband_is_v3_times_normalized_passband_at_each_node(self):
        # The third array of `SpeedQuadrature` is `v³ · P(v/v0, θ) / P(0, 1)`
        # evaluated at every node, used by the integrator to hoist the
        # speed-and-passband-only factor out of the inner exp loop. Verify
        # the formula by reconstructing it from the public passband helpers.
        sw = _proton_params()
        _, speed_quadrature = get_speed_quadrature(
            sw, self.response_grid, REGION_SUNGLASSES, 0.0
        )
        on_axis_peak = interpolate_passband(
            self.response_grid.sg_passband, 0.0, 1.0
        )
        expected = np.array(
            [
                v**3
                * interpolate_passband(
                    self.response_grid.sg_passband,
                    0.0,
                    v / self.response_grid.central_speed,
                )
                / on_axis_peak
                for v in speed_quadrature.points
            ]
        )
        np.testing.assert_allclose(
            speed_quadrature.speed_cubed_times_passband, expected
        )


if __name__ == "__main__":
    unittest.main()
