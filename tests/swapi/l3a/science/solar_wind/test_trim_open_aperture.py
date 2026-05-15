import math
import unittest

import numpy as np

from imap_l3_processing.swapi.l3a.science.solar_wind.trim_open_aperture import (
    OA_SCAN_RESOLUTION,
    OA_SCAN_THRESHOLD,
    trim_open_aperture,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    SolarWindParams,
    bulk_speed,
    thermal_speed,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
)
from imap_l3_processing.swapi.response.passband_grid import build_passband_grid
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid
from tests.swapi._helpers import NOMINAL_SWAPI_TO_RTN_ROTATION, proton_params
from tests.swapi.response.test_passband_grid import _gaussian_values_df


# --- fixtures ---------------------------------------------------------------


def _make_response_grid(
    central_speed: float = 450.0,
    azimuthal_transmission: np.ndarray | None = None,
    azimuthal_transmission_spacing: float = 1.0,
) -> ResponseGrid:
    if azimuthal_transmission is None:
        # T(|az|) = 1 sampled on [0°, 180°] at 1° spacing → 181 nodes.
        azimuthal_transmission = np.ones(181)
    df = _gaussian_values_df(sigma_el=2.0)
    sg_passband = build_passband_grid(df)
    oa_passband = build_passband_grid(df)
    return ResponseGrid(
        sg_passband=sg_passband,
        oa_passband=oa_passband,
        central_speed=central_speed,
        # Picked arbitrarily — only the count-rate prefactor scales with it,
        # and tests here either compare to literal sentinels or assert
        # window-shape properties that are invariant to overall amplitude.
        central_effective_area=0.5,
        azimuthal_transmission=AzimuthalTransmissionGrid(
            values=azimuthal_transmission,
            spacing=azimuthal_transmission_spacing,
        ),
    )


def _proton_params_at_elevation(
    speed: float, bulk_el_deg: float, density: float = 5.0, temperature: float = 1.0e5
) -> SolarWindParams:
    v_x_swapi = -speed * math.cos(math.radians(bulk_el_deg))
    v_z_swapi = -speed * math.sin(math.radians(bulk_el_deg))
    v_rtn = NOMINAL_SWAPI_TO_RTN_ROTATION @ np.array(
        [v_x_swapi, 0.0, v_z_swapi]
    )
    return proton_params(
        velocity_rtn=tuple(v_rtn), density=density, temperature=temperature
    )


def _trim(
    rg: ResponseGrid,
    sw: SolarWindParams,
    *,
    azimuth_lo: float,
    azimuth_hi: float,
    sg_rate: float,
    rotation_matrix: np.ndarray | None = None,
    min_elevation: float = -10.0,
    max_elevation: float = 10.0,
):
    if rotation_matrix is None:
        rotation_matrix = NOMINAL_SWAPI_TO_RTN_ROTATION
  
    return trim_open_aperture(
        rg,
        sw,
        rotation_matrix,
        min_elevation,
        max_elevation,
        azimuth_lo=azimuth_lo,
        azimuth_hi=azimuth_hi,
        sg_rate=sg_rate,
    )


# --- tests ------------------------------------------------------------------


class TestSkipsWhenAzimuthWindowIsEmpty(unittest.TestCase):
    """A degenerate input-window short-circuits to the (0, 0) sentinel without scanning."""

    def test_skips_when_window_has_zero_width(self):
        """A zero-width azimuth clamp short-circuits to the (0, 0) sentinel without scanning."""
        rg = _make_response_grid()
        sw = proton_params()
        lo, hi = _trim(rg, sw, azimuth_lo=30.0, azimuth_hi=30.0, sg_rate=10.0)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_skips_when_window_is_inverted(self):
        """An inverted clamp (`hi < lo`) also short-circuits to the (0, 0) sentinel."""
        rg = _make_response_grid()
        sw = proton_params()
        lo, hi = _trim(rg, sw, azimuth_lo=50.0, azimuth_hi=20.0, sg_rate=10.0)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestSkipsWhenSgRateRelativeFloorDominates(unittest.TestCase):
    """When the OA upper-bound rate falls below `1e-3·sg_rate`, OA is skipped."""

    def test_skips_when_oa_upper_bound_below_sg_relative_floor(self):
        """A cold bulk far from OA with a huge sg_rate trips the relative-fraction floor and skips OA."""
        rg = _make_response_grid()
        sw = proton_params(temperature=1.0e4)
        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=1.0e6)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestSkipsWhenAbsoluteRateFloorDominates(unittest.TestCase):
    """When the OA upper-bound rate falls below the 0.1 Hz absolute floor, OA is skipped."""

    def test_skips_when_oa_upper_bound_below_absolute_floor(self):
        """With sg_rate=0, the 0.1 Hz absolute floor alone forces a skip for a cold bulk far from OA."""
        rg = _make_response_grid()
        sw = proton_params(temperature=1.0e4)
        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_skips_when_sg_relative_below_absolute_floor(self):
        """When `1e-3·sg_rate = 0.05 Hz < 0.1 Hz`, the `max(...)` picks the absolute floor and still skips."""
        rg = _make_response_grid()
        sw = proton_params(temperature=1.0e4)
        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=50.0)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestReturnsTrimmedWindowWhenOaIsRelevant(unittest.TestCase):
    """A bulk well aligned with OA yields a trimmed window contained in the input clamp."""

    def setUp(self):
        # Under NOMINAL_SWAPI_TO_RTN_ROTATION, bulk along -T_RTN (i.e.
        # (0, -450, 0)) maps to v_swapi = (-450, 0, 0) → azimuth_inst = +90°.
        self.rg = _make_response_grid()
        self.sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0))
        self.az_lo_in = 20.0
        self.az_hi_in = 150.0
        # zero sg_rate keeps only the 0.1 Hz floor — the upper-bound estimate
        # for a 5 cm⁻³, 100 kK plasma well aligned with OA+ exceeds that easily.
        self.sg_rate = 0.0
        self.lo, self.hi = _trim(
            self.rg,
            self.sw,
            azimuth_lo=self.az_lo_in,
            azimuth_hi=self.az_hi_in,
            sg_rate=self.sg_rate,
        )

    def test_returned_window_has_positive_width(self):
        """A bulk deep inside OA+ produces a non-empty trimmed window."""
        self.assertGreater(self.hi, self.lo)

    def test_returned_window_lies_inside_input_clamp(self):
        """The trimmed window is contained in the input azimuth clamp."""
        self.assertGreaterEqual(self.lo, self.az_lo_in)
        self.assertLessEqual(self.hi, self.az_hi_in)

    def test_returned_endpoints_lie_on_the_scan_grid(self):
        """Both endpoints coincide with nodes of `linspace(az_lo, az_hi, OA_SCAN_RESOLUTION)`."""
        scan = np.linspace(self.az_lo_in, self.az_hi_in, OA_SCAN_RESOLUTION)
        self.assertTrue(np.any(np.isclose(scan, self.lo)))
        self.assertTrue(np.any(np.isclose(scan, self.hi)))

    def test_returned_window_brackets_the_bulk_azimuth(self):
        """The trimmed window strictly contains the bulk azimuth (+90° here)."""
        bulk_azimuth = 90.0
        self.assertLess(self.lo, bulk_azimuth)
        self.assertGreater(self.hi, bulk_azimuth)


class TestTrimAt1eMinus6Threshold(unittest.TestCase):
    """OA_SCAN_THRESHOLD selects one scan node outside the first/last above-threshold node."""

    def test_endpoints_expand_one_node_past_threshold_crossings(self):
        """The trim returns one scan node outside the first/last node where the integrand exceeds OA_SCAN_THRESHOLD × max."""
        # `interpolate_azimuthal_transmission` hard-codes T(|az|)=1 on the OA
        # plateau [31°, 115°], which contains the [20°, 150°] scan window's
        # interior. The integrand reduces to the Maxwellian centered at the
        # bulk azimuth, which we recompute here.
        rg = _make_response_grid()
        sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0))  # bulk_az = +90°, bulk_el = 0°

        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)

        scan = np.linspace(20.0, 150.0, OA_SCAN_RESOLUTION)
        sigma = thermal_speed(sw)
        speed = bulk_speed(sw)
        central = rg.central_speed
        delta_v_sq = (
            central**2 + speed**2
            - 2.0 * central * speed * np.cos(np.radians(scan - 90.0))
        )
        integrand = np.exp(-delta_v_sq / (2.0 * sigma**2))
        above_idx = np.where(integrand > OA_SCAN_THRESHOLD * integrand.max())[0]
        self.assertGreater(above_idx.size, 0)
        expected_lo = scan[max(int(above_idx[0]) - 1, 0)]
        expected_hi = scan[min(int(above_idx[-1]) + 1, OA_SCAN_RESOLUTION - 1)]
        self.assertAlmostEqual(lo, expected_lo)
        self.assertAlmostEqual(hi, expected_hi)


class TestSymmetryBetweenOaPositiveAndNegative(unittest.TestCase):
    """OA+ and OA- windows are exact mirror images for mirror-symmetric bulks."""

    def test_mirror_symmetric_bulk_yields_mirror_symmetric_windows(self):
        """Bulks at ±90° produce OA+ and OA- windows that are exact mirror images of each other."""
        rg = _make_response_grid()

        # Under NOMINAL_SWAPI_TO_RTN_ROTATION:
        #   v_rtn = (0, -450, 0) → v_swapi = (-450, 0, 0) → az_inst = +90° (OA+).
        #   v_rtn = (0, +450, 0) → v_swapi = (+450, 0, 0) → az_inst = -90° (OA-).
        sw_pos = proton_params(velocity_rtn=(0.0, -450.0, 0.0))
        sw_neg = proton_params(velocity_rtn=(0.0, +450.0, 0.0))

        lo_pos, hi_pos = _trim(rg, sw_pos, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)
        # OA-: same magnitudes, mirrored around 0.
        lo_neg, hi_neg = _trim(rg, sw_neg, azimuth_lo=-150.0, azimuth_hi=-20.0, sg_rate=0.0)

        # The implementation picks scan-grid nodes by index, and
        # `linspace(20, 150, N)` mirrors `linspace(-150, -20, N)` up to a
        # single ULP from `linspace`'s internal accumulation order.
        np.testing.assert_allclose(lo_neg, -hi_pos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(hi_neg, -lo_pos, atol=1e-12, rtol=0)


class TestSkipPolicyAgainstSgRate(unittest.TestCase):
    """Raising sg_rate above the OA upper-bound rate flips an open window to the (0, 0) sentinel."""

    def test_high_sg_rate_forces_skip_for_otherwise_aligned_bulk(self):
        """A bulk that yields a real window at sg_rate=0 flips to the (0, 0) sentinel once sg_rate is cranked above the OA upper bound."""
        rg = _make_response_grid()
        sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0))  # az_inst = +90°

        lo_open, hi_open = _trim(
            rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0
        )
        self.assertGreater(hi_open, lo_open)

        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=1.0e15)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestScanAtBulkElevationClampedToOaRange(unittest.TestCase):
    """A bulk outside `[min_elevation, max_elevation]` is clamped to the OA elevation range during the scan."""

    def test_bulk_elevation_above_range_yields_non_empty_window(self):
        """A bulk at +20° elevation (above the +10° OA cap) still yields a non-empty trimmed window after the scan elevation is clamped to +10°."""
        rg = _make_response_grid()
        sw = _proton_params_at_elevation(speed=450.0, bulk_el_deg=20.0)

        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main()
