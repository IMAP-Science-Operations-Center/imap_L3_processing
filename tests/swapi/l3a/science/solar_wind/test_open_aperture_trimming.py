import math
import unittest

import numpy as np

from imap_l3_processing.swapi.l3a.science.solar_wind.open_aperture_trimming import (
    OA_SCAN_RESOLUTION,
    trim_oa_azimuth_by_integrand,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    SolarWindParams,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
)
from imap_l3_processing.swapi.response.passband_grid import build_passband_grid
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid
from tests.swapi._helpers import proton_params
from tests.swapi.response.test_passband_grid import _gaussian_values_df


# --- fixtures ---------------------------------------------------------------


def _make_response_grid(
    central_speed: float = 450.0,
    azimuthal_transmission: np.ndarray | None = None,
    azimuthal_transmission_spacing: float = 1.0,
) -> ResponseGrid:
    """A real ResponseGrid backed by a real PassbandGrid. The transmission
    curve is a long flat plateau by default so trim threshold-crossings come
    from the Maxwellian alone unless the test overrides it."""
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
    """Bulk velocity in the RTN frame placed so that, under the identity
    rotation, `bulk_az_inst = +90°` and `bulk_el_inst = bulk_el_deg`.

    Derivation: with `R = I`, `θ_b = arcsin(-v_z / |v|)` and
    `φ_b = atan2(-v_x, -v_y)`. Putting `v_y = 0` and `v_x = -|v|·cos(el)`
    gives `φ_b = atan2(|v|·cos(el), 0) = +90°`. Putting
    `v_z = -|v|·sin(el)` gives `θ_b = arcsin(sin(el)) = el`."""
    v_x = -speed * math.cos(math.radians(bulk_el_deg))
    v_z = -speed * math.sin(math.radians(bulk_el_deg))
    return proton_params(
        velocity_rtn=(v_x, 0.0, v_z), density=density, temperature=temperature
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
    """Thin wrapper that fixes the boilerplate prefix every test reuses
    (identity rotation, OA elevation clamp [-10°, +10°]) so the varying
    inputs — azimuth window and sg_rate — stay visually obvious."""
    if rotation_matrix is None:
        rotation_matrix = np.eye(3)
    return trim_oa_azimuth_by_integrand(
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
    """Tests for `trim_oa_azimuth_by_integrand`, degenerate input-window branch."""

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
    """Tests for `trim_oa_azimuth_by_integrand`, `1e-3·sg_rate` skip branch."""

    def test_skips_when_oa_upper_bound_below_sg_relative_floor(self):
        """A cold bulk far from OA with a huge sg_rate trips the relative-fraction floor and skips OA."""
        rg = _make_response_grid()
        sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0), temperature=1.0e4)
        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=1.0e6)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestSkipsWhenAbsoluteRateFloorDominates(unittest.TestCase):
    """Tests for `trim_oa_azimuth_by_integrand`, 0.1 Hz absolute-floor skip branch."""

    def test_skips_when_oa_upper_bound_below_absolute_floor(self):
        """With sg_rate=0, the 0.1 Hz absolute floor alone forces a skip for a cold bulk far from OA."""
        rg = _make_response_grid()
        sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0), temperature=1.0e4)
        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_skips_when_sg_relative_below_absolute_floor(self):
        """When `1e-3·sg_rate = 0.05 Hz < 0.1 Hz`, the `max(...)` picks the absolute floor and still skips."""
        rg = _make_response_grid()
        sw = proton_params(velocity_rtn=(0.0, -450.0, 0.0), temperature=1.0e4)
        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=50.0)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestReturnsTrimmedWindowWhenOaIsRelevant(unittest.TestCase):
    """Tests for `trim_oa_azimuth_by_integrand`, well-aligned bulk shared-fixture branch."""

    def setUp(self):
        # Bulk along -X_RTN → azimuth_inst = atan2(450, 0) = +90°.
        self.rg = _make_response_grid()
        self.sw = proton_params(velocity_rtn=(-450.0, 0.0, 0.0))
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
    """Tests for `trim_oa_azimuth_by_integrand`, OA_SCAN_THRESHOLD endpoint selection."""

    def test_endpoints_expand_one_node_past_threshold_crossings(self):
        """A step-function transmission plateau drives the trim to return one scan node outside the first/last above-threshold node."""
        # Hand-built scenario with a transmission curve that's zero outside a
        # narrow azimuth band, so the scan integrand is a step function we can
        # reason about exactly without re-running production code.
        #
        # T(|az|) is 1 inside [85°, 95°] and 0 elsewhere. With a uniform-Maxwellian
        # bulk (scaling `M(φ)` by zero kills that whole node regardless), the
        # scan g(φ) on the 64-node grid linspace(20, 150, 64) is non-zero at
        # exactly the scan nodes whose nearest 1° transmission sample is in the
        # plateau. linspace(20, 150, 64) has spacing 130/63 ≈ 2.0635°, and the
        # nodes inside [85°, 95°] are indices 32, 33, 34 (≈86.03°, 88.10°,
        # 90.16°, 92.22°, 94.29° — but azimuth lookup quantizes to integer °, so
        # the plateau is exactly nodes whose floor(|az|) ∈ [85, 95)).
        transmission = np.zeros(181)
        transmission[85:95] = 1.0  # T = 1 on [85°, 94°]
        rg = _make_response_grid(azimuthal_transmission=transmission)
        sw = proton_params(velocity_rtn=(-450.0, 0.0, 0.0))  # bulk_az = 90°

        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)

        scan = np.linspace(20.0, 150.0, OA_SCAN_RESOLUTION)
        # Find which nodes lie in [85°, 94°] — those are the "above-threshold"
        # nodes for this transmission. The trim must return one node outside
        # the first/last of those indices.
        in_plateau = (scan >= 85.0) & (scan < 95.0)
        plateau_idx = np.where(in_plateau)[0]
        self.assertGreater(plateau_idx.size, 0)
        expected_lo = scan[max(int(plateau_idx[0]) - 1, 0)]
        expected_hi = scan[min(int(plateau_idx[-1]) + 1, OA_SCAN_RESOLUTION - 1)]
        self.assertAlmostEqual(lo, expected_lo)
        self.assertAlmostEqual(hi, expected_hi)


class TestSymmetryBetweenOaPositiveAndNegative(unittest.TestCase):
    """Tests for `trim_oa_azimuth_by_integrand`, OA+/OA- mirror-symmetry contract."""

    def test_mirror_symmetric_bulk_yields_mirror_symmetric_windows(self):
        """Bulks at ±90° produce OA+ and OA- windows that are exact mirror images of each other."""
        rg = _make_response_grid()

        # Bulk along -X_RTN → azimuth_inst = +90° (deep in OA+).
        sw_pos = proton_params(velocity_rtn=(-450.0, 0.0, 0.0))
        # Bulk along +X_RTN → azimuth_inst = -90° (deep in OA-).
        sw_neg = proton_params(velocity_rtn=(+450.0, 0.0, 0.0))

        lo_pos, hi_pos = _trim(rg, sw_pos, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)
        # OA-: same magnitudes, mirrored around 0.
        lo_neg, hi_neg = _trim(rg, sw_neg, azimuth_lo=-150.0, azimuth_hi=-20.0, sg_rate=0.0)

        # The implementation picks scan-grid nodes by index, and
        # `linspace(20, 150, N)` mirrors `linspace(-150, -20, N)` up to a
        # single ULP from `linspace`'s internal accumulation order.
        np.testing.assert_allclose(lo_neg, -hi_pos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(hi_neg, -lo_pos, atol=1e-12, rtol=0)


class TestSkipPolicyAgainstSgRate(unittest.TestCase):
    """Tests for `trim_oa_azimuth_by_integrand`, sg_rate-driven skip transition."""

    def test_high_sg_rate_forces_skip_for_otherwise_aligned_bulk(self):
        """A bulk that yields a real window at sg_rate=0 flips to the (0, 0) sentinel once sg_rate is cranked above the OA upper bound."""
        rg = _make_response_grid()
        sw = proton_params(velocity_rtn=(-450.0, 0.0, 0.0))

        lo_open, hi_open = _trim(
            rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0
        )
        self.assertGreater(hi_open, lo_open)

        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=1.0e15)
        self.assertEqual((lo, hi), (0.0, 0.0))


class TestScanAtBulkElevationClampedToOaRange(unittest.TestCase):
    """Tests for `trim_oa_azimuth_by_integrand`, bulk-elevation clamp to `[min_elevation, max_elevation]`."""

    def test_bulk_elevation_above_range_yields_non_empty_window(self):
        """A bulk at +20° elevation (above the +10° OA cap) still yields a non-empty trimmed window after the scan elevation is clamped to +10°."""
        rg = _make_response_grid()
        sw = _proton_params_at_elevation(speed=450.0, bulk_el_deg=20.0)

        lo, hi = _trim(rg, sw, azimuth_lo=20.0, azimuth_hi=150.0, sg_rate=0.0)
        self.assertGreater(hi, lo)


if __name__ == "__main__":
    unittest.main()
