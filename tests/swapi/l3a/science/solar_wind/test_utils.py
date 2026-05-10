"""Direct tests for `solar_wind.utils`:

- `average_spin_axis_rtn` — averages the body +Y direction in RTN over a chunk
  of per-sweep rotation matrices and returns a unit vector.
- `count_rate_conversion_factor` — converts an unnormalized integrand sum into
  a count rate (Hz) using density, central effective area, and thermal speed.
"""

import math
import unittest

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    SolarWindParams,
    thermal_speed,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import (
    average_spin_axis_rtn,
    count_rate_conversion_factor,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
)
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid


class TestAverageSpinAxisRtn(unittest.TestCase):
    """SWAPI's spin axis is body +Y. Each rotation matrix's *column* index 1 is
    that axis expressed in RTN. The helper averages those direction vectors
    over the chunk and renormalizes to unit length."""

    def test_single_rotation_returns_its_y_axis(self):
        rotation = np.eye(3)
        # Identity matrix's column 1 is +Y_RTN = (0, 1, 0).
        np.testing.assert_array_equal(
            average_spin_axis_rtn(rotation[np.newaxis]), [0.0, 1.0, 0.0]
        )

    def test_constant_axis_across_sweeps_returns_that_axis(self):
        rotations = np.stack([np.eye(3)] * 5)
        np.testing.assert_array_equal(
            average_spin_axis_rtn(rotations), [0.0, 1.0, 0.0]
        )

    def test_averages_two_axes_at_pm_45_deg_to_a_unit_vector(self):
        # Body +Y aligned with +Y_RTN ± rotated 45° in the RT-plane:
        # the mean is along +Y, and the helper renormalizes it to length 1.
        cos45 = math.cos(math.radians(45))
        rotation_pos = np.eye(3).copy()
        rotation_pos[:, 1] = [cos45, cos45, 0.0]
        rotation_neg = np.eye(3).copy()
        rotation_neg[:, 1] = [-cos45, cos45, 0.0]

        result = average_spin_axis_rtn(np.stack([rotation_pos, rotation_neg]))
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0])

    def test_normalizes_average_when_input_columns_are_not_unit_length(self):
        # Synthetic non-rotation matrices whose column-1 vectors have varying
        # magnitudes — the helper must still produce a unit-length output
        # (production rotations are unit, but this guards the renormalization).
        rotations = np.stack(
            [
                np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([[1.0, 1.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            ]
        )
        result = average_spin_axis_rtn(rotations)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)


class TestCountRateConversionFactor(unittest.TestCase):
    """The forward-model integrand is computed in normalized units (density-1
    Maxwellian sampled in (deg², km/s)). `count_rate_conversion_factor` maps
    that integrand to a count rate by multiplying through:
      A_eff_central · n · (√(2π) σ)^(-3) · (π/180)² · 1e5  (km → cm)."""

    def _proton_params(
        self,
        density: float = 5.0,
        temperature: float = 100_000.0,
    ) -> SolarWindParams:
        return SolarWindParams(
            density=density,
            bulk_velocity_rtn=np.array([-450.0, 0.0, 0.0]),
            temperature=temperature,
            mass=PROTON_MASS_KG,
        )

    def _response_grid(
        self, central_effective_area: float = 0.5
    ) -> ResponseGrid:
        # Only `central_effective_area` is read by the function under test;
        # the other fields are filled with sentinels (NaN/empty) so a future
        # caller that starts using them will fail loudly.
        return ResponseGrid(
            sg_passband=None,
            oa_passband=None,
            central_speed=float("nan"),
            central_effective_area=central_effective_area,
            azimuthal_transmission=AzimuthalTransmissionGrid(
                values=np.array([]), spacing=float("nan")
            ),
        )

    def test_matches_closed_form_formula(self):
        sw = self._proton_params(density=5.0, temperature=100_000.0)
        rg = self._response_grid(central_effective_area=0.5)

        sigma = thermal_speed(sw)
        deg_to_rad_squared = (math.pi / 180.0) ** 2  # (deg → rad)² for az·el grid
        km_to_cm = 1e5  # bring the cm⁻³ density into the cm³/s rate
        expected = (
            rg.central_effective_area
            * sw.density
            * (math.sqrt(2 * math.pi) * sigma) ** -3
            * deg_to_rad_squared
            * km_to_cm
        )
        np.testing.assert_allclose(
            count_rate_conversion_factor(sw, rg), expected, rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
