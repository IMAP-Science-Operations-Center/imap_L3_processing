"""Direct tests for `solar_wind.state` — the `SolarWindParams` NamedTuple,
its vector encoding for the LM optimizer, and the bulk-speed / thermal-speed /
temperature conversions."""

import math
import unittest

import numpy as np

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    METERS_PER_KILOMETER,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    N_STATE,
    VELOCITY_SLICE,
    SolarWindParams,
    bulk_angles_in_instrument_frame,
    bulk_speed,
    temperature_to_thermal_speed,
    thermal_speed,
    thermal_speed_to_temperature,
)


def _proton_params(
    density: float = 5.0,
    velocity_rtn=(-450.0, 0.0, 0.0),
    temperature: float = 100_000.0,
) -> SolarWindParams:
    """Solar-wind proton state at typical inertial-RTN slow-wind values."""
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.array(velocity_rtn),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


class TestStateVectorLayout(unittest.TestCase):
    """Tests for the module-level state-vector index constants (`LOG_DENSITY_IDX`, `LOG_TEMPERATURE_IDX`, `VELOCITY_SLICE`, `N_STATE`) that pin the LM optimizer's flat 5-element layout."""

    def test_layout_is_log_density_log_temperature_then_velocity(self):
        """Indices 0 and 1 hold the log-encoded density and temperature, indices 2-4 hold the velocity components, and the total length is 5."""
        self.assertEqual(LOG_DENSITY_IDX, 0)
        self.assertEqual(LOG_TEMPERATURE_IDX, 1)
        self.assertEqual(VELOCITY_SLICE, slice(2, 5))
        self.assertEqual(N_STATE, 5)


class TestSolarWindParamsToVector(unittest.TestCase):
    """Tests for `SolarWindParams.to_vector`."""

    def test_density_and_temperature_are_log_encoded(self):
        """A params object with density 5 and temperature 1e5 produces a state whose density and temperature slots equal log(5) and log(1e5)."""
        sw = _proton_params(density=5.0, temperature=100_000.0)
        state = sw.to_vector()
        self.assertAlmostEqual(state[LOG_DENSITY_IDX], math.log(5.0))
        self.assertAlmostEqual(state[LOG_TEMPERATURE_IDX], math.log(100_000.0))

    def test_velocity_is_stored_unchanged(self):
        """The three velocity components are written into the velocity slice verbatim, without any rotation or rescaling."""
        velocity = np.array([-450.0, 10.0, -5.0])
        sw = _proton_params(velocity_rtn=tuple(velocity))
        state = sw.to_vector()
        np.testing.assert_array_equal(state[VELOCITY_SLICE], velocity)

    def test_vector_has_n_state_length(self):
        """The encoded vector has exactly `N_STATE` elements so it matches the optimizer's expected length."""
        self.assertEqual(_proton_params().to_vector().shape, (N_STATE,))


class TestSolarWindParamsFromVector(unittest.TestCase):
    """Tests for `SolarWindParams.from_vector`, the inverse of `to_vector`."""

    def test_from_vector_inverts_to_vector(self):
        """Encoding a params object to a vector and decoding it back recovers the original density, temperature, velocity, and mass."""
        original = _proton_params(
            density=3.7, velocity_rtn=(-420.0, 5.0, -2.0), temperature=85_000.0
        )
        round_tripped = SolarWindParams.from_vector(
            original.to_vector(), mass=original.mass
        )
        self.assertAlmostEqual(round_tripped.density, original.density)
        self.assertAlmostEqual(round_tripped.temperature, original.temperature)
        np.testing.assert_array_equal(
            round_tripped.bulk_velocity_rtn, original.bulk_velocity_rtn
        )
        self.assertEqual(round_tripped.mass, original.mass)

    def test_from_vector_carries_mass_through_unchanged(self):
        """Mass is not encoded in the state vector, so the caller-supplied species mass appears on the decoded params object unchanged."""
        state = np.array([0.0, 0.0, -400.0, 0.0, 0.0])
        sw_proton = SolarWindParams.from_vector(state, mass=PROTON_MASS_KG)
        self.assertEqual(sw_proton.mass, PROTON_MASS_KG)


class TestBulkSpeed(unittest.TestCase):
    """Tests for `bulk_speed`."""

    def test_returns_magnitude_of_bulk_velocity(self):
        """A bulk velocity of (-300, 40, -50) km/s returns the Euclidean norm sqrt(300^2 + 40^2 + 50^2)."""
        sw = _proton_params(velocity_rtn=(-300.0, 40.0, -50.0))
        expected = math.sqrt(300.0**2 + 40.0**2 + 50.0**2)
        self.assertAlmostEqual(bulk_speed(sw), expected)

    def test_zero_velocity_returns_zero(self):
        """A bulk velocity of (0, 0, 0) returns 0 exactly, with no divide-by-zero artefacts."""
        sw = _proton_params(velocity_rtn=(0.0, 0.0, 0.0))
        self.assertEqual(bulk_speed(sw), 0.0)


class TestBulkAnglesInInstrumentFrame(unittest.TestCase):
    """Tests for `bulk_angles_in_instrument_frame`, which returns (azimuth, elevation) in degrees in the instrument XYZ frame: azimuth = atan2(-v_x_inst, -v_y_inst), elevation = asin(-v_z_inst / |v|)."""

    def test_velocity_along_minus_y_inst_returns_zero_angles(self):
        """Under identity rotation, an RTN wind along -Y becomes v_inst = (0, -450, 0) and yields azimuth = 0 and elevation = 0."""
        sw = _proton_params(velocity_rtn=(0.0, -450.0, 0.0))
        azimuth, elevation = bulk_angles_in_instrument_frame(sw, np.eye(3))
        self.assertAlmostEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, 0.0)

    def test_velocity_with_negative_x_inst_component_yields_positive_azimuth(self):
        """A wind with negative instrument-X component (v_inst_x = -50) gives a strictly positive azimuth, since azimuth = atan2(+50, +450) > 0."""
        sw = _proton_params(velocity_rtn=(-50.0, -450.0, 0.0))
        azimuth, _ = bulk_angles_in_instrument_frame(sw, np.eye(3))
        self.assertGreater(azimuth, 0.0)

    def test_velocity_with_positive_z_inst_component_yields_negative_elevation(self):
        """A wind with positive instrument-Z component (v_inst_z = +50) gives a strictly negative elevation, since elevation = asin(-50/|v|) < 0."""
        sw = _proton_params(velocity_rtn=(0.0, -450.0, 50.0))
        _, elevation = bulk_angles_in_instrument_frame(sw, np.eye(3))
        self.assertLess(elevation, 0.0)

    def test_uses_rotation_argument_to_transform_into_instrument_frame(self):
        """When the rotation maps +Y_inst to -X_RTN, an RTN wind along -X_RTN lands along +Y_inst and produces azimuth = atan2(0, -1) = +/-180 degrees, catching regressions that drop the rotation."""
        rotation_xyz_to_rtn = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        sw = _proton_params(velocity_rtn=(-450.0, 0.0, 0.0))
        azimuth, _ = bulk_angles_in_instrument_frame(sw, rotation_xyz_to_rtn)
        self.assertAlmostEqual(abs(azimuth), 180.0, places=10)


class TestThermalSpeedAndTemperature(unittest.TestCase):
    """Tests for `thermal_speed`, `temperature_to_thermal_speed`, and `thermal_speed_to_temperature` — the Maxwellian sigma = sqrt(kT/m) conversions in km/s."""

    def test_thermal_speed_matches_analytic_formula(self):
        """For a proton at 1e5 K, `thermal_speed` returns the closed-form sqrt(k*T/m) value in km/s."""
        sw = _proton_params(temperature=100_000.0)
        expected_km_s = (
            math.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * 100_000.0 / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        self.assertAlmostEqual(thermal_speed(sw), expected_km_s)

    def test_temperature_to_thermal_speed_round_trips_via_inverse(self):
        """For temperatures spanning 1e3 to 1e7 K, converting to thermal speed and back via the inverse recovers the original temperature."""
        for temperature_k in [1_000.0, 1e5, 1e7]:
            with self.subTest(temperature=temperature_k):
                speed = temperature_to_thermal_speed(PROTON_MASS_KG, temperature_k)
                recovered = thermal_speed_to_temperature(speed, PROTON_MASS_KG)
                self.assertAlmostEqual(recovered, temperature_k)

    def test_thermal_speed_scales_with_sqrt_of_temperature(self):
        """At fixed mass, quadrupling the temperature (50 kK to 200 kK) doubles the thermal speed, confirming the sigma proportional to sqrt(T) scaling."""
        speed_low = temperature_to_thermal_speed(PROTON_MASS_KG, 50_000.0)
        speed_high = temperature_to_thermal_speed(PROTON_MASS_KG, 200_000.0)
        np.testing.assert_allclose(speed_high / speed_low, 2.0)

    def test_thermal_speed_uses_params_temperature_and_mass(self):
        """`thermal_speed(SolarWindParams)` is a wrapper that reads T and m from the params and matches a direct `temperature_to_thermal_speed` call with the same inputs."""
        sw = _proton_params(temperature=120_000.0)
        np.testing.assert_allclose(
            thermal_speed(sw),
            temperature_to_thermal_speed(PROTON_MASS_KG, 120_000.0),
        )


if __name__ == "__main__":
    unittest.main()
