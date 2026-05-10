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
    """The optimizer reads/writes a flat 5-element state vector. The index
    constants pin the layout and must match LM expectations."""

    def test_layout_is_log_density_log_temperature_then_velocity(self):
        # Pin the actual mapping so a reader can see the layout from the test
        # alone: indices 0–1 hold the log-encoded scalars, indices 2–4 hold
        # the velocity components.
        self.assertEqual(LOG_DENSITY_IDX, 0)
        self.assertEqual(LOG_TEMPERATURE_IDX, 1)
        self.assertEqual(VELOCITY_SLICE, slice(2, 5))
        self.assertEqual(N_STATE, 5)


class TestSolarWindParamsToVector(unittest.TestCase):
    def test_density_and_temperature_are_log_encoded(self):
        sw = _proton_params(density=5.0, temperature=100_000.0)
        state = sw.to_vector()
        self.assertAlmostEqual(state[LOG_DENSITY_IDX], math.log(5.0))
        self.assertAlmostEqual(state[LOG_TEMPERATURE_IDX], math.log(100_000.0))

    def test_velocity_is_stored_unchanged(self):
        velocity = np.array([-450.0, 10.0, -5.0])
        sw = _proton_params(velocity_rtn=tuple(velocity))
        state = sw.to_vector()
        np.testing.assert_array_equal(state[VELOCITY_SLICE], velocity)

    def test_vector_has_n_state_length(self):
        self.assertEqual(_proton_params().to_vector().shape, (N_STATE,))


class TestSolarWindParamsFromVector(unittest.TestCase):
    def test_from_vector_inverts_to_vector(self):
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
        # `mass` is not encoded in the state vector — the caller must supply it
        # because it's species-specific and held constant during the LM fit.
        state = np.array([0.0, 0.0, -400.0, 0.0, 0.0])
        sw_proton = SolarWindParams.from_vector(state, mass=PROTON_MASS_KG)
        self.assertEqual(sw_proton.mass, PROTON_MASS_KG)


class TestBulkSpeed(unittest.TestCase):
    def test_returns_magnitude_of_bulk_velocity(self):
        sw = _proton_params(velocity_rtn=(-300.0, 40.0, -50.0))
        expected = math.sqrt(300.0**2 + 40.0**2 + 50.0**2)
        self.assertAlmostEqual(bulk_speed(sw), expected)

    def test_zero_velocity_returns_zero(self):
        sw = _proton_params(velocity_rtn=(0.0, 0.0, 0.0))
        self.assertEqual(bulk_speed(sw), 0.0)


class TestBulkAnglesInInstrumentFrame(unittest.TestCase):
    """`bulk_angles_in_instrument_frame(sw, rotation_xyz_to_rtn)` returns the
    bulk-velocity angles in the *instrument* (XYZ) frame:
        azimuth_deg   = atan2(-v_x_inst, -v_y_inst)
        elevation_deg = asin(-v_z_inst / |v|)
    The rotation matrix supplied has columns equal to the instrument-frame
    basis vectors expressed in RTN, so `rotation_xyz_to_rtn.T @ v_rtn` is
    `v` in the instrument frame."""

    def test_velocity_along_minus_y_inst_returns_zero_angles(self):
        # Identity rotation makes the instrument frame coincide with RTN.
        # Velocity along -Y_RTN ⇒ v_inst = (0, -450, 0):
        # azimuth = atan2(0, +450) = 0; elevation = asin(0/450) = 0.
        sw = _proton_params(velocity_rtn=(0.0, -450.0, 0.0))
        azimuth, elevation = bulk_angles_in_instrument_frame(sw, np.eye(3))
        self.assertAlmostEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, 0.0)

    def test_velocity_with_negative_x_inst_component_yields_positive_azimuth(self):
        # v_inst_x = -50 ⇒ azimuth = atan2(+50, +450) > 0.
        sw = _proton_params(velocity_rtn=(-50.0, -450.0, 0.0))
        azimuth, _ = bulk_angles_in_instrument_frame(sw, np.eye(3))
        self.assertGreater(azimuth, 0.0)

    def test_velocity_with_positive_z_inst_component_yields_negative_elevation(self):
        # v_inst_z = +50 ⇒ elevation = asin(-50/|v|) < 0.
        sw = _proton_params(velocity_rtn=(0.0, -450.0, 50.0))
        _, elevation = bulk_angles_in_instrument_frame(sw, np.eye(3))
        self.assertLess(elevation, 0.0)

    def test_uses_rotation_argument_to_transform_into_instrument_frame(self):
        # The columns of `rotation_xyz_to_rtn` are the instrument basis
        # vectors expressed in RTN. Pick a rotation whose +Y_inst column is
        # -X_RTN — then a wind moving along -X_RTN becomes a wind moving
        # along +Y_inst, giving v_inst_y > 0.
        # azimuth = atan2(-v_inst_x, -v_inst_y) = atan2(0, -1) = π → ±180°.
        rotation_xyz_to_rtn = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        sw = _proton_params(velocity_rtn=(-450.0, 0.0, 0.0))
        azimuth, _ = bulk_angles_in_instrument_frame(sw, rotation_xyz_to_rtn)
        # Result should be ±180°, not 0° (which is what an identity-rotation
        # baseline would give). Catches a regression that drops the rotation.
        self.assertAlmostEqual(abs(azimuth), 180.0, places=10)


class TestThermalSpeedAndTemperature(unittest.TestCase):
    """`thermal_speed` is the Maxwellian σ = √(kT/m) in km/s. Round-trip the
    speed → temperature inverse to make sure both directions agree."""

    def test_thermal_speed_matches_analytic_formula(self):
        sw = _proton_params(temperature=100_000.0)
        expected_km_s = (
            math.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * 100_000.0 / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        self.assertAlmostEqual(thermal_speed(sw), expected_km_s)

    def test_temperature_to_thermal_speed_round_trips_via_inverse(self):
        for temperature_k in [1_000.0, 1e5, 1e7]:
            with self.subTest(temperature=temperature_k):
                speed = temperature_to_thermal_speed(PROTON_MASS_KG, temperature_k)
                recovered = thermal_speed_to_temperature(speed, PROTON_MASS_KG)
                self.assertAlmostEqual(recovered, temperature_k)

    def test_thermal_speed_scales_with_sqrt_of_temperature(self):
        # σ ∝ √T at fixed mass — quadrupling T doubles σ.
        speed_low = temperature_to_thermal_speed(PROTON_MASS_KG, 50_000.0)
        speed_high = temperature_to_thermal_speed(PROTON_MASS_KG, 200_000.0)
        np.testing.assert_allclose(speed_high / speed_low, 2.0)

    def test_thermal_speed_uses_params_temperature_and_mass(self):
        # `thermal_speed(SolarWindParams)` is just a wrapper that pulls T and m
        # off the params and calls `temperature_to_thermal_speed`.
        sw = _proton_params(temperature=120_000.0)
        np.testing.assert_allclose(
            thermal_speed(sw),
            temperature_to_thermal_speed(PROTON_MASS_KG, 120_000.0),
        )


if __name__ == "__main__":
    unittest.main()
