import unittest

import numpy as np
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_CHARGE_OVER_MASS_C_PER_KG,
    ALPHA_PARTICLE_MASS_KG,
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    EV_TO_KELVIN,
    PROTON_CHARGE_OVER_MASS_C_PER_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_alpha_solar_wind_moments import (
    AlphaSolarWindMoments,
    fit_solar_wind_alpha_moments,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.uncertainties import (
    make_correlated_velocity,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.response.speed_calculation import (
    SWAPI_COARSE_SWEEP_BINS,
    esa_voltage_to_alpha_speed,
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path


_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


def _swapi_response():
    return SwapiResponse.from_files(
        _AZIMUTHAL_TRANSMISSION_PATH,
        _CENTRAL_EFFECTIVE_AREA_PATH,
        _PASSBAND_FIT_COEFFICIENTS_PATH,
    )


class TestSpeciesSpeedConversions(unittest.TestCase):
    def test_alpha_speed_at_known_voltage(self):
        # v_0^α(529 V) = sqrt(2*1.89*2*e/m_α*529)/1000 ≈ 310.5 km/s; computed analytically.
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(529.0), 310.533, rtol=2e-3
        )

    def test_alpha_speed_uses_absolute_voltage(self):
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(-1000.0), esa_voltage_to_alpha_speed(1000.0)
        )

    def test_alpha_speed_close_to_proton_speed_at_half_voltage(self):
        # If alpha mass were exactly 4 m_p, v_0^α(V) = v_0^p(V/2). Real m_α/m_p ≈ 3.97,
        # giving a ~0.4% deviation — verify within 1% tolerance.
        for V in [200.0, 1000.0, 4000.0]:
            np.testing.assert_allclose(
                esa_voltage_to_alpha_speed(V),
                esa_voltage_to_proton_speed(V / 2.0),
                rtol=1e-2,
            )


class TestPassbandGridSpecies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = _swapi_response()

    def test_alpha_central_speed_close_to_proton_over_sqrt2_at_same_voltage(self):
        # Exactly 1/√2 if m_α = 4 m_p; real m_α ≈ 3.97 m_p gives ~0.4% deviation.
        V = 1000.0
        p_cs = self.sr.central_speed(V, PROTON_MASS_PER_CHARGE_M_P_PER_E)
        a_cs = self.sr.central_speed(V, ALPHA_MASS_PER_CHARGE_M_P_PER_E)
        np.testing.assert_allclose(a_cs, p_cs / np.sqrt(2.0), rtol=1e-2)

    def test_grid_cache_keys_on_voltage_only(self):
        # Grid is V-only; same voltage always returns the exact same cached object.
        V = 800.0
        self.sr.warm_cache([V])
        g1 = self.sr.create_passband_grid(V)
        g2 = self.sr.create_passband_grid(V)
        self.assertIs(g1, g2)
        # Species-specific quantities come from sr.central_speed(), not the grid.
        p_cs = self.sr.central_speed(V, PROTON_MASS_PER_CHARGE_M_P_PER_E)
        a_cs = self.sr.central_speed(V, ALPHA_MASS_PER_CHARGE_M_P_PER_E)
        self.assertNotAlmostEqual(p_cs, a_cs, places=0)


_N_SWEEPS = 5
_N_BINS = 62


def _make_proton_moments(**kw):
    """Construct ProtonSolarWindFitResult from nominal test values."""
    velocity_covariance = kw.pop("velocity_covariance", None)
    if not hasattr(kw["density"], "nominal_value"):
        kw["density"] = ufloat(float(kw["density"]), np.nan)
    if not hasattr(kw["temperature"], "nominal_value"):
        kw["temperature"] = ufloat(float(kw["temperature"]), np.nan)
    if not hasattr(kw["bulk_velocity_rtn"][0], "nominal_value"):
        if velocity_covariance is None:
            kw["bulk_velocity_rtn"] = tuple(
                ufloat(float(v), np.nan) for v in kw["bulk_velocity_rtn"]
            )
        else:
            kw["bulk_velocity_rtn"] = make_correlated_velocity(
                np.asarray(kw["bulk_velocity_rtn"], dtype=float),
                np.asarray(velocity_covariance, dtype=float),
            )
    return ProtonSolarWindFitResult(**kw)


class TestFlagsAndGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = _swapi_response()
        cls.esa_flat = np.tile(np.geomspace(60.0, 5000.0, _N_BINS)[::-1], _N_SWEEPS)
        cls.sr.warm_cache(cls.esa_flat)

    def _bogus_count_rate(self):
        return np.zeros_like(self.esa_flat, dtype=float)

    def test_stale_proton_returns_stale_flag(self):
        proton = _make_proton_moments(
            density=5.0,
            temperature=10.0,
            bulk_velocity_rtn=np.array([450.0, 0.0, 0.0]),
            bad_fit_flag=int(SwapiL3Flags.FIT_FAILED),
        )
        result = fit_solar_wind_alpha_moments(
            count_rate=self._bogus_count_rate(),
            esa_voltage=self.esa_flat,
            measurement_time=np.zeros(len(self.esa_flat), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=proton,
            magnetic_field_direction=np.array([1.0, 0.0, 0.0]),
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
        )
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.STALE_PROTON))
        self.assertTrue(np.isnan(result.density.nominal_value))
        self.assertTrue(np.all(np.isnan(result.bulk_velocity_rtn_nominal())))

    def test_invalid_mag_short_circuits_before_proton_velocity_guard(self):
        """Invalid MAG returns FIT_FAILED immediately without inspecting proton velocity."""
        proton = _make_proton_moments(
            density=5.0,
            temperature=10.0,
            bulk_velocity_rtn=np.array(
                [0.0, 0.0, 0.0]
            ),  # would trip FIT_FAILED downstream
            bad_fit_flag=int(SwapiL3Flags.NONE),
        )
        result = fit_solar_wind_alpha_moments(
            count_rate=self._bogus_count_rate(),
            esa_voltage=self.esa_flat,
            measurement_time=np.zeros(len(self.esa_flat), dtype="int64"),
            swapi_response=self.sr,
            proton_moments=proton,
            magnetic_field_direction=np.full(3, np.nan),
            alpha_effective_area_scale=1.0,
            proton_effective_area_scale=1.0,
            rotation_matrices=np.tile(np.eye(3), (len(self.esa_flat), 1, 1)),
        )
        self.assertEqual(result.bad_fit_flag, int(SwapiL3Flags.FIT_FAILED))
        self.assertTrue(np.isnan(result.delta_v.nominal_value))


# -------------------------------------------------------------------------
# Regression tests on real L2 spectra
# -------------------------------------------------------------------------

_FIXTURE_PATH = get_test_data_path("swapi/alpha_fit_test_spectra.npz")


def _load_fixture(name: str) -> dict:
    """Load a named fixture from the .npz file."""
    data = np.load(_FIXTURE_PATH)
    prefix = f"{name}__"
    return {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}


if __name__ == "__main__":
    unittest.main()
