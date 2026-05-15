"""Post-fit fill-value branches of `calculate_pickup_ion_values`.

End-to-end optimizer/Hessian/coincidence-rate behavior is exercised by the MC
parameter-recovery test in `test_monte_carlo_fit_pickup_ion.py`. These tests
mock those three seams so each assertion isolates one of the post-fit guards
(`BAD_FIT` short-circuit, background > 1 Hz fill)."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

_MODULE_PATH = (
    "imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values"
)
_N_SWEEPS = 5
_N_COARSE_BINS = 62
_SW_VELOCITY_RTN_KMS = np.array([400.0, 0.0, 0.0])
_VOLTAGE_PER_STEP = np.linspace(100.0, 8000.0, _N_COARSE_BINS)


def _spice_state_passing_every_bin():
    """Energy cutoffs that admit every coarse bin and a Vasyliunas–Siscoe
    distribution with a finite distance so `min_speed_kms` calculation runs."""
    distribution = MagicMock(spec=VasyliunasSiscoeDistribution)
    distribution.distance_km = ONE_AU_IN_KM
    return 0.0, 1.0e9, distribution


def _density_lookup_table():
    table = MagicMock()
    table.get_minimum_distance.return_value = 1.0
    return table


def _good_nominal(**overrides):
    base = {
        "cooling_index": 1.5,
        "ionization_rate": 1e-7,
        "cutoff_speed": 450.0,
        "background_count_rate": 0.1,
    }
    base.update(overrides)
    return base


def _run_calculate_with_mocked_fit(
    *,
    nominal,
    observed_per_step,
    modeled_per_step,
    cov_external_diag=(1.0, 1.0, 1.0, 1.0),
):
    """Drive `calculate_pickup_ion_values` through the post-fit branches.

    `observed_per_step` and `modeled_per_step` are tiled across `_N_SWEEPS`
    sweeps; differences between them set the residual sum of squares that
    feeds the R² guard. `cov_external_diag` becomes the diagonal of the
    mocked external-coordinate covariance — passing negative entries yields
    NaN σ̂ and exercises the non-positive-definite Hessian branch."""
    voltages = np.tile(_VOLTAGE_PER_STEP, (_N_SWEEPS, 1))
    observed_per_step = np.broadcast_to(
        np.asarray(observed_per_step, dtype=float), (_N_COARSE_BINS,)
    )
    modeled_per_step = np.broadcast_to(
        np.asarray(modeled_per_step, dtype=float), (_N_COARSE_BINS,)
    )
    count_rates = np.tile(observed_per_step, (_N_SWEEPS, 1))
    modeled_rates = np.tile(modeled_per_step, (_N_SWEEPS, 1))
    bulk_sw_per_bin_swapi_kms = np.tile(
        _SW_VELOCITY_RTN_KMS, (_N_SWEEPS, _N_COARSE_BINS, 1)
    )

    fake_result = MagicMock()
    fake_result.var_names = [
        "cooling_index",
        "ionization_rate",
        "cutoff_speed",
        "background_count_rate",
    ]
    fake_result.x = np.zeros(4)
    fake_result.params.valuesdict.return_value = nominal
    fake_minimizer = MagicMock()
    fake_minimizer.minimize.return_value = fake_result
    fake_minimizer._int2ext_cov_x.return_value = np.diag(cov_external_diag)

    with patch(f"{_MODULE_PATH}.build_chunk_collapsed_response") as mock_build, patch(
        f"{_MODULE_PATH}.lmfit.Minimizer", return_value=fake_minimizer
    ), patch(
        f"{_MODULE_PATH}.ndt.Hessian", return_value=lambda _: np.eye(4)
    ), patch(
        f"{_MODULE_PATH}.calculate_coincidence_rate", return_value=modeled_rates
    ):
        mock_build.return_value = MagicMock()
        lower_energy_cutoff, upper_energy_cutoff, vasyliunas_siscoe_distribution = (
            _spice_state_passing_every_bin()
        )
        return calculate_pickup_ion_values(
            swapi_response=MagicMock(),
            voltages=voltages,
            count_rates=count_rates,
            sw_velocity_rtn_kms=_SW_VELOCITY_RTN_KMS,
            bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
            density_of_neutral_helium_lookup_table=_density_lookup_table(),
            lower_energy_cutoff=lower_energy_cutoff,
            upper_energy_cutoff=upper_energy_cutoff,
            vasyliunas_siscoe_distribution=vasyliunas_siscoe_distribution,
        )


def _assert_all_nan_params(tc, fitting_params):
    for value in (
        fitting_params.cooling_index,
        fitting_params.ionization_rate,
        fitting_params.cutoff_speed,
        fitting_params.background_count_rate,
    ):
        tc.assertTrue(np.isnan(value.nominal_value))
        tc.assertTrue(np.isnan(value.std_dev))


class CalculatePickupIonValuesFillTest(unittest.TestCase):
    def test_zero_variance_observations_fill_all_params_with_bad_fit(self):
        """When every observed count rate is identical the total sum of
        squares is zero and R² is undefined; `BAD_FIT` is set and every
        parameter is reported as NaN ± NaN."""
        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=5.0,
            modeled_per_step=5.0,
        )

        self.assertEqual(int(result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan_params(self, result.fitting_params)

    def test_low_r_squared_fills_all_params_with_bad_fit(self):
        """When the model misses non-constant observations badly enough that
        R² < 0.9, `BAD_FIT` is set and every parameter is reported as NaN ±
        NaN — values are not retained."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)
        modeled_per_step = np.zeros(_N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=observed_per_step,
            modeled_per_step=modeled_per_step,
        )

        self.assertEqual(int(result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan_params(self, result.fitting_params)

    def test_non_positive_definite_hessian_fills_all_params_with_bad_fit(self):
        """A non-positive-definite Hessian gives a covariance with negative
        diagonal entries; `np.sqrt(np.diag(cov))` then yields NaN σ̂. The
        guard sets `BAD_FIT` and every parameter is NaN ± NaN."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
            cov_external_diag=(-1.0, -1.0, -1.0, -1.0),
        )

        self.assertEqual(int(result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT))
        _assert_all_nan_params(self, result.fitting_params)

    def test_background_above_one_hz_fills_background_only(self):
        """When the fitted background exceeds 1 Hz the flat term is absorbing
        real signal; the background is reported as NaN ± NaN, the other three
        parameters are unchanged, and the fit flag stays NONE."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=1.5),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
        )
        fitting_params = result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        self.assertTrue(np.isnan(fitting_params.background_count_rate.nominal_value))
        self.assertTrue(np.isnan(fitting_params.background_count_rate.std_dev))
        for value in (
            fitting_params.cooling_index,
            fitting_params.ionization_rate,
            fitting_params.cutoff_speed,
        ):
            self.assertTrue(np.isfinite(value.nominal_value))
            self.assertTrue(np.isfinite(value.std_dev))

    def test_background_at_one_hz_is_not_filled(self):
        """The background guard uses a strict inequality (`> 1.0`); a fit
        sitting exactly at 1 Hz is retained."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=1.0),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
        )
        fitting_params = result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        self.assertEqual(fitting_params.background_count_rate.nominal_value, 1.0)
        self.assertTrue(np.isfinite(fitting_params.background_count_rate.std_dev))

    def test_clean_fit_returns_all_finite_params_with_no_flag(self):
        """A perfect fit (R² = 1) with a background ≤ 1 Hz returns all four
        parameters with finite nominal and σ̂ and the fit flag is NONE — the
        baseline against which the fill-value branches above are deviations."""
        observed_per_step = np.linspace(1.0, 10.0, _N_COARSE_BINS)

        result = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            observed_per_step=observed_per_step,
            modeled_per_step=observed_per_step,
        )
        fitting_params = result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        for value in (
            fitting_params.cooling_index,
            fitting_params.ionization_rate,
            fitting_params.cutoff_speed,
            fitting_params.background_count_rate,
        ):
            self.assertTrue(np.isfinite(value.nominal_value))
            self.assertTrue(np.isfinite(value.std_dev))


if __name__ == "__main__":
    unittest.main()
