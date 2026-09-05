import unittest
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import numpy as np

from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.constants import SWAPI_COARSE_SWEEP_BINS
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values import (
    PickupIonFitResult,
    _calculate_pickup_ion_fit_energy_range,
    calculate_pickup_ion_values,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi._helpers import NOMINAL_TEST_EPOCH_TT2000

_MODULE_PATH = (
    "imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_pickup_ion_values"
)
_N_SWEEPS = 50
_SWEEP_LEN = 72
_N_COARSE_BINS = 62
_BULK_SW_VELOCITY_SWAPI_KMS = np.array([400.0, 0.0, 0.0])
_ENERGY_PER_COARSE_STEP = np.linspace(100.0, 8000.0, _N_COARSE_BINS)

_ENERGY_RANGE_ADMITTING_EVERY_BIN = (0.0, 1.0e9)
_MODELED_RATES = np.full((_N_SWEEPS, _N_COARSE_BINS), 3.0)


def _full_sweep_from_coarse_steps(per_coarse_step):
    """Lay per-coarse-step values out on the (50, 72) sweep the L2 arrays use."""
    full_sweep = np.zeros((_N_SWEEPS, _SWEEP_LEN))
    full_sweep[:, SWAPI_COARSE_SWEEP_BINS] = per_coarse_step
    return full_sweep


def _vasyliunas_siscoe_distribution():
    """A Vasyliunas-Siscoe distribution with a finite distance so the
    `min_speed_kms` calculation runs."""
    distribution = MagicMock(spec=VasyliunasSiscoeDistribution)
    distribution.distance_km = ONE_AU_IN_KM
    return distribution


def _good_nominal(**overrides):
    base = {
        "cooling_index": 1.5,
        "ionization_rate": 1e-7,
        "cutoff_speed": 450.0,
        "background_count_rate": 0.1,
    }
    base.update(overrides)
    return base


class _MockedFitRun(NamedTuple):
    fit_result: PickupIonFitResult
    is_good_fit_mock: MagicMock
    calculate_coincidence_rate_mock: MagicMock


def _run_calculate_with_mocked_fit(
    *,
    nominal,
    observed_per_step=1.0,
    cov_external_diag=(1.0, 1.0, 1.0, 1.0),
    fit_is_good=True,
):
    """Drive `calculate_pickup_ion_values` through its post-fit branches.

    The response builder, optimizer, Hessian, coincidence-rate model and
    goodness-of-fit check are all mocked, so `nominal` sets what the fit
    reports and `fit_is_good` picks which post-fit branch is taken.
    `cov_external_diag` becomes the diagonal of the mocked external-coordinate
    covariance — passing negative entries yields NaN σ̂ and exercises the
    non-positive-definite Hessian branch."""
    esa_energies = _full_sweep_from_coarse_steps(_ENERGY_PER_COARSE_STEP)
    count_rates = _full_sweep_from_coarse_steps(
        np.broadcast_to(np.asarray(observed_per_step, dtype=float), (_N_COARSE_BINS,))
    )
    bulk_sw_per_bin_swapi_kms = np.tile(
        _BULK_SW_VELOCITY_SWAPI_KMS, (_N_SWEEPS, _N_COARSE_BINS, 1)
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
        f"{_MODULE_PATH}.calculate_coincidence_rate", return_value=_MODELED_RATES
    ) as mock_calculate_coincidence_rate, patch(
        f"{_MODULE_PATH}._calculate_pickup_ion_fit_energy_range",
        return_value=_ENERGY_RANGE_ADMITTING_EVERY_BIN,
    ), patch(
        f"{_MODULE_PATH}.is_good_fit", return_value=fit_is_good
    ) as mock_is_good_fit:
        mock_build.return_value = MagicMock()
        fit_result = calculate_pickup_ion_values(
            swapi_response=MagicMock(),
            esa_energies=esa_energies,
            count_rates=count_rates,
            bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
            vasyliunas_siscoe_distribution=_vasyliunas_siscoe_distribution(),
            time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
        )

    return _MockedFitRun(
        fit_result=fit_result,
        is_good_fit_mock=mock_is_good_fit,
        calculate_coincidence_rate_mock=mock_calculate_coincidence_rate,
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


class CalculatePickupIonValuesShapeTest(unittest.TestCase):
    """Tests for the input shape guards at the top of `calculate_pickup_ion_values`."""

    def test_wrong_input_shapes_are_rejected(self):
        """Each of the three science inputs must arrive on the sweep layout the
        rest of the function assumes, and a mis-shaped one raises rather than
        being silently reshaped."""
        full_sweep = np.zeros((_N_SWEEPS, _SWEEP_LEN))
        coarse_vectors = np.zeros((_N_SWEEPS, _N_COARSE_BINS, 3))
        cases = [
            (
                "esa_energies given on the coarse sweep instead of the full sweep",
                np.zeros((_N_SWEEPS, _N_COARSE_BINS)),
                full_sweep,
                coarse_vectors,
            ),
            (
                "count_rates given on the coarse sweep instead of the full sweep",
                full_sweep,
                np.zeros((_N_SWEEPS, _N_COARSE_BINS)),
                coarse_vectors,
            ),
            (
                "bulk velocities given on the full sweep instead of the coarse sweep",
                full_sweep,
                full_sweep,
                np.zeros((_N_SWEEPS, _SWEEP_LEN, 3)),
            ),
        ]

        for label, esa_energies, count_rates, bulk_sw_per_bin_swapi_kms in cases:
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    calculate_pickup_ion_values(
                        swapi_response=MagicMock(),
                        esa_energies=esa_energies,
                        count_rates=count_rates,
                        bulk_sw_per_bin_swapi_kms=bulk_sw_per_bin_swapi_kms,
                        vasyliunas_siscoe_distribution=_vasyliunas_siscoe_distribution(),
                        time_as_tt2000=NOMINAL_TEST_EPOCH_TT2000,
                    )


class CalculatePickupIonValuesFillTest(unittest.TestCase):
    """Tests for the post-fit guards in `calculate_pickup_ion_values`, with the
    response builder, optimizer, Hessian, coincidence-rate model and
    goodness-of-fit check mocked so each test isolates one guard."""

    def test_non_positive_definite_hessian_fills_all_params_with_bad_fit(self):
        """A non-positive-definite Hessian gives a covariance with negative
        diagonal entries, so σ̂ comes back NaN; `BAD_FIT` is set, every
        parameter is NaN ± NaN, and the goodness-of-fit check never runs."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            cov_external_diag=(-1.0, -1.0, -1.0, -1.0),
        )

        self.assertEqual(
            int(run.fit_result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT)
        )
        _assert_all_nan_params(self, run.fit_result.fitting_params)
        run.is_good_fit_mock.assert_not_called()

    def test_rejected_goodness_of_fit_fills_all_params_with_bad_fit(self):
        """When the goodness-of-fit check rejects a converged fit, `BAD_FIT` is
        set and every parameter is reported as NaN ± NaN rather than retained."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(),
            fit_is_good=False,
        )

        self.assertEqual(
            int(run.fit_result.fitting_params.flags), int(SwapiL3Flags.BAD_FIT)
        )
        _assert_all_nan_params(self, run.fit_result.fitting_params)

    def test_goodness_of_fit_judges_a_background_free_model_on_the_whole_coarse_sweep(
        self,
    ):
        """The goodness-of-fit check is handed model rates evaluated with the
        background zeroed out, the fitted background as a separate scalar, and
        observations across all 62 coarse steps rather than the fit window."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=0.4),
            observed_per_step=2.0,
        )

        modeled_params = run.calculate_coincidence_rate_mock.call_args.args[2]
        self.assertEqual(modeled_params.background_count_rate, 0.0)
        self.assertEqual(modeled_params.cooling_index, 1.5)
        self.assertEqual(modeled_params.cutoff_speed, 450.0)

        goodness_of_fit_args = run.is_good_fit_mock.call_args.kwargs
        np.testing.assert_array_equal(
            goodness_of_fit_args["model_rates"], _MODELED_RATES
        )
        self.assertEqual(goodness_of_fit_args["background_rate"], 0.4)
        self.assertEqual(goodness_of_fit_args["cutoff_speed_kms"], 450.0)
        np.testing.assert_allclose(
            goodness_of_fit_args["observed_rates"], np.full((_N_SWEEPS, _N_COARSE_BINS), 2.0)
        )
        np.testing.assert_allclose(
            goodness_of_fit_args["esa_energies"], _ENERGY_PER_COARSE_STEP
        )
        self.assertEqual(
            goodness_of_fit_args["min_fitting_energy"],
            _ENERGY_RANGE_ADMITTING_EVERY_BIN[0],
        )

    def test_background_above_one_hz_fills_background_only(self):
        """When the fitted background exceeds 1 Hz the flat term is absorbing
        real signal; the background is reported as NaN ± NaN, the other three
        parameters are unchanged, and the fit flag stays NONE."""
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=1.5),
        )
        fitting_params = run.fit_result.fitting_params

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
        run = _run_calculate_with_mocked_fit(
            nominal=_good_nominal(background_count_rate=1.0),
        )
        fitting_params = run.fit_result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        self.assertEqual(fitting_params.background_count_rate.nominal_value, 1.0)
        self.assertTrue(np.isfinite(fitting_params.background_count_rate.std_dev))

    def test_accepted_fit_returns_all_finite_params_with_no_flag(self):
        """A converged fit that passes the goodness-of-fit check with a
        background ≤ 1 Hz returns all four parameters with finite nominal and σ̂
        and the fit flag NONE — the baseline the fill branches deviate from."""
        run = _run_calculate_with_mocked_fit(nominal=_good_nominal())
        fitting_params = run.fit_result.fitting_params

        self.assertEqual(int(fitting_params.flags), int(SwapiL3Flags.NONE))
        for value in (
            fitting_params.cooling_index,
            fitting_params.ionization_rate,
            fitting_params.cutoff_speed,
            fitting_params.background_count_rate,
        ):
            self.assertTrue(np.isfinite(value.nominal_value))
            self.assertTrue(np.isfinite(value.std_dev))


class CalculatePickupIonFitEnergyRangeTest(unittest.TestCase):
    """Tests for `_calculate_pickup_ion_fit_energy_range`."""

    def test_window_is_fixed_multiples_of_the_peak_bin_energy(self):
        ascending_energies = [250.0, 500.0, 1000.0, 2000.0, 4000.0]
        descending_energies = [4000.0, 2000.0, 1000.0, 500.0, 250.0]
        lower_edge_for_1000_ev_peak = 5656.854249492381
        lower_edge_for_500_ev_peak = 2828.4271247461903

        cases = [
            (
                "ascending sweep peaking at 1000 eV",
                ascending_energies,
                [1.0, 5.0, 100.0, 5.0, 1.0],
                lower_edge_for_1000_ev_peak,
                16000.0,
            ),
            (
                "peak one bin lower halves both edges",
                ascending_energies,
                [1.0, 100.0, 5.0, 5.0, 1.0],
                lower_edge_for_500_ev_peak,
                8000.0,
            ),
            (
                "descending sweep, as SWAPI actually steps energy",
                descending_energies,
                [1.0, 5.0, 100.0, 5.0, 1.0],
                lower_edge_for_1000_ev_peak,
                16000.0,
            ),
            (
                "peak height does not shift the window",
                ascending_energies,
                [1000.0, 5000.0, 100000.0, 5000.0, 1000.0],
                lower_edge_for_1000_ev_peak,
                16000.0,
            ),
            (
                "tied maxima resolve to the lower step index",
                ascending_energies,
                [1.0, 100.0, 100.0, 1.0, 1.0],
                lower_edge_for_500_ev_peak,
                8000.0,
            ),
        ]

        for (
            label,
            energies,
            count_rates,
            expected_lower_edge,
            expected_upper_edge,
        ) in cases:
            with self.subTest(case=label):
                lower_edge, upper_edge = _calculate_pickup_ion_fit_energy_range(
                    np.array(energies), np.array(count_rates)
                )

                self.assertAlmostEqual(lower_edge, expected_lower_edge)
                self.assertAlmostEqual(upper_edge, expected_upper_edge)


if __name__ == "__main__":
    unittest.main()
