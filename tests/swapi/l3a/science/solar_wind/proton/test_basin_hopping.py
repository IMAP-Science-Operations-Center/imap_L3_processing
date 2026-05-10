"""Tests for `solar_wind.proton.basin_hopping.escape_local_minimum`."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    OptimizeSolarWindParamsResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton import basin_hopping
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.basin_hopping import (
    _MAX_BASIN_REFINE_ITERS,
    _ROTATED_RMSE_RATIO_THRESHOLD,
    _flip_vector_about_axis,
    _flipped_seed,
    escape_local_minimum,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams


# Local proton-state fixture. Mirrors `tests/swapi/l3a/science/solar_wind/test_state.py`'s
# `_proton_params` (same defaults); duplicated here to keep this test file self-contained.
def _proton_params(
    density: float = 5.0,
    velocity_rtn=(-450.0, 0.0, 0.0),
    temperature: float = 1.0e5,
) -> SolarWindParams:
    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=np.asarray(velocity_rtn, dtype=float),
        temperature=temperature,
        mass=PROTON_MASS_KG,
    )


def _single_sweep_ctx(
    spin_axis_column=np.array([0.0, 1.0, 0.0]),
) -> SolarWindFitContext:
    """Single-sweep context whose +Y_inst column equals `spin_axis_column`,
    so `average_spin_axis_rtn(ctx.rotation_matrices)` returns that axis."""
    rotation = np.eye(3)
    rotation[:, 1] = spin_axis_column
    return SolarWindFitContext(
        count_rate=np.zeros(1),
        esa_voltage=np.array([1000.0]),
        response_grids=[None],
        rotation_matrices=rotation[np.newaxis, :, :],
        mass_kg=PROTON_MASS_KG,
    )


def _result(sw: SolarWindParams, mse: float) -> OptimizeSolarWindParamsResult:
    """Build an `OptimizeSolarWindParamsResult` mock whose `.mse` returns `mse`
    and whose `.sw_params` returns `sw`. Uses `Mock(spec=...)` so the test does
    not depend on how `.mse` is computed internally."""
    mock = Mock(spec=OptimizeSolarWindParamsResult)
    mock.sw_params = sw
    mock.mse = mse
    return mock


class TestFlipVectorAboutAxis(unittest.TestCase):
    """`_flip_vector_about_axis(v, axis)` is the Householder reflection
    `2 axis (axis·v) − v` — i.e. the 180° rotation of `v` about `axis`. The
    component of `v` parallel to `axis` is preserved; the perpendicular
    component is inverted."""

    def test_perpendicular_component_is_inverted(self):
        # axis = +Y, v = +X (purely perpendicular) → flip = -X.
        v = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            _flip_vector_about_axis(v, axis), [-1.0, 0.0, 0.0]
        )

    def test_parallel_component_is_preserved(self):
        # v = +Y (purely parallel to axis) → 180° rotation is a no-op.
        v = np.array([0.0, 3.0, 0.0])
        axis = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(_flip_vector_about_axis(v, axis), v)

    def test_mixed_vector_inverts_only_perpendicular_part(self):
        # v = (1, 2, -3), axis = +Y.
        # Parallel part: (0, 2, 0) — preserved.
        # Perpendicular part: (1, 0, -3) — inverted to (-1, 0, 3).
        # Sum: (-1, 2, 3).
        v = np.array([1.0, 2.0, -3.0])
        axis = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            _flip_vector_about_axis(v, axis), [-1.0, 2.0, 3.0]
        )

    def test_flip_is_its_own_inverse(self):
        v = np.array([-450.0, 7.0, -3.0])
        axis = np.array([0.0, 1.0, 0.0])
        once = _flip_vector_about_axis(v, axis)
        twice = _flip_vector_about_axis(once, axis)
        np.testing.assert_allclose(twice, v)

    def test_flip_preserves_magnitude(self):
        # Householder reflection is an isometry: |flip(v)| == |v|.
        v = np.array([-450.0, 50.0, -20.0])
        axis = np.array([0.1, 0.99, 0.0])
        axis = axis / np.linalg.norm(axis)
        np.testing.assert_allclose(
            np.linalg.norm(_flip_vector_about_axis(v, axis)),
            np.linalg.norm(v),
        )


class TestEscapeLocalMinimum(unittest.TestCase):
    """Tests for the post-LM-1 wrong-basin detector. Mocks `_flipped_seed`
    and `optimize_solar_wind_params` so the cheap-gate / acceptance / cap
    logic is exercised in isolation from the forward model and optimizer."""

    # The cheap-gate condition is `flipped_mse >= threshold² × current.mse`;
    # squaring threshold compares MSE against an RMSE ratio.
    _GATE_FACTOR = _ROTATED_RMSE_RATIO_THRESHOLD**2

    # ---- Cheap-gate early exit: no LM-2 call when flipped seed is too bad ----

    def test_returns_lm1_unchanged_when_flipped_seed_far_worse(self):
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=1.0)
        flipped_mse_far_above_gate = lm1.mse * self._GATE_FACTOR * 10.0
        flipped_params = _proton_params(velocity_rtn=(450.0, 0.0, 0.0))

        with patch.object(
            basin_hopping,
            "_flipped_seed",
            return_value=(flipped_mse_far_above_gate, flipped_params),
        ) as mock_flipped, patch.object(
            basin_hopping, "optimize_solar_wind_params"
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, lm1)
        # `_flipped_seed` ran exactly once (loop bailed after the gate check).
        self.assertEqual(mock_flipped.call_count, 1)
        # The optimizer was never re-run on the flipped seed.
        mock_opt.assert_not_called()

    def test_returns_lm1_unchanged_when_flipped_seed_exactly_at_gate(self):
        # Gate is `>=`: equality also triggers the early exit.
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=1.0)
        flipped_mse_at_gate = lm1.mse * self._GATE_FACTOR

        with patch.object(
            basin_hopping,
            "_flipped_seed",
            return_value=(flipped_mse_at_gate, _proton_params()),
        ), patch.object(
            basin_hopping, "optimize_solar_wind_params"
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, lm1)
        mock_opt.assert_not_called()

    # The acceptance condition is `flipped_result.mse > current.mse` → reject;
    # so equality is *accepted* (i.e. not strictly less). We don't pin this case
    # because the production code path treats an exactly-equal LM-2 fit as a
    # no-op replacement (same numerical state) — the next iteration will
    # re-flip from the same basin and either improve or bail at the gate.

    # ---- LM-2 accepted when it lands at strictly lower MSE ----

    def test_returns_lm2_when_lm2_mse_is_lower(self):
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=10.0)
        better_params = _proton_params(velocity_rtn=(450.0, 0.0, 0.0))
        better_lm2 = _result(better_params, mse=1.0)

        # Iteration 1: flipped seed clears the gate, LM-2 lands lower → accepted.
        # Iteration 2: simulate "next flipped seed is far worse than the new
        # current" so the loop terminates cleanly via the cheap gate.
        flipped_seed_mse_iter1 = lm1.mse * (self._GATE_FACTOR / 2.0)
        flipped_seed_mse_iter2 = better_lm2.mse * (self._GATE_FACTOR * 2.0)

        with patch.object(
            basin_hopping,
            "_flipped_seed",
            side_effect=[
                (flipped_seed_mse_iter1, better_params),
                (flipped_seed_mse_iter2, better_params),
            ],
        ) as mock_flipped, patch.object(
            basin_hopping,
            "optimize_solar_wind_params",
            return_value=better_lm2,
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, better_lm2)
        # LM-2 ran exactly once (iteration 2 exited via the cheap gate).
        self.assertEqual(mock_opt.call_count, 1)
        # Two iterations of the loop ran; iter 2 short-circuited at the gate.
        self.assertEqual(mock_flipped.call_count, 2)

    # ---- LM-2 rejected when it converges to higher MSE ----

    def test_returns_lm1_when_lm2_mse_is_higher(self):
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=1.0)
        worse_params = _proton_params(velocity_rtn=(450.0, 0.0, 0.0))
        worse_lm2 = _result(worse_params, mse=5.0)

        # Flipped seed clears the cheap gate (within threshold²×current).
        flipped_seed_mse = lm1.mse * (self._GATE_FACTOR / 2.0)
        with patch.object(
            basin_hopping,
            "_flipped_seed",
            return_value=(flipped_seed_mse, worse_params),
        ), patch.object(
            basin_hopping,
            "optimize_solar_wind_params",
            return_value=worse_lm2,
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, lm1)
        self.assertEqual(mock_opt.call_count, 1)

    # ---- Iteration cap: at most _MAX_BASIN_REFINE_ITERS LM-2 calls ----

    def test_runs_at_most_max_basin_refine_iters_lm2_calls(self):
        ctx = _single_sweep_ctx()
        # Strictly-decreasing MSE chain so every iteration accepts the LM-2
        # result. If the loop weren't bounded, this would never terminate.
        # Use `1.0 / (i+1)` so MSE stays strictly positive for any chain length.
        chain_length = _MAX_BASIN_REFINE_ITERS + 4
        results = [
            _result(
                _proton_params(velocity_rtn=(-450.0 + i, 0.0, 0.0)),
                mse=1.0 / (i + 1),
            )
            for i in range(chain_length)
        ]
        lm1 = results[0]
        gate_factor = self._GATE_FACTOR
        with patch.object(
            basin_hopping,
            "_flipped_seed",
            # Always within gate (factor / 2 < gate_factor), always with a fresh seed.
            side_effect=[
                (results[i].mse * (gate_factor / 2.0), results[i].sw_params)
                for i in range(chain_length - 1)
            ],
        ), patch.object(
            basin_hopping,
            "optimize_solar_wind_params",
            side_effect=results[1:],
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        # Pin the cap from the source constant — never hardcode the number.
        self.assertEqual(mock_opt.call_count, _MAX_BASIN_REFINE_ITERS)
        # Returns the LM-2 result accepted in the final allowed iteration.
        self.assertIs(out, results[_MAX_BASIN_REFINE_ITERS])


class TestFlippedSeedDensityFallback(unittest.TestCase):
    """`_flipped_seed` builds a candidate `SolarWindParams` at the spin-axis-
    flipped velocity and rescales its density via `optimal_density_scale`. If
    that scale comes back non-positive or non-finite (e.g. the flipped
    velocity points the bulk away from the aperture, so unit-density forward-
    model rates are all zero and the curve-fit collapses), passing it through
    to LM as `n=0` would silently corrupt the next basin-hop attempt. The
    helper instead falls back to LM-1's converged density so the basin-check
    MSE is still evaluated on a sane parameter set."""

    def _ctx_and_lm1(self):
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(density=7.5), mse=1.0)
        return ctx, lm1

    def test_returns_lm1_density_when_optimal_scale_is_zero(self):
        # Mock the forward model + the density rescale so no real numerical
        # path is required. `optimal_density_scale=0.0` simulates a flipped-
        # velocity geometry that produces all-zero unit-density rates.
        ctx, lm1 = self._ctx_and_lm1()
        with patch.object(
            basin_hopping,
            "model_solar_wind_ideal_coincidence_rates",
            return_value=(np.zeros_like(ctx.count_rate), None),
        ), patch.object(
            basin_hopping,
            "optimal_density_scale",
            return_value=0.0,
        ):
            _, flipped_params = _flipped_seed(
                lm1, ctx, np.array([0.0, 1.0, 0.0])
            )

        self.assertEqual(flipped_params.density, lm1.sw_params.density)

    def test_returns_lm1_density_when_optimal_scale_is_negative(self):
        # `<= 0` covers both zero (no rates anywhere) and negative (curve_fit
        # converged onto a negative density to minimize residuals).
        ctx, lm1 = self._ctx_and_lm1()
        with patch.object(
            basin_hopping,
            "model_solar_wind_ideal_coincidence_rates",
            return_value=(np.zeros_like(ctx.count_rate), None),
        ), patch.object(
            basin_hopping,
            "optimal_density_scale",
            return_value=-2.0,
        ):
            _, flipped_params = _flipped_seed(
                lm1, ctx, np.array([0.0, 1.0, 0.0])
            )

        self.assertEqual(flipped_params.density, lm1.sw_params.density)

    def test_returns_lm1_density_when_optimal_scale_is_nan(self):
        # `optimal_density_scale` can return NaN when curve_fit fails to
        # converge — the fallback also catches non-finite results.
        ctx, lm1 = self._ctx_and_lm1()
        with patch.object(
            basin_hopping,
            "model_solar_wind_ideal_coincidence_rates",
            return_value=(np.zeros_like(ctx.count_rate), None),
        ), patch.object(
            basin_hopping,
            "optimal_density_scale",
            return_value=float("nan"),
        ):
            _, flipped_params = _flipped_seed(
                lm1, ctx, np.array([0.0, 1.0, 0.0])
            )

        self.assertEqual(flipped_params.density, lm1.sw_params.density)


if __name__ == "__main__":
    unittest.main()
