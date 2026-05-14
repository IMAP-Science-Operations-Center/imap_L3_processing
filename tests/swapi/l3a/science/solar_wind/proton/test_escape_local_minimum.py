import unittest
from unittest.mock import Mock, patch

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton import fit_model
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    _MAX_BASIN_REFINE_ITERS,
    _ROTATED_RMSE_RATIO_THRESHOLD,
    OptimizeSolarWindParamsResult,
    escape_local_minimum,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from tests.swapi._helpers import proton_params as _shared_proton_params


# Spin-axis-mirror tests assume the slow-wind bulk lies along -R̂ (the spin
# axis), so the file-local default puts the velocity along the RTN R-axis.
def _proton_params(
    density: float = 5.0,
    velocity_rtn=(-450.0, 0.0, 0.0),
    temperature: float = 1.0e5,
) -> SolarWindParams:
    return _shared_proton_params(
        density=density,
        velocity_rtn=velocity_rtn,
        temperature=temperature,
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


class TestEscapeLocalMinimum(unittest.TestCase):
    """Tests for `escape_local_minimum`; mocks `flipped_seed` and `optimize_solar_wind_params` so cheap-gate, acceptance, and iteration-cap logic are exercised in isolation from the forward model."""

    # The cheap-gate condition is `flipped_mse >= threshold² × current.mse`;
    # squaring threshold compares MSE against an RMSE ratio.
    _GATE_FACTOR = _ROTATED_RMSE_RATIO_THRESHOLD**2

    # ---- Cheap-gate early exit: no LM-2 call when flipped seed is too bad ----

    def test_returns_lm1_unchanged_whenflipped_seed_far_worse(self):
        """A flipped seed whose MSE is well past the threshold²×current gate bails out immediately, leaving LM-1 as the returned result without ever invoking the LM-2 optimizer."""
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=1.0)
        flipped_mse_far_above_gate = lm1.mse * self._GATE_FACTOR * 10.0
        flipped_params = _proton_params(velocity_rtn=(450.0, 0.0, 0.0))

        with patch.object(
            fit_model,
            "flipped_seed",
            return_value=(flipped_mse_far_above_gate, flipped_params),
        ) as mock_flipped, patch.object(
            fit_model, "optimize_solar_wind_params"
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, lm1)
        # `flipped_seed` ran exactly once (loop bailed after the gate check).
        self.assertEqual(mock_flipped.call_count, 1)
        # The optimizer was never re-run on the flipped seed.
        mock_opt.assert_not_called()

    def test_returns_lm1_unchanged_whenflipped_seed_exactly_at_gate(self):
        """A flipped seed whose MSE exactly equals the gate threshold also short-circuits, because the gate comparison is `>=` rather than strictly greater than."""
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=1.0)
        flipped_mse_at_gate = lm1.mse * self._GATE_FACTOR

        with patch.object(
            fit_model,
            "flipped_seed",
            return_value=(flipped_mse_at_gate, _proton_params()),
        ), patch.object(
            fit_model, "optimize_solar_wind_params"
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, lm1)
        mock_opt.assert_not_called()

    # ---- LM-2 accepted when it lands at strictly lower MSE ----

    def test_returns_lm2_when_lm2_mse_is_lower(self):
        """When the flipped seed clears the cheap gate and LM-2 converges to a strictly lower MSE than LM-1, the function adopts the LM-2 result and continues iterating until the next gate check bails out."""
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
            fit_model,
            "flipped_seed",
            side_effect=[
                (flipped_seed_mse_iter1, better_params),
                (flipped_seed_mse_iter2, better_params),
            ],
        ) as mock_flipped, patch.object(
            fit_model,
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
        """When LM-2 clears the cheap gate but converges to a higher MSE than LM-1, the function rejects the flipped basin and returns the original LM-1 result."""
        ctx = _single_sweep_ctx()
        lm1 = _result(_proton_params(), mse=1.0)
        worse_params = _proton_params(velocity_rtn=(450.0, 0.0, 0.0))
        worse_lm2 = _result(worse_params, mse=5.0)

        # Flipped seed clears the cheap gate (within threshold²×current).
        flipped_seed_mse = lm1.mse * (self._GATE_FACTOR / 2.0)
        with patch.object(
            fit_model,
            "flipped_seed",
            return_value=(flipped_seed_mse, worse_params),
        ), patch.object(
            fit_model,
            "optimize_solar_wind_params",
            return_value=worse_lm2,
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        self.assertIs(out, lm1)
        self.assertEqual(mock_opt.call_count, 1)

    # ---- Iteration cap: at most _MAX_BASIN_REFINE_ITERS LM-2 calls ----

    def test_runs_at_most_max_basin_refine_iters_lm2_calls(self):
        """Feeding a strictly-decreasing MSE chain that would otherwise accept indefinitely confirms the loop is hard-bounded at `_MAX_BASIN_REFINE_ITERS` LM-2 calls and returns the result accepted in the final allowed iteration."""
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
            fit_model,
            "flipped_seed",
            # Always within gate (factor / 2 < gate_factor), always with a fresh seed.
            side_effect=[
                (results[i].mse * (gate_factor / 2.0), results[i].sw_params)
                for i in range(chain_length - 1)
            ],
        ), patch.object(
            fit_model,
            "optimize_solar_wind_params",
            side_effect=results[1:],
        ) as mock_opt:
            out = escape_local_minimum(lm1, ctx)

        # Pin the cap from the source constant — never hardcode the number.
        self.assertEqual(mock_opt.call_count, _MAX_BASIN_REFINE_ITERS)
        # Returns the LM-2 result accepted in the final allowed iteration.
        self.assertIs(out, results[_MAX_BASIN_REFINE_ITERS])


if __name__ == "__main__":
    unittest.main()
