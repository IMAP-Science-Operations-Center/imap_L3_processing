"""Tests for `solar_wind.fit_context` — the `SolarWindFitContext` NamedTuple
and its `build_solar_wind_fit_context` factory.

The factory drops invalid sweeps (zero/negative/non-finite ESA voltage) from
the count-rate, voltage, and rotation-matrix arrays in lockstep, then asks the
SwapiResponse to materialize a per-sweep ResponseGrid for each kept voltage."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from imap_l3_processing.constants import PROTON_MASS_KG
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    AzimuthalTransmissionGrid,
)
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid


def _swapi_response_returning_per_voltage_response_grid():
    """Mock SwapiResponse whose `create_response_grid(v, ...)` returns a real
    `ResponseGrid` NamedTuple tagged with the voltage in its `central_speed`
    field — lets tests assert on order and per-sweep identity without fighting
    numba's typed-list element-type inference (which rejects MagicMock)."""
    response = MagicMock()

    def _build_response_grid(voltage, *args, **kwargs):
        # Tag the voltage in `central_speed` so tests can recover which
        # voltage produced which grid. Other fields use shape-correct
        # placeholders that satisfy numba's type system.
        return ResponseGrid(
            sg_passband=None,
            oa_passband=None,
            central_speed=float(voltage),
            central_effective_area=0.0,
            azimuthal_transmission=AzimuthalTransmissionGrid(
                values=np.zeros(1), spacing=0.1
            ),
        )

    response.create_response_grid.side_effect = _build_response_grid
    return response


class TestSolarWindFitContextSubset(unittest.TestCase):
    """`subset` selects sweeps by index from every per-sweep array (count_rate,
    esa_voltage, response_grids, rotation_matrices) while leaving `mass_kg`
    intact. This is the operation chunk-fitters use to drop bad sweeps after
    initial QC."""

    def setUp(self):
        # 4 sweeps with distinguishable per-sweep values. The "rotation
        # matrices" here are not real rotations (det ≠ 1) — they're just
        # arrays distinguishable by their first entry, used to verify the
        # subset operation indexes correctly.
        self.full_ctx = SolarWindFitContext(
            count_rate=np.array([10.0, 20.0, 30.0, 40.0]),
            esa_voltage=np.array([100.0, 200.0, 300.0, 400.0]),
            response_grids=["grid_0", "grid_1", "grid_2", "grid_3"],
            rotation_matrices=np.stack(
                [np.eye(3) * (i + 1) for i in range(4)]
            ),
            mass_kg=PROTON_MASS_KG,
        )

    def test_subset_with_all_indices_returns_equivalent_context(self):
        kept = self.full_ctx.subset(np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(kept.count_rate, self.full_ctx.count_rate)
        np.testing.assert_array_equal(kept.esa_voltage, self.full_ctx.esa_voltage)
        # `subset` rebuilds response_grids as a numba.typed.List; cast to a
        # plain list for value-level comparison.
        self.assertEqual(list(kept.response_grids), self.full_ctx.response_grids)
        np.testing.assert_array_equal(
            kept.rotation_matrices, self.full_ctx.rotation_matrices
        )

    def test_subset_picks_per_sweep_arrays_at_the_given_indices(self):
        kept = self.full_ctx.subset(np.array([1, 3]))
        np.testing.assert_array_equal(kept.count_rate, [20.0, 40.0])
        np.testing.assert_array_equal(kept.esa_voltage, [200.0, 400.0])
        self.assertEqual(list(kept.response_grids), ["grid_1", "grid_3"])
        np.testing.assert_array_equal(
            kept.rotation_matrices,
            self.full_ctx.rotation_matrices[[1, 3]],
        )

    def test_subset_preserves_mass(self):
        kept = self.full_ctx.subset(np.array([0]))
        self.assertEqual(kept.mass_kg, PROTON_MASS_KG)


class TestBuildSolarWindFitContext(unittest.TestCase):
    """The factory is responsible for filtering invalid sweeps and asking the
    SwapiResponse for per-sweep `ResponseGrid` objects."""

    def _call_factory(self, *, count_rate, esa_voltage, rotation_matrices=None):
        response = _swapi_response_returning_per_voltage_response_grid()
        if rotation_matrices is None:
            rotation_matrices = np.stack([np.eye(3)] * len(count_rate))
        ctx = build_solar_wind_fit_context(
            count_rate=count_rate,
            esa_voltage=esa_voltage,
            swapi_response=response,
            central_effective_area_scale=1.0,
            rotation_matrices=rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=1.0,
        )
        return ctx, response

    def test_keeps_all_sweeps_when_voltages_are_finite_and_positive(self):
        count_rate = np.array([10.0, 20.0, 30.0])
        esa_voltage = np.array([100.0, 200.0, 300.0])
        ctx, response = self._call_factory(
            count_rate=count_rate, esa_voltage=esa_voltage
        )
        np.testing.assert_array_equal(ctx.count_rate, count_rate)
        np.testing.assert_array_equal(ctx.esa_voltage, esa_voltage)
        # One ResponseGrid per kept voltage.
        self.assertEqual(len(ctx.response_grids), 3)

    def test_drops_sweeps_with_zero_or_negative_voltage(self):
        # Zero and negative voltages signal "no science data" and must be
        # filtered out of every per-sweep array.
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0, 40.0]),
            esa_voltage=np.array([100.0, 0.0, -50.0, 400.0]),
        )
        np.testing.assert_array_equal(ctx.count_rate, [10.0, 40.0])
        np.testing.assert_array_equal(ctx.esa_voltage, [100.0, 400.0])

    def test_drops_sweeps_with_non_finite_voltage(self):
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0]),
            esa_voltage=np.array([100.0, np.nan, np.inf]),
        )
        np.testing.assert_array_equal(ctx.count_rate, [10.0])
        np.testing.assert_array_equal(ctx.esa_voltage, [100.0])

    def test_filters_rotation_matrices_in_lockstep_with_voltages(self):
        # Per-sweep rotation matrices must be dropped at the same indices as
        # the voltages — otherwise downstream geometry is misaligned.
        rotations = np.stack([np.eye(3) * (i + 1) for i in range(3)])
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0]),
            esa_voltage=np.array([100.0, 0.0, 300.0]),
            rotation_matrices=rotations,
        )
        np.testing.assert_array_equal(
            ctx.rotation_matrices, rotations[[0, 2]]
        )

    def test_creates_response_grid_for_each_kept_voltage(self):
        count_rate = np.array([10.0, 20.0])
        esa_voltage = np.array([100.0, 200.0])
        ctx, response = self._call_factory(
            count_rate=count_rate, esa_voltage=esa_voltage
        )
        self.assertEqual(response.create_response_grid.call_count, 2)
        # Each ResponseGrid is tagged with its voltage by the mock factory;
        # verify the order matches the per-sweep voltage order.
        self.assertEqual(ctx.response_grids[0].central_speed, 100.0)
        self.assertEqual(ctx.response_grids[1].central_speed, 200.0)

    def test_filters_response_grids_in_lockstep_with_voltages(self):
        # Companion to the rotation-lockstep test: invalid sweeps must drop
        # their corresponding ResponseGrid too, or downstream code reads
        # the wrong grid for each kept voltage.
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0]),
            esa_voltage=np.array([100.0, np.nan, 300.0]),
        )
        self.assertEqual(len(ctx.response_grids), 2)
        self.assertEqual(ctx.response_grids[0].central_speed, 100.0)
        self.assertEqual(ctx.response_grids[1].central_speed, 300.0)

    def test_passes_species_and_efficiency_args_to_create_response_grid(self):
        response = _swapi_response_returning_per_voltage_response_grid()
        build_solar_wind_fit_context(
            count_rate=np.array([10.0]),
            esa_voltage=np.array([100.0]),
            swapi_response=response,
            central_effective_area_scale=0.42,
            rotation_matrices=np.eye(3)[np.newaxis],
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=2.0,  # alpha-like mass-per-charge
        )
        # The factory may pass species and EA scale positionally or as kwargs;
        # match on `mock.call.args + kwargs` content rather than fixing one form.
        self.assertEqual(response.create_response_grid.call_count, 1)
        call = response.create_response_grid.call_args
        all_args = list(call.args) + list(call.kwargs.values())
        self.assertIn(100.0, all_args)
        self.assertIn(2.0, all_args)
        self.assertIn(0.42, all_args)


if __name__ == "__main__":
    unittest.main()
