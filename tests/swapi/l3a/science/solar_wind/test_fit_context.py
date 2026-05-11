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
    """Mock SwapiResponse whose `get_response_grid(v, ...)` returns a real
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

    response.get_response_grid.side_effect = _build_response_grid
    return response


class TestSolarWindFitContextSubset(unittest.TestCase):
    """Tests for `SolarWindFitContext.subset` — selects per-sweep arrays by index while leaving scalar `mass_kg` intact."""

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
        """Subsetting with the full index range reproduces every per-sweep array unchanged."""
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
        """Selecting indices [1, 3] yields a context whose count_rate, voltage, response_grids, and rotation_matrices are all picked at those positions."""
        kept = self.full_ctx.subset(np.array([1, 3]))
        np.testing.assert_array_equal(kept.count_rate, [20.0, 40.0])
        np.testing.assert_array_equal(kept.esa_voltage, [200.0, 400.0])
        self.assertEqual(list(kept.response_grids), ["grid_1", "grid_3"])
        np.testing.assert_array_equal(
            kept.rotation_matrices,
            self.full_ctx.rotation_matrices[[1, 3]],
        )

    def test_subset_preserves_mass(self):
        """Subsetting to a single sweep leaves the scalar `mass_kg` field untouched."""
        kept = self.full_ctx.subset(np.array([0]))
        self.assertEqual(kept.mass_kg, PROTON_MASS_KG)


class TestBuildSolarWindFitContext(unittest.TestCase):
    """Tests for `build_solar_wind_fit_context` — filters invalid sweeps from per-sweep arrays in lockstep and materializes a per-sweep `ResponseGrid`."""

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
        """When every voltage is finite and positive, no sweeps are dropped and one ResponseGrid is built per sweep."""
        count_rate = np.array([10.0, 20.0, 30.0])
        esa_voltage = np.array([100.0, 200.0, 300.0])
        ctx, response = self._call_factory(
            count_rate=count_rate, esa_voltage=esa_voltage
        )
        np.testing.assert_array_equal(ctx.count_rate, count_rate)
        np.testing.assert_array_equal(ctx.esa_voltage, esa_voltage)
        self.assertEqual(len(ctx.response_grids), 3)

    def test_drops_sweeps_with_zero_or_negative_voltage(self):
        """Sweeps with voltage == 0 or voltage < 0 are filtered out of count_rate and esa_voltage together."""
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0, 40.0]),
            esa_voltage=np.array([100.0, 0.0, -50.0, 400.0]),
        )
        np.testing.assert_array_equal(ctx.count_rate, [10.0, 40.0])
        np.testing.assert_array_equal(ctx.esa_voltage, [100.0, 400.0])

    def test_drops_sweeps_with_non_finite_voltage(self):
        """NaN and inf voltages are treated as invalid and dropped alongside their count_rate entries."""
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0]),
            esa_voltage=np.array([100.0, np.nan, np.inf]),
        )
        np.testing.assert_array_equal(ctx.count_rate, [10.0])
        np.testing.assert_array_equal(ctx.esa_voltage, [100.0])

    def test_filters_rotation_matrices_in_lockstep_with_voltages(self):
        """When a middle sweep is dropped for a zero voltage, its rotation matrix is dropped at the same index so downstream geometry stays aligned."""
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
        """The factory calls `get_response_grid` once per kept voltage and stores the resulting grids in the same order as the input voltages."""
        count_rate = np.array([10.0, 20.0])
        esa_voltage = np.array([100.0, 200.0])
        ctx, response = self._call_factory(
            count_rate=count_rate, esa_voltage=esa_voltage
        )
        self.assertEqual(response.get_response_grid.call_count, 2)
        self.assertEqual(ctx.response_grids[0].central_speed, 100.0)
        self.assertEqual(ctx.response_grids[1].central_speed, 200.0)

    def test_filters_response_grids_in_lockstep_with_voltages(self):
        """When a sweep with a NaN voltage is dropped, no ResponseGrid is built for it and the kept grids stay aligned with the kept voltages."""
        ctx, _ = self._call_factory(
            count_rate=np.array([10.0, 20.0, 30.0]),
            esa_voltage=np.array([100.0, np.nan, 300.0]),
        )
        self.assertEqual(len(ctx.response_grids), 2)
        self.assertEqual(ctx.response_grids[0].central_speed, 100.0)
        self.assertEqual(ctx.response_grids[1].central_speed, 300.0)

    def test_passes_species_and_efficiency_args_to_create_response_grid(self):
        """The mass-per-charge and central-effective-area scale supplied to the factory are forwarded into the `get_response_grid` call for each sweep."""
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
        self.assertEqual(response.get_response_grid.call_count, 1)
        call = response.get_response_grid.call_args
        all_args = list(call.args) + list(call.kwargs.values())
        self.assertIn(100.0, all_args)
        self.assertIn(2.0, all_args)
        self.assertIn(0.42, all_args)


if __name__ == "__main__":
    unittest.main()
