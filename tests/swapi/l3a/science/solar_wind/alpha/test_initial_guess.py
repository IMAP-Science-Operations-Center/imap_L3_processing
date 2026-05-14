import unittest

import numpy as np

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.initial_guess import (
    esa_voltage_to_alpha_speed,
    get_alpha_peak_indices,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR


def _analytic_speed_km_per_s(
    voltage: float, mass_kg: float, charge_c: float
) -> float:
    return float(
        np.sqrt(2 * SWAPI_K_FACTOR * charge_c * abs(voltage) / mass_kg)
        / METERS_PER_KILOMETER
    )


class TestEsaVoltageToAlphaSpeed(unittest.TestCase):
    """Tests for `esa_voltage_to_alpha_speed`."""

    def test_matches_analytical_formula(self):
        """A sweep of representative ESA voltages reproduces the closed-form `sqrt(2 K q V / m)` alpha speed to machine precision."""
        for V in [200.0, 1000.0, 4000.0]:
            with self.subTest(voltage=V):
                np.testing.assert_allclose(
                    esa_voltage_to_alpha_speed(V),
                    _analytic_speed_km_per_s(
                        V, ALPHA_PARTICLE_MASS_KG, ALPHA_PARTICLE_CHARGE_COULOMBS
                    ),
                    rtol=1e-12,
                )

    def test_handles_negative_voltage(self):
        """A negative ESA voltage yields the same alpha speed as its positive counterpart, since the conversion depends on magnitude."""
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(-1000.0), esa_voltage_to_alpha_speed(1000.0)
        )


class TestGetAlphaPeakIndices(unittest.TestCase):
    """Tests for `get_alpha_peak_indices`; SWAPI energies decrease with index so the alpha bump sits at *lower* indices than the proton peak, and the returned slice indexes into that ascending-index axis."""

    PROTON_PEAK_INDEX = 50
    ALPHA_PEAK_INDEX = 30

    def _decreasing_energy_grid_with_alpha_bump(self) -> tuple[np.ndarray, np.ndarray]:
        """SWAPI-style decreasing energy grid plus a Gaussian alpha bump
        centered at `ALPHA_PEAK_INDEX`."""
        energies = np.linspace(20_000.0, 50.0, 72)
        n_alpha = 200.0 * np.exp(
            -((np.arange(72) - self.ALPHA_PEAK_INDEX) ** 2) / 9.0
        )
        return energies, n_alpha

    def test_returns_slice_containing_alpha_peak(self):
        """Given a clean Gaussian alpha bump above the proton peak, the returned slice brackets the bump's center index."""
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        peak_slice = get_alpha_peak_indices(
            residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
        )
        self.assertLessEqual(peak_slice.start, self.ALPHA_PEAK_INDEX)
        self.assertGreater(peak_slice.stop, self.ALPHA_PEAK_INDEX)

    def test_returned_slice_is_fixed_width_of_five_around_argmax(self):
        """The slice spans exactly 5 bins, with 3 above the argmax in energy (lower index) and 1 below (higher index)."""
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        peak_slice = get_alpha_peak_indices(
            residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
        )
        self.assertEqual(peak_slice.stop - peak_slice.start, 5)
        self.assertEqual(peak_slice.start, self.ALPHA_PEAK_INDEX - 3)
        self.assertEqual(peak_slice.stop, self.ALPHA_PEAK_INDEX + 2)

    def test_slice_energies_stay_inside_one_and_a_half_to_four_times_proton_energy(self):
        """For a clean Gaussian bump well inside the search window, every bin in the returned slice has an energy between 1.5x and 4x the proton-peak energy."""
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        peak_slice = get_alpha_peak_indices(
            residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
        )
        proton_energy = energies[self.PROTON_PEAK_INDEX]
        slice_energies = energies[peak_slice]
        self.assertTrue(np.all(slice_energies >= 1.5 * proton_energy))
        self.assertTrue(np.all(slice_energies <= 4.0 * proton_energy))

    def test_raises_when_proton_peak_index_is_zero(self):
        """With `proton_peak_index=0` there are no bins above the proton peak to scan for the 1.5x-energy lower edge, and the function raises before any peak search runs."""
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        with self.assertRaises(Exception) as ctx:
            get_alpha_peak_indices(residuals, energies, proton_peak_index=0)
        self.assertIn("Proton peak too high to find alpha peak", str(ctx.exception))

    def test_raises_when_residuals_are_monotonic_in_high_energy_region(self):
        """Residuals that increase monotonically with energy never satisfy `residuals[i] > residuals[i+1]` above the proton peak, so the function raises instead of locating a fake bump."""
        energies = np.linspace(20_000.0, 50.0, 72)
        monotonic_residuals = np.linspace(1.0, 100.0, 72)
        with self.assertRaises(Exception) as ctx:
            get_alpha_peak_indices(
                monotonic_residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
            )
        self.assertIn("Alpha peak not found", str(ctx.exception))

    def test_rejects_ascending_energy_grid(self):
        """An L1B-style ascending energy axis violates the SWAPI calling convention and is rejected at the top of the function with an AssertionError."""
        ascending_energies = np.linspace(50.0, 20_000.0, 72)
        residuals = np.zeros(72)
        with self.assertRaises(AssertionError):
            get_alpha_peak_indices(
                residuals, ascending_energies, proton_peak_index=self.PROTON_PEAK_INDEX
            )

    def test_raises_when_alpha_peak_lands_within_three_bins_of_high_energy_edge(self):
        """When the alpha argmax is at index < 3, the 5-bin window cannot fit without a negative start index, so the function raises rather than returning a wrap-around slice."""
        energies = np.linspace(20_000.0, 50.0, 72)
        bump_at_index_one = 200.0 * np.exp(-((np.arange(72) - 1) ** 2) / 8.0)
        with self.assertRaises(Exception) as ctx:
            get_alpha_peak_indices(
                bump_at_index_one, energies, proton_peak_index=self.PROTON_PEAK_INDEX
            )
        self.assertIn("Alpha peak too close to high-energy edge", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
