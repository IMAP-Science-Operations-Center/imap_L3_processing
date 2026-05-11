import unittest

import numpy as np

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.utils import (
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

    def test_slice_low_index_edge_sits_at_4x_proton_energy(self):
        """The slice's low-index (high-energy) edge is anchored to the first bin at or below 4x the proton-peak energy, with the bin one index lower exceeding that cap."""
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        peak_slice = get_alpha_peak_indices(
            residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
        )
        max_alpha_energy = 4.0 * energies[self.PROTON_PEAK_INDEX]

        self.assertLessEqual(energies[peak_slice.start], max_alpha_energy)
        if peak_slice.start > 0:
            self.assertGreater(energies[peak_slice.start - 1], max_alpha_energy)

    def test_raises_when_proton_peak_at_high_energy_edge(self):
        """When the proton peak is at index 0 there are no higher-energy bins for the alpha bump to occupy, and the function raises rather than returning a degenerate slice."""
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        with self.assertRaises(Exception) as ctx:
            get_alpha_peak_indices(residuals, energies, proton_peak_index=0)
        self.assertIn("Alpha peak not found", str(ctx.exception))

    def test_raises_when_residuals_are_monotonic_in_high_energy_region(self):
        """Residuals that increase monotonically with energy have no local maximum above the proton peak, so the function raises instead of locating a fake bump."""
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


if __name__ == "__main__":
    unittest.main()
