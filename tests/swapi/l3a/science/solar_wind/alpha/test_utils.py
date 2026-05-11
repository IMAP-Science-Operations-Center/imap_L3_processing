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
    def test_matches_analytical_formula(self):
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
        np.testing.assert_allclose(
            esa_voltage_to_alpha_speed(-1000.0), esa_voltage_to_alpha_speed(1000.0)
        )


class TestGetAlphaPeakIndices(unittest.TestCase):
    """SWAPI energies decrease with index, so "low index" = "high energy".
    `get_alpha_peak_indices` returns a slice into the ascending-index axis
    that contains the alpha bump (which sits at higher energies than the
    proton peak — i.e., at *lower* indices)."""

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
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        peak_slice = get_alpha_peak_indices(
            residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
        )
        self.assertLessEqual(peak_slice.start, self.ALPHA_PEAK_INDEX)
        self.assertGreater(peak_slice.stop, self.ALPHA_PEAK_INDEX)

    def test_slice_low_index_edge_sits_at_4x_proton_energy(self):
        # The high-energy edge of the alpha window is the first index whose
        # energy is ≤ 4× the proton peak's energy. Energies decrease with
        # index, so this edge is `slice.start` (the lowest index in the slice).
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        peak_slice = get_alpha_peak_indices(
            residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
        )
        max_alpha_energy = 4.0 * energies[self.PROTON_PEAK_INDEX]

        self.assertLessEqual(energies[peak_slice.start], max_alpha_energy)
        if peak_slice.start > 0:
            # The neighbor one index lower (one cell *higher* in energy) must
            # be above the cap — i.e. start sits exactly on the boundary.
            self.assertGreater(energies[peak_slice.start - 1], max_alpha_energy)

    def test_raises_when_proton_peak_at_high_energy_edge(self):
        # When the proton peak sits at index 0, there are no higher-energy bins
        # at all — the alpha bump can't exist anywhere. The function refuses
        # rather than returning a degenerate slice.
        energies, residuals = self._decreasing_energy_grid_with_alpha_bump()
        with self.assertRaises(Exception) as ctx:
            get_alpha_peak_indices(residuals, energies, proton_peak_index=0)
        self.assertIn("Alpha peak not found", str(ctx.exception))

    def test_raises_when_residuals_are_monotonic_in_high_energy_region(self):
        # Without a local maximum at higher energies than the proton peak,
        # there is no alpha bump to locate — the function refuses.
        energies = np.linspace(20_000.0, 50.0, 72)
        monotonic_residuals = np.linspace(1.0, 100.0, 72)
        with self.assertRaises(Exception) as ctx:
            get_alpha_peak_indices(
                monotonic_residuals, energies, proton_peak_index=self.PROTON_PEAK_INDEX
            )
        self.assertIn("Alpha peak not found", str(ctx.exception))

    def test_rejects_ascending_energy_grid(self):
        # SWAPI ESA energies are strictly decreasing with index. Passing an
        # L1B-style ascending grid is a calling-convention error, caught at
        # the top of the function rather than producing a wrong result.
        ascending_energies = np.linspace(50.0, 20_000.0, 72)
        residuals = np.zeros(72)
        with self.assertRaises(AssertionError):
            get_alpha_peak_indices(
                residuals, ascending_energies, proton_peak_index=self.PROTON_PEAK_INDEX
            )


if __name__ == "__main__":
    unittest.main()
