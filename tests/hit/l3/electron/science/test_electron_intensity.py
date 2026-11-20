import unittest

import numpy as np

from imap_l3_processing.hit.l3.electron.science.electron_intensity import (
    calculate_electron_intensities,
)
from imap_l3_processing.hit.l3.electron.science.electron_intensity_factors import (
    ElectronIntensityFactors,
)


class TestElectronIntensity(unittest.TestCase):
    def test_converts_a_and_b_side_rates_to_intensities_with_frame_divisor(self) -> None:
        factors = ElectronIntensityFactors(
            a_side_rate_indices=np.arange(6),
            b_side_rate_indices=np.arange(10, 16),
            delta_e_mev=np.full(6, 0.5),
            geometry_factor_cm2_sr=np.full(6, 2.0),
            background=np.zeros(6),
        )
        rate_index = np.arange(20)
        rates = np.tile(60 * (rate_index + 1), (2, 1)).astype(float)
        rates_delta_plus = np.tile(6 * (rate_index + 1), (2, 1)).astype(float)
        rates_delta_minus = np.tile(3 * (rate_index + 1), (2, 1)).astype(float)
        rates[1, 12] = np.nan
        rates_delta_plus[1, 12] = np.nan
        rates_delta_minus[1, 12] = np.nan

        result = calculate_electron_intensities(
            rates, rates_delta_plus, rates_delta_minus, factors
        )

        expected_a_side = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 1))
        expected_a_side_delta_plus = np.tile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (2, 1))
        expected_a_side_delta_minus = np.tile([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], (2, 1))
        expected_b_side = np.tile([11.0, 12.0, 13.0, 14.0, 15.0, 16.0], (2, 1))
        expected_b_side[1, 2] = np.nan
        expected_b_side_delta_plus = np.tile([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], (2, 1))
        expected_b_side_delta_plus[1, 2] = np.nan
        expected_b_side_delta_minus = np.tile([0.55, 0.60, 0.65, 0.70, 0.75, 0.80], (2, 1))
        expected_b_side_delta_minus[1, 2] = np.nan

        np.testing.assert_allclose(result.a_side, expected_a_side)
        np.testing.assert_allclose(result.a_side_delta_plus, expected_a_side_delta_plus)
        np.testing.assert_allclose(result.a_side_delta_minus, expected_a_side_delta_minus)
        np.testing.assert_allclose(result.b_side, expected_b_side, equal_nan=True)
        np.testing.assert_allclose(
            result.b_side_delta_plus, expected_b_side_delta_plus, equal_nan=True
        )
        np.testing.assert_allclose(
            result.b_side_delta_minus, expected_b_side_delta_minus, equal_nan=True
        )
        np.testing.assert_array_equal(np.argwhere(np.isnan(result.b_side)), [[1, 2]])
        self.assertFalse(np.isnan(result.a_side).any())

    def test_applies_each_bins_background_width_and_geometry_factor(self) -> None:
        delta_e_mev = np.array([0.3, 0.1, 0.4, 0.4, 0.8, 88.0])
        geometry_factor_cm2_sr = np.array(
            [1.7792, 0.39942, 0.20431, 0.011787, 0.0013381, 0.000034886]
        )
        background = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        factors = ElectronIntensityFactors(
            a_side_rate_indices=np.arange(6),
            b_side_rate_indices=np.arange(10, 16),
            delta_e_mev=delta_e_mev,
            geometry_factor_cm2_sr=geometry_factor_cm2_sr,
            background=background,
        )
        counts = np.full((1, 20), 600.0)

        result = calculate_electron_intensities(counts, counts, counts, factors)

        per_bin_divisor = 60 * delta_e_mev * geometry_factor_cm2_sr
        expected_intensity = ((600.0 - background) / per_bin_divisor).reshape(1, 6)
        expected_uncertainty = (600.0 / per_bin_divisor).reshape(1, 6)
        np.testing.assert_allclose(result.a_side, expected_intensity)
        np.testing.assert_allclose(result.b_side, expected_intensity)
        np.testing.assert_allclose(result.a_side_delta_plus, expected_uncertainty)
        np.testing.assert_allclose(result.a_side_delta_minus, expected_uncertainty)
        np.testing.assert_allclose(result.b_side_delta_plus, expected_uncertainty)
        np.testing.assert_allclose(result.b_side_delta_minus, expected_uncertainty)

    def test_rejects_rates_without_twenty_ialirt_entries(self) -> None:
        factors = ElectronIntensityFactors(
            a_side_rate_indices=np.arange(6),
            b_side_rate_indices=np.arange(10, 16),
            delta_e_mev=np.full(6, 0.5),
            geometry_factor_cm2_sr=np.full(6, 2.0),
            background=np.zeros(6),
        )
        rates = np.ones((2, 19))

        with self.assertRaises(ValueError) as context:
            calculate_electron_intensities(rates, rates, rates, factors)

        self.assertIn("20 I-ALiRT rates", str(context.exception))
