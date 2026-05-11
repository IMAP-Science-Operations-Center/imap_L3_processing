import os
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import numpy as np
import spacepy.pycdf
from spacepy.pycdf import CDF
from uncertainties import UFloat, ufloat

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.utils import (
    calculate_sw_speed,
    chunk_l2_data,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    read_l2_swapi_data,
    read_mag_rtn_data,
    rotate_rtn_to_dps,
)
from tests.spice_test_case import SpiceTestCase


class TestUtils(TestCase):
    def tearDown(self) -> None:
        if os.path.exists('temp_cdf.cdf'):
            os.remove('temp_cdf.cdf')

    def test_chunk_l2_data(self):
        epoch = np.array([0, 1, 2, 3])
        energy = np.array([[15000, 16000, 17000, 18000, 19000],
                           [25000, 26000, 27000, 28000, 29000],
                           [35000, 36000, 37000, 38000, 39000],
                           [45000, 46000, 47000, 48000, 49000], ]
                          )
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        data = SwapiL2Data(epoch, energy, coincidence_count_rate, coincidence_count_rate_uncertainty)
        chunks = list(chunk_l2_data(data, 2))

        expected_energy_chunk_1 = np.array([[15000, 16000, 17000, 18000, 19000],
                                            [25000, 26000, 27000, 28000, 29000]])
        expected_energy_chunk_2 = np.array([[35000, 36000, 37000, 38000, 39000],
                                            [45000, 46000, 47000, 48000, 49000]])

        expected_count_rate_chunk_1 = np.array([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
        expected_count_rate_uncertainty_chunk_1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        first_chunk = chunks[0]

        np.testing.assert_array_equal(first_chunk.sci_start_time, np.array([0, 1]))
        np.testing.assert_array_equal(expected_energy_chunk_1, first_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_1, first_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_1,
                                      first_chunk.coincidence_count_rate_uncertainty)

        expected_count_rate_chunk_2 = np.array([[14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        expected_count_rate_uncertainty_chunk_2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        second_chunk = chunks[1]
        np.testing.assert_array_equal(np.array([2, 3]), second_chunk.sci_start_time)
        np.testing.assert_array_equal(expected_energy_chunk_2, second_chunk.energy)
        np.testing.assert_array_equal(expected_count_rate_chunk_2, second_chunk.coincidence_count_rate)
        np.testing.assert_array_equal(expected_count_rate_uncertainty_chunk_2,
                                      second_chunk.coincidence_count_rate_uncertainty)

    def test_reading_l2_data_into_model(self):
        path = Path('temp_cdf.cdf')
        if path.exists():
            os.remove(path)

        temp_cdf = CDF('temp_cdf', '')
        temp_cdf["sci_start_time"] = np.array(['2010-01-01T00:00:46.000'])
        temp_cdf["esa_energy"] = np.array([1, -1e31, 3, 4], dtype=float)
        temp_cdf["swp_coin_rate"] = np.array([5, 6, 7, -1e31], dtype=float)
        temp_cdf["swp_coin_rate_stat_uncert_plus"] = np.array([2, 2, -1e31, 2, 2, 2, 2, 2], dtype=float)

        temp_cdf["sci_start_time"].attrs["FILLVAL"] = '0'
        temp_cdf["esa_energy"].attrs["FILLVAL"] = -1e31
        temp_cdf["swp_coin_rate"].attrs["FILLVAL"] = -1e31
        temp_cdf["swp_coin_rate_stat_uncert_plus"].attrs["FILLVAL"] = -1e31

        temp_cdf.close()

        actual_swapi_l2_data = read_l2_swapi_data(CDF("temp_cdf.cdf"))

        epoch_as_tt2000 = 315576112184000000
        np.testing.assert_array_equal(np.array(epoch_as_tt2000), actual_swapi_l2_data.sci_start_time)
        np.testing.assert_array_equal(np.array([1, np.nan, 3, 4]), actual_swapi_l2_data.energy)
        np.testing.assert_array_equal(np.array([5, 6, 7, np.nan]), actual_swapi_l2_data.coincidence_count_rate)
        np.testing.assert_array_equal(np.array([2, 2, np.nan, 2, 2, 2, 2, 2]),
                                      actual_swapi_l2_data.coincidence_count_rate_uncertainty)


class TestCalculateSwSpeed(TestCase):
    def test_2d_array_matches_analytic_formula_per_element(self):
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        expected = (
            np.sqrt(2 * E * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        np.testing.assert_allclose(result, expected)

    def test_2d_array_input_preserves_shape(self):
        E = np.array([[1.0e-16, 2.0e-16], [4.0e-16, 8.0e-16]])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertEqual(result.shape, E.shape)

    def test_empty_array_input_returns_empty_array(self):
        result = calculate_sw_speed(
            PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, np.array([])
        )
        self.assertEqual(result.size, 0)

    def test_ufloat_scalar_propagates_uncertainty(self):
        E = ufloat(1.0e-16, 1.0e-18)
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E)
        self.assertIsInstance(result, UFloat)
        # σ_v = v · σ_E / (2 E)
        expected_nom = (
            np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
            / METERS_PER_KILOMETER
        )
        expected_sigma = expected_nom * E.std_dev / (2 * E.nominal_value)
        self.assertAlmostEqual(result.nominal_value, expected_nom)
        self.assertAlmostEqual(result.std_dev, expected_sigma)

    def test_ufloat_array_input_propagates_uncertainty_per_element(self):
        # Array of UFloat scalars takes the `unumpy.sqrt` branch — different
        # code path from the float-array branch above. Each element should
        # propagate its own σ_E.
        E_values = np.array([ufloat(1.0e-16, 1.0e-18), ufloat(4.0e-16, 2.0e-18)])
        result = calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, E_values)
        self.assertEqual(result.shape, E_values.shape)
        for r, E in zip(result, E_values):
            self.assertIsInstance(r, UFloat)
            expected_nom = (
                np.sqrt(2 * E.nominal_value * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)
                / METERS_PER_KILOMETER
            )
            self.assertAlmostEqual(r.nominal_value, expected_nom)
            self.assertAlmostEqual(
                r.std_dev, expected_nom * E.std_dev / (2 * E.nominal_value)
            )


class TestReadMagRtnData(TestCase):
    def setUp(self) -> None:
        self.cdf_path = Path('temp_mag_cdf.cdf')
        if self.cdf_path.exists():
            os.remove(self.cdf_path)

    def tearDown(self) -> None:
        if self.cdf_path.exists():
            os.remove(self.cdf_path)

    def test_reads_b_rtn_and_epoch_into_mag_data(self):
        epochs = np.array([datetime(2026, 1, 1, 0, 0, 0),
                           datetime(2026, 1, 1, 0, 0, 1)])
        b_rtn = np.array(
            [[1.0, 2.0, 3.0, 0.0],
             [4.0, 5.0, 6.0, 0.0]],
        )
        cdf = CDF(str(self.cdf_path.with_suffix("")), '')
        cdf["epoch"] = epochs
        cdf["b_rtn"] = b_rtn
        cdf["b_rtn"].attrs["FILLVAL"] = -1e31
        cdf.close()

        result = read_mag_rtn_data(self.cdf_path)

        expected_epoch_tt2000 = spacepy.pycdf.lib.v_datetime_to_tt2000(epochs)
        np.testing.assert_array_equal(result.epoch, expected_epoch_tt2000)
        np.testing.assert_array_equal(
            result.mag_data, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )


class TestSwapiSpiceHelpers(SpiceTestCase):
    # 2025-06-06 12:00 UTC — inside the IMAP SPK and attitude coverage windows
    # used by the shipped `spice_kernels/` set.
    _EPOCH_TT2000_NS = spacepy.pycdf.lib.datetime_to_tt2000(
        datetime(2025, 6, 6, 12, 0, 0)
    )

    def test_get_swapi_geometry_returns_orthonormal_rotation_per_time(self):
        # Three TT2000 ns samples spanning ~30 s.
        # IMAP's spin period is 15 s,
        # so these should produce distinct rotations.
        times = self._EPOCH_TT2000_NS + np.array([0, 10_000_000_000, 20_000_000_000])

        rotation_matrices = get_swapi_geometry(times)

        self.assertEqual(rotation_matrices.shape, (3, 3, 3))
        for matrix in rotation_matrices:
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=10)

    def test_rotate_rtn_to_dps_preserves_vector_magnitude(self):
        vector_rtn = np.array([100.0, -50.0, 25.0])

        rotated = rotate_rtn_to_dps(vector_rtn, self._EPOCH_TT2000_NS)

        self.assertEqual(rotated.shape, (3,))
        self.assertAlmostEqual(
            float(np.linalg.norm(rotated)),
            float(np.linalg.norm(vector_rtn)),
            places=8,
        )

    def test_get_spacecraft_velocity_rtn_returns_finite_orbital_velocity(self):
        velocity_rtn = get_spacecraft_velocity_rtn(self._EPOCH_TT2000_NS)

        self.assertEqual(velocity_rtn.shape, (3,))
        self.assertTrue(np.all(np.isfinite(velocity_rtn)))
        # IMAP sits near L1; its barycentric speed is dominated by Earth's
        # ~30 km/s orbital motion. A loose bound rules out unit-conversion
        # mistakes (m/s vs km/s) without pinning a frame-specific value.
        speed = float(np.linalg.norm(velocity_rtn))
        self.assertGreater(speed, 10.0)
        self.assertLess(speed, 60.0)
