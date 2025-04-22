import unittest
from datetime import datetime
from unittest.mock import patch

import imap_processing.tests.ultra.data.mock_data as umd
import numpy as np
from astropy_healpix import HEALPix
from imap_processing.ena_maps.ena_maps import UltraPointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry

from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.ultra.l3.science.ultra_survival_probability import UltraSurvivalProbability


class TestUltraSurvivalProbability(unittest.TestCase):

    @patch('imap_l3_processing.ultra.l3.science.ultra_survival_probability.geometry.frame_transform_az_el')
    def test_ultra_survival_probability_pset_calls_super(self, mock_frame_transform_az_el):
        l1cdataset = umd.mock_l1c_pset_product_healpix(nside=16)
        input_l1c_pset = _create_ultra_l1c_pset_from_xarray(l1cdataset)
        glows = _build_glows_l3e_ultra()

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        l1cdataset.attrs = None

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        self.assertIsInstance(prod, UltraPointingSet)
        self.assertEqual(l1cdataset, prod.data)
        self.assertEqual(geometry.SpiceFrame.IMAP_DPS, prod.spice_reference_frame)

    @patch('imap_l3_processing.ultra.l3.science.ultra_survival_probability.geometry.frame_transform_az_el')
    def test_ultra_survival_probability_rotates_to_glows_frame(self, mock_frame_transform_az_el):
        glows_energies = np.array([1, 2])
        input_l1c_pset = _create_ultra_l1c_pset(glows_energies, np.full((1, 2, 12), 1))
        glows_surv_prob = np.array([[[1] * 6 + [0] * 6, [2] * 6 + [3] * 6]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        rotate_el_by_180 = np.array([[1, -1]] * 12)
        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el * rotate_el_by_180

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        expected_survival_probabilities_values = np.array([[[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                                                            [3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2]]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)
        np.testing.assert_array_equal(survival_probability_times_exposure.coords, prod.data['counts'].coords)

    @patch('imap_l3_processing.ultra.l3.science.ultra_survival_probability.geometry.frame_transform_az_el')
    def test_ultra_survival_probability_is_multiplied_by_exposure(self, mock_frame_transform_az_el):
        glows_energies = np.array([1])
        input_l1c_pset = _create_ultra_l1c_pset(glows_energies, np.full((1, 1, 12), 2))
        glows_surv_prob = np.array([[[1] * 6 + [0] * 6]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        expected_survival_probabilities_values = np.array([[[2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)

    @patch('imap_l3_processing.ultra.l3.science.ultra_survival_probability.geometry.frame_transform_az_el')
    def test_ultra_survival_probability_interpolates_over_energy(self, mock_frame_transform_az_el):
        glows_energies = np.array([1, 100, 10000])
        ultra_energies = np.array([10, 1000])
        input_l1c_pset = _create_ultra_l1c_pset(ultra_energies, np.full((1, 2, 12), 1))
        glows_surv_prob = np.array([[[1] * 12, [3] * 12, [5] * 12]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        expected_survival_probabilities_values = np.array([[[2] * 12, [4] * 12]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_equal(survival_probability_times_exposure.values,
                                      expected_survival_probabilities_values)

    @patch('imap_l3_processing.ultra.l3.science.ultra_survival_probability.geometry.frame_transform_az_el')
    def test_ultra_survival_probability_handles_different_resolutions(self, mock_frame_transform_az_el):
        glows_energies = np.array([1])
        ultra_energies = np.array([10])
        input_l1c_pset = _create_ultra_l1c_pset(ultra_energies, np.full((1, 1, 48), 1))
        glows_surv_prob = np.array([[np.arange(12)]])

        glows = _build_glows_l3e_ultra(nside=1, survival_probabilities=glows_surv_prob, energies=glows_energies)

        mock_frame_transform_az_el.side_effect = lambda et, az_el, to, _from: az_el

        prod = UltraSurvivalProbability(input_l1c_pset, glows)

        expected_survival_probabilities_values = np.array([[[0.7667581, 1.25558603, 1.74441397, 2.2332419, 0.75, 0.25,
                                                             0.75, 1.25, 1.75, 2.25, 2.75, 2.25,
                                                             2.83574061, 2.4043331, 2.9043331, 3.4043331, 3.9043331,
                                                             4.4043331,
                                                             4.9043331, 4.33574061, 4.25, 4.75, 5.25, 5.75,
                                                             6.25, 6.75, 6.25, 4.75, 6.56137065, 6.12996314,
                                                             6.62996314, 7.12996314, 7.62996314, 8.12996314, 8.62996314,
                                                             8.06137065,
                                                             8.75, 8.25, 8.75, 9.25, 9.75, 10.25,
                                                             10.75, 10.25, 8.7667581, 9.25558603, 9.74441397,
                                                             10.2332419]]])
        survival_probability_times_exposure = prod.data['survival_probability_times_exposure']
        np.testing.assert_array_almost_equal(survival_probability_times_exposure.values,
                                             expected_survival_probabilities_values)


def _build_glows_l3e_ultra(survival_probabilities: np.ndarray = None, energies: np.ndarray = None, nside: int = 16):
    energies = np.arange(1, 21) if energies is None else energies
    num_pixels = 12 * (nside ** 2)
    survival_probabilities = np.random.rand(1, len(energies),
                                            num_pixels) if survival_probabilities is None else survival_probabilities
    healpix_pixels = np.arange(num_pixels)

    healpix_grid = HEALPix(nside)
    lon, lat = healpix_grid.healpix_to_lonlat(healpix_pixels)

    return UltraGlowsL3eData(
        epoch=datetime.now(),
        energy=energies,
        healpix_index=healpix_pixels,
        survival_probability=survival_probabilities,
        latitude=np.rad2deg(lat.value),
        longitude=np.rad2deg(lon.value)
    )


def _create_ultra_l1c_pset_from_xarray(l1cdataset):
    l1c_energies = l1cdataset.coords[CoordNames.ENERGY.value].values
    l1c_exposure = np.repeat(l1cdataset["exposure_time"].values[np.newaxis, :], len(l1c_energies), 1)
    l1c_exposure = np.reshape(l1c_exposure, (1, len(l1c_energies), -1))

    l1cdataset["exposure_time"] = (
        [
            CoordNames.TIME.value,
            CoordNames.ENERGY.value,
            CoordNames.HEALPIX_INDEX.value
        ],
        l1c_exposure
    )

    input_l1c_pset = UltraL1CPSet(
        epoch=l1cdataset.coords[CoordNames.TIME.value].values[0],
        energy=l1cdataset.coords[CoordNames.ENERGY.value].values,
        counts=l1cdataset["counts"].values,
        exposure=l1cdataset["exposure_time"].values,
        healpix_index=l1cdataset.coords[CoordNames.HEALPIX_INDEX.value].values,
        latitude=l1cdataset[CoordNames.ELEVATION_L1C.value].values,
        longitude=l1cdataset[CoordNames.AZIMUTH_L1C.value].values,
        sensitivity=l1cdataset["sensitivity"].values
    )
    return input_l1c_pset


def _create_ultra_l1c_pset(energy: np.ndarray,
                           exposure_time: np.ndarray,
                           sensitivity: np.ndarray = None,
                           counts: np.ndarray = None,
                           latitude: np.ndarray = None,
                           longitude: np.ndarray = None):
    counts = counts or np.full_like(exposure_time, 1)
    sensitivity = sensitivity or np.full_like(exposure_time, 1)
    healpix_index = np.arange(exposure_time.shape[-1])
    healpix = HEALPix(nside=int(np.sqrt(len(healpix_index) / 12)))
    lon_pix, lat_pix = healpix.healpix_to_lonlat(healpix_index)
    input_l1c_pset = UltraL1CPSet(
        epoch=datetime(2025, 9, 10),
        energy=energy,
        counts=counts,
        exposure=exposure_time,
        healpix_index=healpix_index,
        latitude=np.rad2deg(lat_pix.value),
        longitude=np.rad2deg(lon_pix.value),
        sensitivity=sensitivity
    )
    return input_l1c_pset


if __name__ == '__main__':
    unittest.main()
