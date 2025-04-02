import unittest
from datetime import datetime
from unittest.mock import patch, sentinel, call, MagicMock

import numpy as np
import xarray as xr

from imap_processing.ena_maps.ena_maps import RectangularPointingSet, RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.geometry import SpiceFrame
from numpy.random import geometric

from imap_l3_processing.hi.l3.science.survival_probability import Sensor, \
    HiSurvivalProbabilitySkyMap, HiSurvivalProbabilityPointingSet


class TestSurvivalProbability(unittest.TestCase):
    def setUp(self):
        self.num_energies = 2
        self.epoch = datetime.now()

        self.l1c_hi_dataset = xr.Dataset({
            "exposure_times": (
                [
                    "epoch",
                    "esa_energy_step",
                    "hi_pset_spin_angle_bin"
                ],
                np.arange(self.num_energies * 3600).reshape((1, self.num_energies, 3600)) + 1.1
            ),
        },
            coords={
                "epoch": [self.epoch],
                "esa_energy_step": np.geomspace(1, 10000, self.num_energies),
                "hi_pset_spin_angle_bin": np.arange(0, 360, 0.1) + 0.05,
            }
        )

        self.glows_data = xr.Dataset({
            "probability_of_survival": (
                [
                    "epoch",
                    "energy",
                    "spin_angle_bin"
                ],
                np.arange((self.num_energies + 1) * 360).reshape((1, self.num_energies + 1, 360)) + 5.4,
            )
        },
            coords={
                "epoch": [self.epoch],
                "energy": np.geomspace(1, 10000, self.num_energies + 1),
                "spin_angle_bin": np.arange(0, 360, 1) + 0.5,
            })

    @patch('imap_l3_processing.hi.l3.science.survival_probability.PointingSet.__init__')
    def test_survival_probability_pointing_set_calls_parent_constructor(self,
                                                                        mock_rectangular_pointing_set_constructor):
        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi45, self.glows_data)
        self.assertIsInstance(pointing_set, PointingSet)

        mock_rectangular_pointing_set_constructor.assert_called_once()

    def test_survival_probability_pointing_set(self):
        test_cases = [
            (Sensor.Hi90, 0),
            (Sensor.Hi45, -45)
        ]

        for sensor, expected_sensor_angle in test_cases:
            with self.subTest(f"{sensor.value}"):
                pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, self.glows_data)
                self.assertIsInstance(pointing_set, PointingSet)
                self.assertEqual(pointing_set.spice_reference_frame, geometry.SpiceFrame.IMAP_DPS)

                self.assertIn("exposure", pointing_set.data.data_vars)
                np.testing.assert_array_equal(
                    pointing_set.data["exposure"].values,
                    self.l1c_hi_dataset["exposure_times"].values)

                self.assertIn(CoordNames.AZIMUTH_L1C.value, pointing_set.data.coords)
                np.testing.assert_array_equal(np.arange(0, 360, 0.1) + 0.05,
                                              pointing_set.data[CoordNames.AZIMUTH_L1C.value].values)

                self.assertIn(CoordNames.ENERGY.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset["esa_energy_step"].values,
                                              pointing_set.data[CoordNames.ENERGY.value].values)

                self.assertIn(CoordNames.TIME.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset["epoch"].values,
                                              pointing_set.data[CoordNames.TIME.value].values)

                np.testing.assert_array_equal(pointing_set.az_el_points[:, 0],
                                              pointing_set.data[CoordNames.AZIMUTH_L1C.value].values)

                np.testing.assert_array_equal(pointing_set.az_el_points[:, 1], np.repeat(expected_sensor_angle, 3600))

    def test_exposure_weighting_with_interpolated_survival_probabilities(self):
        self.l1c_hi_dataset = self.l1c_hi_dataset.assign_coords(esa_energy_step=np.array([10, 10_000]))
        self.glows_data = self.glows_data.assign_coords(energy=np.array([1, 100, 100_000]))
        self.glows_data["probability_of_survival"].values[0, :, 0] = [2, 4, 7]

        expected_interpolated_survival_probabilities = np.array([3, 6])

        sensor, expected_skygrid_elevation_index = Sensor.Hi90, 901
        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, self.glows_data)

        self.assertIn("survival_probability_times_exposure", pointing_set.data.data_vars)
        self.assertEqual((1, self.num_energies, 3600),
                         pointing_set.data["survival_probability_times_exposure"].values.shape)

        survivals = pointing_set.data["survival_probability_times_exposure"].values / self.l1c_hi_dataset[
            "exposure_times"].values
        every_tenth_value = survivals[:, :, ::10, np.newaxis]
        groups_of_ten = survivals.reshape(1, self.num_energies, 360, 10)
        self.assertTrue(np.all(np.isclose(every_tenth_value, groups_of_ten)))

        np.testing.assert_array_equal(
            expected_interpolated_survival_probabilities * self.l1c_hi_dataset["exposure_times"].values[0, :, 0],
            pointing_set.data["survival_probability_times_exposure"].values[0, :, 0])

    @patch('imap_l3_processing.hi.l3.science.survival_probability.RectangularSkyMap.project_pset_values_to_map')
    @patch('imap_l3_processing.hi.l3.science.survival_probability.RectangularSkyMap.__init__')
    def test_survival_probability_map_construction(self, mock_skymap_constructor, mock_project_pset):
        class TestableHiSurvivalProbabilitySkyMap(HiSurvivalProbabilitySkyMap):
            def __init__(self, *args):
                self.data_1d = MagicMock()
                super().__init__(*args)

        actual_sky_map = TestableHiSurvivalProbabilitySkyMap([sentinel.pset_1, sentinel.pset_2], sentinel.spacing_deg,
                                                             sentinel.spice_frame)
        self.assertIsInstance(actual_sky_map, RectangularSkyMap)

        mock_skymap_constructor.assert_called_with(sentinel.spacing_deg, sentinel.spice_frame)

        mock_project_pset.assert_has_calls([
            call(sentinel.pset_1, ["survival_probability_times_exposure", "exposure"]),
            call(sentinel.pset_2, ["survival_probability_times_exposure", "exposure"]),
        ])

    def test_survival_probabilty_returns_exposure_weighted_survival_probablities(self):
        pset1 = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, self.glows_data)
        pset2 = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, self.glows_data)

        pset1.data = pset1.data.assign(
            survival_probability_times_exposure=4 * pset1.data["survival_probability_times_exposure"])
        pset1.data = pset1.data.assign(
            exposure=2 * pset1.data["exposure"])

        summed_survivals = np.empty((1, 2, 3600, 1800))
        summed_survivals[0, :, :, 900] = pset1.data["survival_probability_times_exposure"].values + pset2.data[
            "survival_probability_times_exposure"].values
        summed_exposure = np.empty((1, 2, 3600, 1800))
        summed_exposure[0, :, :, 900] = pset1.data["exposure"].values + pset2.data["exposure"].values
        summed_exposure[summed_exposure == 0] = np.nan

        spice_frame = SpiceFrame.IMAP_DPS
        actual_skymap = HiSurvivalProbabilitySkyMap([pset1, pset2],
                                                    0.1, spice_frame)

        expected_skygrid_1d = np.divide(summed_survivals, summed_exposure).reshape(1, 2, -1)
        np.testing.assert_equal(actual_skymap.data_1d['exposure_weighted_survival_probabilities'].values,
                                expected_skygrid_1d)

        survival_probability_dataset = actual_skymap.to_dataset()

        self.assertIn("exposure_weighted_survival_probabilities", survival_probability_dataset)
        self.assertEqual((1, 2, 3600, 1800),
                         survival_probability_dataset["exposure_weighted_survival_probabilities"].values.shape)
