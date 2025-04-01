import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import xarray as xr

from imap_processing.ena_maps.ena_maps import RectangularPointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames

from imap_l3_processing.hi.l3.science.survival_probability import HiSurvivalProbabilityPointingSet, Sensor


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

    @patch('imap_l3_processing.hi.l3.science.survival_probability.RectangularPointingSet.__init__')
    def test_survival_probability_pointing_set_calls_parent_constructor(self,
                                                                        mock_rectangular_pointing_set_constructor):
        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi45, self.glows_data)
        self.assertIsInstance(pointing_set, RectangularPointingSet)
        mock_rectangular_pointing_set_constructor.assert_called_once_with(pointing_set.data)

    def test_survival_probability_pointing_set(self):
        test_cases = [
            (Sensor.Hi90, 901),
            (Sensor.Hi45, 451)
        ]

        for sensor, expected_skygrid_elevation_index in test_cases:
            with self.subTest(f"{sensor.value}"):
                pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, self.glows_data)
                self.assertIsInstance(pointing_set, RectangularPointingSet)

                skygrid_pointing_mask = np.full((1, self.num_energies, 3600, 1800), False)
                skygrid_pointing_mask[:, :, :, expected_skygrid_elevation_index] = True

                self.assertIn("exposure", pointing_set.data.data_vars)
                self.assertEqual((1, self.num_energies, 3600, 1800), pointing_set.data["exposure"].values.shape)
                np.testing.assert_array_equal(
                    pointing_set.data["exposure"].values[:, :, :, expected_skygrid_elevation_index],
                    self.l1c_hi_dataset["exposure_times"].values)
                pointing_set.data["exposure"].values[:, :, :, expected_skygrid_elevation_index] = 0
                self.assertTrue(
                    np.all(pointing_set.data["exposure"].values[np.logical_not(skygrid_pointing_mask)] == 0))

                self.assertIn(CoordNames.AZIMUTH_L1C.value, pointing_set.data.coords)
                np.testing.assert_array_equal(np.arange(0, 360, 0.1) + 0.05,
                                              pointing_set.data[CoordNames.AZIMUTH_L1C.value].values)
                self.assertIn(CoordNames.ELEVATION_L1C.value, pointing_set.data.coords)
                np.testing.assert_array_equal(np.arange(-90, 90, 0.1) + 0.05,
                                              pointing_set.data[CoordNames.ELEVATION_L1C.value].values)

                self.assertIn(CoordNames.ENERGY.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset["esa_energy_step"].values,
                                              pointing_set.data[CoordNames.ENERGY.value].values)
                self.assertIn(CoordNames.TIME.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset["epoch"].values,
                                              pointing_set.data[CoordNames.TIME.value].values)

    def test_exposure_weighting_with_interpolated_survival_probabilities(self):
        self.l1c_hi_dataset = self.l1c_hi_dataset.assign_coords(esa_energy_step=np.array([10, 10_000]))
        self.glows_data = self.glows_data.assign_coords(energy=np.array([1, 100, 100_000]))
        self.glows_data["probability_of_survival"].values[0, :, 0] = [2, 4, 7]

        expected_interpolated_survival_probabilities = np.array([3, 6])

        sensor, expected_skygrid_elevation_index = Sensor.Hi90, 901
        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, self.glows_data)

        skygrid_pointing_mask = np.full((1, self.num_energies, 3600, 1800), False)
        skygrid_pointing_mask[:, :, :, expected_skygrid_elevation_index] = True

        self.assertIn("survival_probability_times_exposure", pointing_set.data.data_vars)
        self.assertEqual((1, self.num_energies, 3600, 1800),
                         pointing_set.data["survival_probability_times_exposure"].values.shape)
        self.assertTrue(np.all(
            pointing_set.data["survival_probability_times_exposure"].values[
                np.logical_not(skygrid_pointing_mask)] == 0))
        np.testing.assert_array_equal(
            expected_interpolated_survival_probabilities * self.l1c_hi_dataset["exposure_times"].values[0, :, 0],
            pointing_set.data["survival_probability_times_exposure"].values[0, :, 0, expected_skygrid_elevation_index])
