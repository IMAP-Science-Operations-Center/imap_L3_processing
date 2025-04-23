import unittest
from datetime import datetime
from unittest.mock import patch, sentinel, call, MagicMock

import numpy as np
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.l3.models import HiL1cData, HiGlowsL3eData
from imap_l3_processing.hi.l3.science.survival_probability import Sensor, \
    HiSurvivalProbabilitySkyMap, HiSurvivalProbabilityPointingSet, interpolate_angular_data_to_nearest_neighbor
from imap_l3_processing.hi.l3.utils import SpinPhase


class TestSurvivalProbability(unittest.TestCase):
    def setUp(self):
        self.num_energies = 2
        self.epoch = datetime.now()

        self.hi_energies = np.geomspace(1, 10000, self.num_energies)

        self.l1c_hi_dataset = HiL1cData(
            epoch=self.epoch,
            epoch_j2000=np.array([43264184000000]),
            exposure_times=np.arange(self.num_energies * 3600).reshape((1, self.num_energies, 3600)) + 1.1,
            esa_energy_step=np.arange(self.num_energies),
        )

        self.glows_data = HiGlowsL3eData(
            epoch=self.epoch,
            energy=np.geomspace(1, 10000, self.num_energies + 1),
            spin_angle=np.arange(0, 360, 1) + 0.5,
            probability_of_survival=np.arange((self.num_energies + 1) * 360).reshape(
                (1, self.num_energies + 1, 360)) + 5.4)

        l1c_spin_angles = np.linspace(0, 360, 3600, endpoint=False) + 0.05
        self.ram_mask = (l1c_spin_angles < 90) | (l1c_spin_angles > 270)
        self.antiram_mask = np.logical_not(self.ram_mask)

    @patch('imap_l3_processing.hi.l3.science.survival_probability.PointingSet.__init__')
    def test_survival_probability_pointing_set_calls_parent_constructor(self,
                                                                        mock_rectangular_pointing_set_constructor):
        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi45, SpinPhase.RamOnly,
                                                        self.glows_data,
                                                        self.hi_energies)
        self.assertIsInstance(pointing_set, PointingSet)

        mock_rectangular_pointing_set_constructor.assert_called_once()

    def test_survival_probability_pointing_set(self):
        test_cases = [
            (Sensor.Hi90, 0, SpinPhase.RamOnly, self.ram_mask),
            (Sensor.Hi90, 0, SpinPhase.AntiRamOnly, self.antiram_mask),
            (Sensor.Hi45, -45, SpinPhase.RamOnly, self.ram_mask),
            (Sensor.Hi45, -45, SpinPhase.AntiRamOnly, self.antiram_mask),
        ]

        for sensor, expected_sensor_angle, spin_phase, expected_mask in test_cases:
            with self.subTest(f"{sensor.value}"):
                pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, spin_phase,
                                                                self.glows_data,
                                                                self.hi_energies)
                self.assertIsInstance(pointing_set, PointingSet)
                self.assertEqual(pointing_set.spice_reference_frame, geometry.SpiceFrame.IMAP_DPS)

                self.assertIn("exposure", pointing_set.data.data_vars)
                np.testing.assert_array_equal(
                    pointing_set.data["exposure"].values,
                    self.l1c_hi_dataset.exposure_times * expected_mask)

                self.assertIn(CoordNames.AZIMUTH_L1C.value, pointing_set.data.coords)
                np.testing.assert_array_almost_equal(
                    np.concatenate([np.arange(90, 360, 0.1), np.arange(0, 90, 0.1)]) + 0.05,
                    pointing_set.data[CoordNames.AZIMUTH_L1C.value].values)

                self.assertIn(CoordNames.ENERGY.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset.esa_energy_step,
                                              pointing_set.data[CoordNames.ENERGY.value].values)

                self.assertIn(CoordNames.TIME.value, pointing_set.data.coords)
                np.testing.assert_array_equal(self.l1c_hi_dataset.epoch_j2000,
                                              pointing_set.data[CoordNames.TIME.value].values)

                np.testing.assert_array_equal(pointing_set.az_el_points[:, 0],
                                              pointing_set.data[CoordNames.AZIMUTH_L1C.value].values)

                np.testing.assert_array_equal(pointing_set.az_el_points[:, 1], np.repeat(expected_sensor_angle, 3600))

    def test_exposure_weighting_with_interpolated_survival_probabilities(self):
        test_cases = [
            (SpinPhase.RamOnly, self.ram_mask),
            (SpinPhase.AntiRamOnly, self.antiram_mask),
        ]

        for spin_phase, expected_mask in test_cases:
            with (self.subTest(spin_phase)):
                self.hi_energies = np.array([10, 10_000])
                self.glows_data.energy = np.array([1, 100, 100_000])
                self.glows_data.probability_of_survival = np.repeat([2, 4, 7], 360).reshape(1, 3, 360)

                expected_interpolated_survival_probabilities = \
                    np.repeat([3, 6], 3600).reshape(1, 2, 3600) * self.l1c_hi_dataset.exposure_times * expected_mask

                sensor, expected_skygrid_elevation_index = Sensor.Hi90, 901
                pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, sensor, spin_phase,
                                                                self.glows_data,
                                                                self.hi_energies)

                self.assertIn("survival_probability_times_exposure", pointing_set.data.data_vars)
                self.assertEqual((1, self.num_energies, 3600),
                                 pointing_set.data["survival_probability_times_exposure"].values.shape)

                np.testing.assert_array_equal(
                    expected_interpolated_survival_probabilities,
                    pointing_set.data["survival_probability_times_exposure"].values)

    def test_exposure_weighted_survivals_are_repeated_to_match_l1c_shape(self):
        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                        self.glows_data,
                                                        self.hi_energies)

        survivals = pointing_set.data[
                        "survival_probability_times_exposure"].values / self.l1c_hi_dataset.exposure_times
        every_tenth_value = survivals[:, :, ::10, np.newaxis]
        groups_of_ten = survivals.reshape(1, self.num_energies, 360, 10)
        self.assertTrue(np.all(np.isclose(every_tenth_value, groups_of_ten)))

    @patch("imap_l3_processing.hi.l3.science.survival_probability.interpolate_angular_data_to_nearest_neighbor")
    def test_survivals_matched_with_corresponding_exposures(self, mock_interpolate):
        self.hi_energies = np.array([1, 100])
        self.glows_data.energy = np.array([1, 100, 100_000])

        first_energy_corresponding_glows_data = np.linspace(0, 1, 3600)
        second_energy_corresponding_glows_data = np.linspace(0, 1, 3600) + 100.2

        mock_interpolate.side_effect = [first_energy_corresponding_glows_data,
                                        second_energy_corresponding_glows_data]

        pointing_set = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                        self.glows_data,
                                                        self.hi_energies)

        pset_spin_angles = np.linspace(0, 360, 3600, endpoint=False) + 0.05
        pset_azimuths = np.mod(pset_spin_angles + 90, 360)

        self.assertEqual(2, mock_interpolate.call_count)
        get_glows_data_for_first_energy_args = mock_interpolate.call_args_list[0].args
        np.testing.assert_array_equal(pset_azimuths, get_glows_data_for_first_energy_args[0])
        np.testing.assert_array_equal(self.glows_data.spin_angle, get_glows_data_for_first_energy_args[1])
        np.testing.assert_array_equal(self.glows_data.probability_of_survival[0, 0],
                                      get_glows_data_for_first_energy_args[2])

        get_glows_data_for_second_energy_args = mock_interpolate.call_args_list[1].args
        np.testing.assert_array_equal(pset_azimuths, get_glows_data_for_second_energy_args[0])
        np.testing.assert_array_equal(self.glows_data.spin_angle, get_glows_data_for_second_energy_args[1])
        np.testing.assert_array_equal(self.glows_data.probability_of_survival[0, 1],
                                      get_glows_data_for_second_energy_args[2])

        corresponding_glows_data = np.array(
            [first_energy_corresponding_glows_data, second_energy_corresponding_glows_data])[np.newaxis, ...]

        np.testing.assert_array_almost_equal(pointing_set.data["survival_probability_times_exposure"].values,
                                             corresponding_glows_data * self.l1c_hi_dataset.exposure_times * self.ram_mask)

    def test_interpolate_angular_data_to_nearest_neighbor(self):
        input_cases = [
            (0, 360),
            (0.1, 360),
            (0.4, 360),
            (0.5, 360),
            (0.6, 1),
            (120.2, 120),
            (359.4, 359),
            (359.6, 360),
            (360, 360),
        ]
        basic_glows_spin_angles = np.linspace(1, 360, 360, endpoint=True)
        basic_glows_data = np.linspace(1, 360, 360, endpoint=True) + 1000
        glows_cases = [
            ("basic", basic_glows_spin_angles, basic_glows_data),
            ("0 instead of 360", np.mod(basic_glows_spin_angles, 360), basic_glows_data),
            ("with 0 at start", np.roll(np.mod(basic_glows_spin_angles, 360), 1), np.roll(basic_glows_data, 1)),
            ("rolled", np.roll(basic_glows_spin_angles, 90), np.roll(basic_glows_data, 90)),
        ]
        for name, glows_spin_angles, glows_data in glows_cases:
            with self.subTest(name):
                input_angles, expected_output = np.array(input_cases).T
                result = interpolate_angular_data_to_nearest_neighbor(input_angles, glows_spin_angles, glows_data)
                np.testing.assert_array_equal(expected_output + 1000, result)

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

    def test_survival_probability_sky_map_returns_exposure_weighted_survival_probabilities(self):
        pset1 = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                 self.glows_data, self.hi_energies)
        pset2 = HiSurvivalProbabilityPointingSet(self.l1c_hi_dataset, Sensor.Hi90, SpinPhase.RamOnly,
                                                 self.glows_data, self.hi_energies)

        pset1.data = pset1.data.assign(
            survival_probability_times_exposure=4 * pset1.data["survival_probability_times_exposure"])
        pset1.data = pset1.data.assign(
            exposure=2 * pset1.data["exposure"])

        summed_pset_survival_prob_by_spin_angle = pset1.data["survival_probability_times_exposure"].values + pset2.data[
            "survival_probability_times_exposure"].values
        summed_pset_survival_prob_by_azimuth = np.roll(summed_pset_survival_prob_by_spin_angle, 900, axis=-1)

        summed_pset_exposure_by_spin_angle = pset1.data["exposure"].values + pset2.data["exposure"].values
        summed_pset_exposure_by_azimuth = np.roll(summed_pset_exposure_by_spin_angle, 900, axis=-1)

        survival_prob_in_skygrid_shape = np.zeros((1, 2, 3600, 1800))
        survival_prob_in_skygrid_shape[:, :, :, 900] = summed_pset_survival_prob_by_azimuth

        summed_exposure = np.zeros((1, 2, 3600, 1800))
        summed_exposure[:, :, :, 900] = summed_pset_exposure_by_azimuth
        summed_exposure[summed_exposure == 0] = np.nan

        spice_frame = SpiceFrame.IMAP_DPS
        actual_skymap = HiSurvivalProbabilitySkyMap([pset1, pset2],
                                                    0.1, spice_frame)

        expected_exposure_weighted_survival_skygrid = np.divide(survival_prob_in_skygrid_shape, summed_exposure)

        survival_probability_dataset = actual_skymap.to_dataset()

        self.assertIn("exposure_weighted_survival_probabilities", survival_probability_dataset)
        self.assertEqual((1, 2, 3600, 1800),
                         survival_probability_dataset["exposure_weighted_survival_probabilities"].values.shape)
        np.testing.assert_array_equal(expected_exposure_weighted_survival_skygrid,
                                      survival_probability_dataset["exposure_weighted_survival_probabilities"].values)
