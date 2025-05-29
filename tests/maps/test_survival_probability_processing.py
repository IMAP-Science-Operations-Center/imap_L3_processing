from datetime import datetime
from unittest.mock import sentinel, patch, call, Mock

import numpy as np
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import PixelSize, parse_map_descriptor
from imap_l3_processing.maps.map_models import RectangularIntensityMapData
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities
from tests.maps.test_builders import create_h1_l3_data, create_l1c_pset, create_l3e_pset
from tests.spice_test_case import SpiceTestCase


class TestSurvivalProbabilityProcessing(SpiceTestCase):
    @patch('imap_l3_processing.maps.survival_probability_processing.RectangularSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.maps.survival_probability_processing.RectangularSurvivalProbabilityPointingSet')
    @patch('imap_l3_processing.maps.survival_probability_processing.combine_glows_l3e_with_l1c_pointing')
    def test_process_survival_probability(self, mock_combine_glows_l3e_with_l1c_pointing,
                                          mock_survival_probability_pointing_set, mock_survival_skymap):
        rng = np.random.default_rng()
        input_map_flux = rng.random((1, 9, 90, 45))
        epoch = datetime.now()

        input_map: RectangularIntensityMapData = create_h1_l3_data(epoch=[epoch], flux=input_map_flux)

        intensity_map_data = input_map.intensity_map_data
        input_map.intensity_map_data.energy = sentinel.hi_l2_energies

        l2_grid = PixelSize.FourDegrees
        l2_descriptor_parts = Mock(sensor=sentinel.l2_sensor, spin_phase=sentinel.l2_spin, grid=l2_grid)
        dependencies = HiLoL3SurvivalDependencies(l2_data=input_map,
                                                  l1c_data=sentinel.l1c_data,
                                                  glows_l3e_data=sentinel.glows_l3e_data,
                                                  l2_map_descriptor_parts=l2_descriptor_parts,
                                                  dependency_file_paths=[], )

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.hi_l1c_1, sentinel.glows_l3e_1),
                                                                 (sentinel.hi_l1c_2, sentinel.glows_l3e_2),
                                                                 (sentinel.hi_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        computed_survival_probabilities = rng.random((1, 9, 90, 45))
        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset({
            "exposure_weighted_survival_probabilities": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA_L1C.value,
                    CoordNames.AZIMUTH_L2.value,
                    CoordNames.ELEVATION_L2.value,
                ],
                computed_survival_probabilities
            )
        },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                CoordNames.AZIMUTH_L2.value: rng.random((90,)),
                CoordNames.ELEVATION_L2.value: rng.random((45,)),
            })

        survival_data = process_survival_probabilities(dependencies)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_data, sentinel.l1c_data)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.hi_l1c_1, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_1,
                 sentinel.hi_l2_energies),
            call(sentinel.hi_l1c_2, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_2,
                 sentinel.hi_l2_energies),
            call(sentinel.hi_l1c_3, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_3, sentinel.hi_l2_energies)
        ])

        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     l2_grid,
                                                     SpiceFrame.ECLIPJ2000)

        mock_survival_skymap.return_value.to_dataset.assert_called_once()

        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity,
                                      intensity_map_data.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity_stat_unc,
                                      intensity_map_data.ena_intensity_stat_unc / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data.intensity_map_data.ena_intensity_sys_err,
                                      intensity_map_data.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data.intensity_map_data.epoch, intensity_map_data.epoch)
        np.testing.assert_array_equal(survival_data.intensity_map_data.epoch_delta, intensity_map_data.epoch_delta)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy, intensity_map_data.energy)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy_delta_plus,
                                      intensity_map_data.energy_delta_plus)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy_delta_minus,
                                      intensity_map_data.energy_delta_minus)
        np.testing.assert_array_equal(survival_data.intensity_map_data.energy_label, intensity_map_data.energy_label)
        np.testing.assert_array_equal(survival_data.intensity_map_data.latitude, intensity_map_data.latitude)
        np.testing.assert_array_equal(survival_data.coords.latitude_delta, input_map.coords.latitude_delta)
        np.testing.assert_array_equal(survival_data.coords.latitude_label, input_map.coords.latitude_label)
        np.testing.assert_array_equal(survival_data.intensity_map_data.longitude, intensity_map_data.longitude)
        np.testing.assert_array_equal(survival_data.coords.longitude_delta, input_map.coords.longitude_delta)
        np.testing.assert_array_equal(survival_data.coords.longitude_label, input_map.coords.longitude_label)
        np.testing.assert_array_equal(survival_data.intensity_map_data.exposure_factor,
                                      intensity_map_data.exposure_factor)
        np.testing.assert_array_equal(survival_data.intensity_map_data.obs_date, intensity_map_data.obs_date)
        np.testing.assert_array_equal(survival_data.intensity_map_data.obs_date_range,
                                      intensity_map_data.obs_date_range)
        np.testing.assert_array_equal(survival_data.intensity_map_data.solid_angle, intensity_map_data.solid_angle)

    def test_integration_uses_fill_values_for_missing_l3e_data(self):
        t1 = datetime(2025, 4, 29, 12)
        t2 = datetime(2025, 5, 7, 12)
        t3 = datetime(2025, 5, 8, 12)
        t4 = datetime(2025, 5, 15, 12)
        l1c_psets = [create_l1c_pset(t1), create_l1c_pset(t2), create_l1c_pset(t3), create_l1c_pset(t4)]
        l3e_psets = [create_l3e_pset(t1), create_l3e_pset(t3), create_l3e_pset(t4)]
        l2_intensity_map = create_h1_l3_data()

        descriptor = parse_map_descriptor("h90-ena-h-sf-nsp-ram-hae-4deg-3mo")
        survival_dependencies = HiLoL3SurvivalDependencies(l2_intensity_map, l1c_psets, l3e_psets, descriptor)

        output_map = process_survival_probabilities(survival_dependencies)
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 76, :], np.full(45, 2.0))
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 78, :], np.full(45, np.nan))
        np.testing.assert_equal(output_map.intensity_map_data.ena_intensity[0, 0, 80, :], np.full(45, 2.0))
