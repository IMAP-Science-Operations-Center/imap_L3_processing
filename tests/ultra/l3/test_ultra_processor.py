import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
import xarray as xr
from imap_data_access.processing_input import AncillaryInput, ScienceInput, ProcessingInputCollection
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.spatial_utils import AzElSkyGrid
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.constants import TT2000_EPOCH
from imap_l3_processing.maps.map_models import HealPixIntensityMapData, IntensityMapData, HealPixCoords, \
    HealPixSpectralIndexDataProduct, SpectralIndexMapData, RectangularIntensityDataProduct, \
    RectangularSpectralIndexDataProduct, RectangularSpectralIndexMapData, RectangularIntensityMapData
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies, UltraL3SpectralIndexDependencies
from imap_l3_processing.ultra.l3.ultra_processor import UltraProcessor
from tests.test_helpers import get_test_data_path


class TestUltraProcessor(unittest.TestCase):

    def test_process_survival_probability_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_survival_probability(degree_spacing)

    def test_process_spectral_index_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_spectral_index(degree_spacing)

    def test_process_combined_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_combined_sensor(degree_spacing)

    def test_process_combined_survival_corrected_all_spacings(self):
        for degree_spacing in [2, 4, 6]:
            with self.subTest(spacing=degree_spacing):
                self._test_process_combined_sensor_survival_probability(degree_spacing)

    @patch('imap_l3_processing.ultra.l3.ultra_processor.HealPixIntensityMapData')
    @patch('imap_l3_processing.utils.spiceypy')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbability')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3Dependencies.fetch_dependencies')
    def _test_process_survival_probability(self, degree_spacing, mock_fetch_dependencies,
                                           mock_combine_glows_l3e_with_l1c_pointing,
                                           mock_survival_probability_pointing_set, mock_survival_skymap,
                                           mock_save_data, mock_spiceypy,
                                           mock_healpix_intensity_map_data_class):
        healpix_intensity_map_data = mock_healpix_intensity_map_data_class.return_value

        rng = np.random.default_rng()
        healpix_indices = np.arange(12)
        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()

        mock_spiceypy.ktotal.return_value = 1

        fake_spice = Path("path/to/fake/spice.tls")
        mock_spiceypy.kdata.return_value = [fake_spice]

        input_l2_map = _create_ultra_l2_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)

        input_l2_map.intensity_map_data.energy = sentinel.ultra_l2_energies

        input_l2_map_name = "imap_ultra_l2_a-map-descriptor_20250601_v000.cdf"
        input_l1c_pset_name = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_glows_l3e_name = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"

        input_deps = ProcessingInputCollection(ScienceInput(input_l2_map_name))

        mock_fetch_dependencies.return_value = UltraL3Dependencies(
            ultra_l2_map=input_l2_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[Path(input_l2_map_name), Path(input_l1c_pset_name), Path(input_glows_l3e_name)],
            energy_bin_group_sizes=sentinel.bin_groups,
        )

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.ultra_l1c_1, sentinel.glows_l3e_1),
                                                                 (sentinel.ultra_l1c_2, sentinel.glows_l3e_2),
                                                                 (sentinel.ultra_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"u90-ena-h-sf-sp-full-hae-{degree_spacing}deg-6mo"
                                       )

        computed_survival_probabilities = rng.random((1, 9, healpix_indices.shape[0]))

        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset({
            "exposure_weighted_survival_probabilities": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA_L1C.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                computed_survival_probabilities
            )
        },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                CoordNames.HEALPIX_INDEX.value: healpix_indices,
            })

        mock_healpix_skymap = Mock()
        healpix_intensity_map_data.to_healpix_skymap = Mock(return_value=mock_healpix_skymap)

        observation_date_as_float = np.arange(90 * 45).reshape(90, 45) * 3600 * 1e9

        expected_converted_datetimes = TT2000_EPOCH + timedelta(hours=1) * np.arange(
            90 * 45).reshape(90, 45)

        mock_rectangular_map_dataset = {
            "obs_date": Mock(values=observation_date_as_float),
            "obs_date_range": Mock(values=sentinel.rectangular_obs_date_range),
            "exposure_factor": Mock(values=sentinel.rectangular_exposure_factor),
            "ena_intensity": Mock(values=sentinel.rectangular_ena_intensity),
            "ena_intensity_stat_uncert": Mock(values=sentinel.rectangular_ena_intensity_stat_uncert),
            "ena_intensity_sys_err": Mock(values=sentinel.rectangular_ena_intensity_sys_err),
        }

        solid_angle_computed_by_rectangular_skymap = np.array([[1, 2], [3, 4], [5, 6]])
        expected_output_solid_angle = np.array([[1, 3, 5], [2, 4, 6]])

        mock_rectangular_sky_map = Mock(spec=RectangularSkyMap)
        mock_rectangular_sky_map.sky_grid = AzElSkyGrid(degree_spacing)
        mock_rectangular_sky_map.solid_angle_grid = solid_angle_computed_by_rectangular_skymap
        mock_rectangular_sky_map.to_dataset.return_value = mock_rectangular_map_dataset
        mock_healpix_skymap.to_rectangular_skymap.return_value = mock_rectangular_sky_map, 0

        processor = UltraProcessor(input_deps, input_metadata)
        product = processor.process(SpiceFrame.IMAP_GCS)

        mock_fetch_dependencies.assert_called_once_with(input_deps)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_sp, sentinel.ultra_l1c_pset)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.ultra_l1c_1, sentinel.glows_l3e_1, bin_groups=sentinel.bin_groups),
            call(sentinel.ultra_l1c_2, sentinel.glows_l3e_2, bin_groups=sentinel.bin_groups),
            call(sentinel.ultra_l1c_3, sentinel.glows_l3e_3, bin_groups=sentinel.bin_groups)
        ])
        intensity_data = input_l2_map.intensity_map_data
        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     SpiceFrame.IMAP_GCS, input_l2_map.coords.nside)

        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        mock_healpix_intensity_map_data_class.assert_called_once()
        healpix_intensity_map_data_kwargs = mock_healpix_intensity_map_data_class.call_args_list[0].kwargs

        actual_intensity_map_data = healpix_intensity_map_data_kwargs["intensity_map_data"]

        np.testing.assert_array_equal(actual_intensity_map_data.ena_intensity,
                                      intensity_data.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(actual_intensity_map_data.ena_intensity_stat_uncert,
                                      intensity_data.ena_intensity_stat_uncert / computed_survival_probabilities)
        np.testing.assert_array_equal(actual_intensity_map_data.ena_intensity_sys_err,
                                      intensity_data.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(actual_intensity_map_data.epoch, intensity_data.epoch)
        np.testing.assert_array_equal(actual_intensity_map_data.epoch_delta, intensity_data.epoch_delta)
        np.testing.assert_array_equal(actual_intensity_map_data.energy, intensity_data.energy)
        np.testing.assert_array_equal(actual_intensity_map_data.energy_delta_plus, intensity_data.energy_delta_plus)
        np.testing.assert_array_equal(actual_intensity_map_data.energy_delta_minus,
                                      intensity_data.energy_delta_minus)
        np.testing.assert_array_equal(actual_intensity_map_data.energy_label, intensity_data.energy_label)
        np.testing.assert_array_equal(actual_intensity_map_data.latitude, intensity_data.latitude)
        np.testing.assert_array_equal(actual_intensity_map_data.longitude, intensity_data.longitude)
        np.testing.assert_array_equal(actual_intensity_map_data.exposure_factor, intensity_data.exposure_factor)
        np.testing.assert_array_equal(actual_intensity_map_data.obs_date, intensity_data.obs_date)
        np.testing.assert_array_equal(actual_intensity_map_data.obs_date_range, intensity_data.obs_date_range)
        np.testing.assert_array_equal(actual_intensity_map_data.solid_angle, intensity_data.solid_angle)

        healpix_intensity_map_data.to_healpix_skymap.assert_called_once()

        expected_value_keys = [
            "exposure_factor",
            "ena_intensity",
            "ena_intensity_stat_uncert",
            "ena_intensity_sys_err",
            "obs_date",
            "obs_date_range",
        ]

        mock_healpix_skymap.to_rectangular_skymap.assert_called_once_with(degree_spacing, expected_value_keys)

        mock_save_data.assert_called_once()
        actual_rectangular_data_product = mock_save_data.call_args_list[0].args[0]

        self.assertIsInstance(actual_rectangular_data_product, RectangularIntensityDataProduct)

        self.assertEqual(4, len(actual_rectangular_data_product.parent_file_names))
        self.assertEqual({input_l2_map_name, fake_spice.name, input_l1c_pset_name, input_glows_l3e_name},
                         set(actual_rectangular_data_product.parent_file_names))

        actual_rectangular_data = actual_rectangular_data_product.data

        self.assertIsInstance(actual_rectangular_data.intensity_map_data, IntensityMapData)

        # @formatter:off
        expected_healpix_map_data = healpix_intensity_map_data.intensity_map_data
        self.assertEqual(expected_healpix_map_data.epoch, actual_rectangular_data.intensity_map_data.epoch)
        self.assertEqual(expected_healpix_map_data.epoch_delta, actual_rectangular_data.intensity_map_data.epoch_delta)
        self.assertEqual(expected_healpix_map_data.energy, actual_rectangular_data.intensity_map_data.energy)
        self.assertEqual(expected_healpix_map_data.energy_delta_plus, actual_rectangular_data.intensity_map_data.energy_delta_plus)
        self.assertEqual(expected_healpix_map_data.energy_delta_minus, actual_rectangular_data.intensity_map_data.energy_delta_minus)
        self.assertEqual(expected_healpix_map_data.energy_label, actual_rectangular_data.intensity_map_data.energy_label)

        self.assertEqual(actual_rectangular_data.intensity_map_data.exposure_factor, sentinel.rectangular_exposure_factor)
        self.assertEqual(actual_rectangular_data.intensity_map_data.ena_intensity, sentinel.rectangular_ena_intensity)
        self.assertEqual(actual_rectangular_data.intensity_map_data.ena_intensity_stat_uncert, sentinel.rectangular_ena_intensity_stat_uncert)
        self.assertEqual(actual_rectangular_data.intensity_map_data.ena_intensity_sys_err, sentinel.rectangular_ena_intensity_sys_err)
        np.testing.assert_array_equal(actual_rectangular_data.intensity_map_data.obs_date.data, expected_converted_datetimes)
        np.testing.assert_array_equal(actual_rectangular_data.intensity_map_data.obs_date.mask, np.ma.getmask(expected_converted_datetimes))
        self.assertEqual(actual_rectangular_data.intensity_map_data.obs_date_range, sentinel.rectangular_obs_date_range)
        np.testing.assert_array_equal(actual_rectangular_data.intensity_map_data.solid_angle, expected_output_solid_angle)

        np.testing.assert_array_equal(actual_rectangular_data.intensity_map_data.latitude, mock_rectangular_sky_map.sky_grid.el_bin_midpoints)
        np.testing.assert_array_equal(actual_rectangular_data.intensity_map_data.longitude, mock_rectangular_sky_map.sky_grid.az_bin_midpoints)
        np.testing.assert_array_equal(actual_rectangular_data.coords.latitude_label, mock_rectangular_sky_map.sky_grid.el_bin_midpoints.astype(str))
        np.testing.assert_array_equal(actual_rectangular_data.coords.longitude_label, mock_rectangular_sky_map.sky_grid.az_bin_midpoints.astype(str))
        # @formatter:on

        self.assertEqual((int(360 / degree_spacing),), actual_rectangular_data.coords.longitude_delta.shape)
        self.assertTrue(np.all(degree_spacing / 2 == actual_rectangular_data.coords.longitude_delta))

        self.assertEqual((int(180 / degree_spacing),), actual_rectangular_data.coords.latitude_delta.shape)
        self.assertTrue(np.all(degree_spacing / 2 == actual_rectangular_data.coords.latitude_delta))

        self.assertEqual([mock_save_data.return_value], product)

    @patch('imap_l3_processing.ultra.l3.ultra_processor.MapProcessor.get_parent_file_names')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraProcessor._process_healpix_intensity_to_rectangular')
    @patch("imap_l3_processing.ultra.l3.ultra_processor.ExposureWeightedCombination")
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3CombinedDependencies.fetch_dependencies')
    def _test_process_combined_sensor(self, degree_spacing, mock_fetch_dependencies, mock_save_data,
                                      mock_exposure_weighted_combination,
                                      mock_healpix_to_rectangular, _):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"ulc-ena-h-sf-nsp-full-hae-{degree_spacing}deg-6mo",
                                       )

        combined_dependencies = mock_fetch_dependencies.return_value
        combined_dependencies.u90_l2_map = sentinel.u90_l2_map
        combined_dependencies.u45_l2_map = sentinel.u45_l2_map
        combined_dependencies.dependency_file_paths = [
            Path("folder/u45_map", Path("folder/u90_map"), Path("folder/u45_l1c"), Path("folder/u90_l1c"))]

        processor = UltraProcessor(sentinel.dependencies, input_metadata)
        product = processor.process(spice_frame_name=sentinel.spice_frame)

        mock_fetch_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_exposure_weighted_combination.return_value.combine_healpix_intensity_map_data.assert_called_once_with(
            [sentinel.u45_l2_map, sentinel.u90_l2_map])

        mock_combine_maps_return_value = mock_exposure_weighted_combination.return_value.combine_healpix_intensity_map_data.return_value

        mock_healpix_to_rectangular.assert_called_once_with(mock_combine_maps_return_value, degree_spacing)

        mock_save_data.assert_called_once_with(mock_healpix_to_rectangular.return_value)
        self.assertEqual([mock_save_data.return_value], product)

    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraProcessor._process_survival_probability')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraProcessor._process_healpix_intensity_to_rectangular')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.MapProcessor.get_parent_file_names')
    @patch("imap_l3_processing.ultra.l3.ultra_processor.combine_healpix_intensity_map_data")
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3CombinedDependencies.fetch_dependencies')
    def _test_process_combined_sensor_survival_probability(self, degree_spacing, mock_fetch_dependencies,
                                                           mock_save_data, mock_combine_maps,
                                                           mock_get_parent_file_names, mock_healpix_to_rectangular,
                                                           mock_process_survival_probability):
        mock_get_parent_file_names.return_value = ["ram_map", "antiram_map"]
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"ulc-ena-h-sf-sp-full-hae-{degree_spacing}deg-6mo",
                                       )
        mock_dependencies = Mock()
        mock_dependencies.u45_l2_map = sentinel.u45_l2_map
        mock_dependencies.u90_l2_map = sentinel.u90_l2_map
        mock_dependencies.u45_l1c_psets = [sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3]
        mock_dependencies.u90_l1c_psets = [sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3]
        mock_dependencies.glows_l3e_psets = [sentinel.glows_pset_1, sentinel.glows_pset_2, sentinel.glows_pset_3]
        mock_dependencies.dependency_file_paths = sentinel.dependency_file_paths
        mock_dependencies.energy_bin_group_sizes = sentinel.energy_bin_sizes
        mock_fetch_dependencies.return_value = mock_dependencies

        expected_u45_dependency = UltraL3Dependencies(
            ultra_l2_map=mock_dependencies.u45_l2_map,
            ultra_l1c_pset=mock_dependencies.u45_l1c_psets,
            glows_l3e_sp=mock_dependencies.glows_l3e_psets,
            dependency_file_paths=mock_dependencies.dependency_file_paths,
            energy_bin_group_sizes=mock_dependencies.energy_bin_group_sizes,
        )

        expected_u90_dependency = UltraL3Dependencies(
            ultra_l2_map=mock_dependencies.u90_l2_map,
            ultra_l1c_pset=mock_dependencies.u90_l1c_psets,
            glows_l3e_sp=mock_dependencies.glows_l3e_psets,
            dependency_file_paths=mock_dependencies.dependency_file_paths,
            energy_bin_group_sizes=mock_dependencies.energy_bin_group_sizes,
        )

        mock_process_survival_probability.side_effect = [sentinel.u45_l2_survival_corrected_map,
                                                         sentinel.u90_l2_survival_corrected_map]

        processor = UltraProcessor(sentinel.dependencies, input_metadata)
        product = processor.process(spice_frame_name=sentinel.spice_frame)

        mock_fetch_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_combine_maps.assert_called_once_with(
            [sentinel.u45_l2_survival_corrected_map, sentinel.u90_l2_survival_corrected_map])

        mock_healpix_to_rectangular.assert_called_once_with(mock_combine_maps.return_value, degree_spacing)

        mock_save_data.assert_called_once_with(mock_healpix_to_rectangular.return_value)
        self.assertEqual([mock_save_data.return_value], product)

        mock_process_survival_probability.assert_has_calls([
            call(expected_u45_dependency, sentinel.spice_frame),
            call(expected_u90_dependency, sentinel.spice_frame)
        ])

    @patch('imap_l3_processing.ultra.l3.ultra_processor.HealPixIntensityMapData')
    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbability')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3Dependencies.fetch_dependencies')
    def test_defaults_to_ECLIPJ2000_spice_frame(self, mock_fetch_dependencies,
                                                mock_combine_glows_l3e_with_l1c_pointing,
                                                mock_survival_probability_pointing_set, mock_survival_skymap,
                                                mock_save_data, mock_spiceypy,
                                                mock_healpix_intensity_map_data_class):
        healpix_intensity_map_data = mock_healpix_intensity_map_data_class.return_value

        rng = np.random.default_rng()
        healpix_indices = np.arange(12)
        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()

        mock_spiceypy.ktotal.return_value = 1

        fake_spice = Path("path/to/fake/spice.tls")
        mock_spiceypy.kdata.return_value = [fake_spice]

        input_l2_map = _create_ultra_l2_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)

        input_l2_map.intensity_map_data.energy = sentinel.ultra_l2_energies

        input_l2_map_name = "imap_ultra_l2_a-map-descriptor_20250601_v000.cdf"
        input_l1c_pset_name = "imap_ultra_l1c_a-pset-descriptor_20250601_v000.cdf"
        input_glows_l3e_name = "imap_glows_l3e_a-glows-descriptor_20250601_v000.cdf"

        input_deps = ProcessingInputCollection(ScienceInput(input_l2_map_name))

        mock_fetch_dependencies.return_value = UltraL3Dependencies(
            ultra_l2_map=input_l2_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp,
            dependency_file_paths=[Path(input_l2_map_name), Path(input_l1c_pset_name), Path(input_glows_l3e_name)],
            energy_bin_group_sizes=None
        )

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"u90-ena-h-sf-sp-full-hae-2deg-6mo"
                                       )

        computed_survival_probabilities = rng.random((1, 9, healpix_indices.shape[0]))

        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset({
            "exposure_weighted_survival_probabilities": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA_L1C.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                computed_survival_probabilities
            )
        },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY_ULTRA_L1C.value: rng.random((9,)),
                CoordNames.HEALPIX_INDEX.value: healpix_indices,
            })

        mock_healpix_skymap = Mock()
        healpix_intensity_map_data.to_healpix_skymap = Mock(return_value=mock_healpix_skymap)

        observation_date_as_float = np.arange(90 * 45).reshape(90, 45) * 3600 * 1e9

        mock_rectangular_map_dataset = {
            "obs_date": Mock(values=observation_date_as_float),
            "obs_date_range": Mock(values=sentinel.rectangular_obs_date_range),
            "exposure_factor": Mock(values=sentinel.rectangular_exposure_factor),
            "ena_intensity": Mock(values=sentinel.rectangular_ena_intensity),
            "ena_intensity_stat_uncert": Mock(values=sentinel.rectangular_ena_intensity_stat_unc),
            "ena_intensity_sys_err": Mock(values=sentinel.rectangular_ena_intensity_sys_err),
        }

        solid_angle_computed_by_rectangular_skymap = np.array([[1, 2], [3, 4], [5, 6]])

        mock_rectangular_sky_map = Mock(spec=RectangularSkyMap)
        mock_rectangular_sky_map.sky_grid = AzElSkyGrid(2)
        mock_rectangular_sky_map.solid_angle_grid = solid_angle_computed_by_rectangular_skymap
        mock_rectangular_sky_map.to_dataset.return_value = mock_rectangular_map_dataset
        mock_healpix_skymap.to_rectangular_skymap.return_value = mock_rectangular_sky_map, 0

        processor = UltraProcessor(input_deps, input_metadata)
        processor.process()

        mock_survival_skymap.assert_called_once_with([], SpiceFrame.ECLIPJ2000, input_l2_map.coords.nside)

    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.calculate_spectral_index_for_multiple_ranges')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def _test_process_spectral_index(self, degree_spacing,
                                     mock_fetch_dependencies,
                                     mock_calculate_spectral_index, mock_save_data,
                                     mock_spiceypy):

        mock_spiceypy.ktotal.return_value = 0

        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        energy_range_file_name = 'imap_ultra_energy-range-descriptor_20250601_v000.dat'
        input_deps = ProcessingInputCollection(ScienceInput(map_file_name), AncillaryInput(energy_range_file_name))

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-{degree_spacing}deg-6mo")
        input_map_data = RectangularIntensityMapData(Mock(), Mock())
        dependencies = UltraL3SpectralIndexDependencies(input_map_data, sentinel.energy_ranges)
        mock_fetch_dependencies.return_value = dependencies

        mock_spectral_index_map_data = Mock(spec=SpectralIndexMapData)
        mock_calculate_spectral_index.return_value = mock_spectral_index_map_data

        expected_parent_file_names = [map_file_name, energy_range_file_name]

        processor = UltraProcessor(input_deps, input_metadata)
        product = processor.process()

        mock_save_data.assert_called_once()
        actual_rectangular_data_product = mock_save_data.call_args_list[0].args[0]
        self.assertIsInstance(actual_rectangular_data_product, RectangularSpectralIndexDataProduct)
        self.assertEqual(expected_parent_file_names, actual_rectangular_data_product.parent_file_names)
        self.assertEqual(processor.input_metadata, actual_rectangular_data_product.input_metadata)

        self.assertIsInstance(actual_rectangular_data_product, RectangularSpectralIndexDataProduct)

        actual_rectangular_data: RectangularSpectralIndexMapData = actual_rectangular_data_product.data
        self.assertIsInstance(actual_rectangular_data_product.data, RectangularSpectralIndexMapData)
        self.assertIs(mock_spectral_index_map_data, actual_rectangular_data.spectral_index_map_data)
        self.assertIs(input_map_data.coords, actual_rectangular_data.coords)

        mock_fetch_dependencies.assert_called_once_with(input_deps)
        mock_calculate_spectral_index.assert_called_once_with(dependencies.map_data.intensity_map_data,
                                                              sentinel.energy_ranges)
        self.assertEqual([mock_save_data.return_value], product)

    def test_process_raises_exception_when_generating_a_healpix_map(self):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-nside8-6mo")

        processor = UltraProcessor(ProcessingInputCollection(), input_metadata)

        with self.assertRaises(NotImplementedError):
            processor.process()

    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def test_process_spectral_index_validating_output_values(self, mock_fetch_dependencies, mock_save_data):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-6deg-6mo")
        input_map_path = get_test_data_path('ultra/fake_ultra_map_data_with_breakpoint_at_15keV.cdf')
        fit_energy_ranges_path = get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat')
        dependencies = UltraL3SpectralIndexDependencies.from_file_paths(input_map_path, fit_energy_ranges_path)
        mock_fetch_dependencies.return_value = dependencies

        expected_ena_spectral_index = np.array([2] * (60 * 30) + [3.5] * (60 * 30)).reshape(1, 2, 60, 30)

        processing_input_collection = ProcessingInputCollection()
        processor = UltraProcessor(processing_input_collection, input_metadata)
        product = processor.process()

        actual_data_product: HealPixSpectralIndexDataProduct = mock_save_data.call_args[0][0]

        np.testing.assert_array_almost_equal(actual_data_product.data.spectral_index_map_data.ena_spectral_index,
                                             expected_ena_spectral_index)
        self.assertEqual([mock_save_data.return_value], product)


def _create_ultra_l2_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None,
                          intensity_stat_unc=None, healpix_indices=None) -> HealPixIntensityMapData:
    epoch = epoch if epoch is not None else np.array([datetime.now()])
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    healpix_indices = healpix_indices if healpix_indices is not None else np.arange(12)
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(healpix_indices)), fill_value=1)
    intensity_stat_uncert = intensity_stat_unc if intensity_stat_unc is not None else np.full(
        flux.shape,
        fill_value=1)

    if isinstance(flux, np.ndarray):
        more_real_flux = flux
    else:
        more_real_flux = np.full((len(epoch), len(lon), len(lat), 9), fill_value=1)

    return HealPixIntensityMapData(
        IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([0]),
            energy=energy,
            energy_delta_plus=energy_delta,
            energy_delta_minus=energy_delta,
            energy_label=np.array(["energy"]),
            latitude=lat,
            longitude=lon,
            exposure_factor=np.full_like(flux, 0),
            obs_date=np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1)),
            obs_date_range=np.full_like(more_real_flux, 0),
            solid_angle=np.full_like(more_real_flux, 0),
            ena_intensity=flux,
            ena_intensity_stat_uncert=intensity_stat_uncert,
            ena_intensity_sys_err=np.full_like(flux, 0),
        ),
        HealPixCoords(
            pixel_index=healpix_indices,
            pixel_index_label=np.full(healpix_indices.shape, "healpix index label")
        )
    )
