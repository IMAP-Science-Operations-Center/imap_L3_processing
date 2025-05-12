import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, call, sentinel, MagicMock

import numpy as np
import xarray as xr
from imap_data_access.processing_input import ProcessingInputCollection
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing import spice_wrapper
from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.hi.l3.models import HiL3IntensityDataProduct, \
    HiIntensityMapData, HiL1cData, HiGlowsL3eData
from imap_l3_processing.hi.l3.utils import PixelSize, MapDescriptorParts, parse_map_descriptor
from imap_l3_processing.map_models import RectangularSpectralIndexDataProduct
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path


class TestHiProcessor(unittest.TestCase):
    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.hi.hi_processor.upload')
    @patch('imap_l3_processing.hi.hi_processor.HiL3SpectralFitDependencies.fetch_dependencies')
    @patch('imap_l3_processing.hi.hi_processor.spectral_fit')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    def test_process_spectral_fit(self, mock_save_data, mock_spectral_fit,
                                  mock_fetch_dependencies, mock_upload,
                                  mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l2_or_l3_map"]
        lat = np.array([0, 45])
        long = np.array([0, 45, 90])
        energy = np.array([1, 15]) + 0.5
        energy_delta = np.array([0.5, 0.5])
        epoch = np.array([datetime.now()])
        flux = np.full((1, 2, 3, 2), 1)
        intensity_stat_unc = 5

        hi_l3_data = _create_h1_l3_data(lat=lat, lon=long, energy=energy, epoch=epoch, flux=flux,
                                        intensity_stat_unc=intensity_stat_unc,
                                        energy_delta=energy_delta)
        hi_l3_data.exposure_factor = np.full_like(flux, 1)
        dependencies = HiL3SpectralFitDependencies(hi_l3_data=hi_l3_data)

        upstream_dependencies = ProcessingInputCollection()

        mock_fetch_dependencies.return_value = dependencies

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="h90-spx-h-hf-sp-full-hae-4deg-6mo",
                                       )

        expected_unc = np.full((1, 1, 3, 2), 1)
        expected_index = np.full_like(expected_unc, 2)
        mock_spectral_fit.return_value = expected_index, expected_unc
        processor = HiProcessor(upstream_dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(upstream_dependencies)
        mock_spectral_fit.assert_called_once_with(hi_l3_data.ena_intensity,
                                                  np.square(hi_l3_data.ena_intensity_stat_unc),
                                                  hi_l3_data.energy)

        mock_save_data.assert_called_once()
        actual_hi_data_product = mock_save_data.call_args_list[0].args[0]
        self.assertIsInstance(actual_hi_data_product, RectangularSpectralIndexDataProduct)
        self.assertEqual(input_metadata, actual_hi_data_product.input_metadata)
        actual_data = actual_hi_data_product.data
        spectral_index_data = actual_data.spectral_index_map_data
        coords = actual_data.coords
        expected_energy_delta_minus = np.array([3])
        expected_energy_delta_plus = np.array([12])
        expected_energy = np.array([4])
        expected_exposure = np.full((1, 1, 3, 2), 2)
        np.testing.assert_array_equal(expected_energy_delta_minus, spectral_index_data.energy_delta_minus)
        np.testing.assert_array_equal(expected_energy_delta_plus, spectral_index_data.energy_delta_plus)
        np.testing.assert_array_equal(expected_index, spectral_index_data.ena_spectral_index)
        np.testing.assert_array_equal(expected_unc,
                                      spectral_index_data.ena_spectral_index_stat_unc)
        np.testing.assert_array_equal(spectral_index_data.energy, expected_energy)
        np.testing.assert_array_equal(spectral_index_data.latitude, hi_l3_data.latitude)
        np.testing.assert_array_equal(spectral_index_data.longitude, hi_l3_data.longitude)
        np.testing.assert_array_equal(spectral_index_data.epoch, hi_l3_data.epoch)
        np.testing.assert_array_equal(spectral_index_data.exposure_factor, expected_exposure)
        self.assertEqual(["l2_or_l3_map"], actual_hi_data_product.parent_file_names)
        np.testing.assert_array_equal(coords.latitude_delta, hi_l3_data.latitude_delta)
        np.testing.assert_array_equal(coords.longitude_delta, hi_l3_data.longitude_delta)
        np.testing.assert_array_equal(coords.latitude_label, hi_l3_data.latitude_label)
        np.testing.assert_array_equal(coords.longitude_label, hi_l3_data.longitude_label)

        mock_upload.assert_called_once_with(mock_save_data.return_value)

    def test_spectral_fit_against_validation_data(self):
        test_cases = [
            ("hi45", "hi/fake_l2_maps/hi45-6months.cdf", "hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi90", "hi/fake_l2_maps/hi90-6months.cdf", "hi/validation/IMAP-Hi90_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/IMAP-Hi90_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi45-zirnstein-mondel", "hi/fake_l2_maps/hi45-zirnstein-mondel-6months.cdf",
             "hi/validation/IMAP-Hi45_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/IMAP-Hi45_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi90-zirnstein-mondel", "hi/fake_l2_maps/hi90-zirnstein-mondel-6months.cdf",
             "hi/validation/IMAP-Hi90_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/IMAP-Hi90_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam_sig.csv"),
        ]

        for name, input_file_path, expected_gamma_path, expected_sigma_path in test_cases:
            with self.subTest(name):
                dependencies = HiL3SpectralFitDependencies.from_file_paths(
                    get_test_data_path(input_file_path)
                )

                expected_gamma = np.loadtxt(get_test_data_path(expected_gamma_path), delimiter=",", dtype=str).T
                expected_gamma[expected_gamma == "NaN"] = "-1"
                expected_gamma = expected_gamma.astype(np.float64)
                expected_gamma[expected_gamma == -1] = np.nan

                expected_gamma_sigma = np.loadtxt(get_test_data_path(expected_sigma_path), delimiter=",",
                                                  dtype=str).T
                expected_gamma_sigma[expected_gamma_sigma == "NaN"] = "-1"
                expected_gamma_sigma = expected_gamma_sigma.astype(np.float64)
                expected_gamma_sigma[expected_gamma_sigma == -1] = np.nan

                input_metadata = InputMetadata(instrument="hi",
                                               data_level="l3",
                                               start_date=datetime.now(),
                                               end_date=datetime.now() + timedelta(days=1),
                                               version="",
                                               descriptor="spectral-fit-index",
                                               )
                processor = HiProcessor(None, input_metadata)
                output_data = processor.process_spectral_fit_index(dependencies)

                np.testing.assert_allclose(output_data.spectral_index_map_data.ena_spectral_index[0, 0],
                                           expected_gamma, atol=1e-3)
                np.testing.assert_allclose(output_data.spectral_index_map_data.ena_spectral_index_stat_unc[0, 0],
                                           expected_gamma_sigma, atol=1e-3)

    def test_spectral_fit_other_fields(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])
        lat = np.arange(-90, 90, 45)
        lon = np.arange(0, 360, 45)

        input_map = _create_h1_l3_data(energy=input_energies, energy_delta=input_deltas, lat=lat, lon=lon)
        input_map.obs_date[0, 0] = datetime(2025, 1, 1)
        input_map.obs_date[0, 1] = datetime(2025, 1, 1)
        input_map.obs_date[0, 2] = datetime(2027, 1, 1)

        input_map.exposure_factor[0, 0] = 1.0
        input_map.exposure_factor[0, 1] = 2.0
        input_map.exposure_factor[0, 2] = 3.0

        input_map.obs_date_range[0, 0] = 1
        input_map.obs_date_range[0, 1] = 1
        input_map.obs_date_range[0, 2] = 3

        processor = HiProcessor(sentinel.input_metadata, sentinel.dependencies)

        dependencies = HiL3SpectralFitDependencies(input_map)

        output = processor.process_spectral_fit_index(dependencies)

        self.assertEqual(output.spectral_index_map_data.energy.shape[0], 1)
        self.assertEqual(output.spectral_index_map_data.energy[0], 10)
        np.testing.assert_allclose(output.spectral_index_map_data.energy_delta_minus, np.array([9]))
        np.testing.assert_allclose(output.spectral_index_map_data.energy_delta_plus, np.array([90]))
        self.assertEqual(output.spectral_index_map_data.energy_label.shape[0], 1)
        self.assertEqual("1.0 - 100.0 keV", output.spectral_index_map_data.energy_label[0])

        expected_ena_shape = np.array([1, 1, len(lon), len(lat)])
        np.testing.assert_array_almost_equal(output.spectral_index_map_data.ena_spectral_index,
                                             np.zeros(expected_ena_shape))
        np.testing.assert_array_equal(output.spectral_index_map_data.ena_spectral_index_stat_unc,
                                      np.zeros(expected_ena_shape))
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date,
                                      np.full(expected_ena_shape, datetime(2026, 1, 1)))
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date_range, np.full(expected_ena_shape, 2))
        np.testing.assert_array_equal(output.spectral_index_map_data.exposure_factor, np.full(expected_ena_shape, 6))

    def test_spectral_fit_other_fields_with_one_zero_exposure(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])
        lat = np.arange(-90, 90, 45)
        lon = np.arange(0, 360, 45)

        input_map = _create_h1_l3_data(energy=input_energies, energy_delta=input_deltas, lat=lat, lon=lon)
        input_map.obs_date[0, 0] = datetime(2025, 1, 1)
        input_map.obs_date[0, 1] = datetime(2025, 1, 1)
        input_map.obs_date[0, 2] = datetime(2027, 1, 1)
        input_map.obs_date.mask = np.ma.getmaskarray(input_map.obs_date)
        input_map.obs_date.mask[0, 1] = True

        input_map.exposure_factor[0, 0] = 1.0
        input_map.exposure_factor[0, 1] = 0.0
        input_map.exposure_factor[0, 2] = 1.0

        input_map.obs_date_range[0, 0] = 1
        input_map.obs_date_range[0, 1] = 1
        input_map.obs_date_range[0, 2] = 3
        input_map.obs_date_range.mask = np.ma.getmaskarray(input_map.obs_date_range)
        input_map.obs_date_range.mask[0, 1] = True

        processor = HiProcessor(sentinel.input_metadata, sentinel.dependencies)

        dependencies = HiL3SpectralFitDependencies(input_map)

        output = processor.process_spectral_fit_index(dependencies)

        expected_ena_shape = np.array([1, 1, len(lon), len(lat)])
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date,
                                      np.full(expected_ena_shape, datetime(2026, 1, 1)), strict=True)
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date_range, np.full(expected_ena_shape, 2.0),
                                      strict=True)

    def test_spectral_fit_other_fields_with_all_zero_exposure(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])
        lat = np.arange(-90, 90, 45)
        lon = np.arange(0, 360, 45)

        input_map = _create_h1_l3_data(energy=input_energies, energy_delta=input_deltas, lat=lat, lon=lon)
        input_map.obs_date.mask = np.ma.getmaskarray(input_map.obs_date)
        input_map.obs_date.mask[:] = True

        input_map.exposure_factor[:] = 0.0

        input_map.obs_date_range.mask = np.ma.getmaskarray(input_map.obs_date_range)
        input_map.obs_date_range.mask[:] = True

        processor = HiProcessor(sentinel.input_metadata, sentinel.dependencies)

        dependencies = HiL3SpectralFitDependencies(input_map)

        output = processor.process_spectral_fit_index(dependencies)

        expected_ena_shape = np.array([1, 1, len(lon), len(lat)])

        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date.mask, np.full(expected_ena_shape, True),
                                      strict=True)
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date_range.mask,
                                      np.full(expected_ena_shape, True), strict=True)

    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.hi.hi_processor.upload')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    @patch('imap_l3_processing.hi.hi_processor.HiSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.hi.hi_processor.HiSurvivalProbabilityPointingSet')
    @patch('imap_l3_processing.hi.hi_processor.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.hi.hi_processor.HiL3SurvivalDependencies.fetch_dependencies')
    def test_process_survival_probability(self, mock_fetch_dependencies, mock_combine_glows_l3e_with_l1c_pointing,
                                          mock_survival_probability_pointing_set, mock_survival_skymap, mock_save_data,
                                          mock_upload, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l2_map", "spice1"]
        rng = np.random.default_rng()
        input_map_flux = rng.random((1, 9, 90, 45))
        epoch = datetime.now()

        input_map: HiIntensityMapData = _create_h1_l3_data(epoch=[epoch], flux=input_map_flux)

        input_map.energy = sentinel.hi_l2_energies

        mock_l2_grid_size = MagicMock(spec=PixelSize.FourDegrees)

        expected_grid_size = int(rng.integers(0, 10000000000))
        mock_l2_grid_size.__int__.return_value = expected_grid_size

        l2_descriptor_parts = MapDescriptorParts(sentinel.l2_sensor, sentinel.l2_cg, sentinel.l2_survival,
                                                 sentinel.l2_spin, mock_l2_grid_size, sentinel.l2_duration,
                                                 sentinel.l2_quantity)
        mock_fetch_dependencies.return_value = HiL3SurvivalDependencies(l2_data=input_map,
                                                                        hi_l1c_data=sentinel.hi_l1c_data,
                                                                        glows_l3e_data=sentinel.glows_l3e_data,
                                                                        l2_map_descriptor_parts=l2_descriptor_parts,
                                                                        dependency_file_paths=[Path("foo/l2_map"),
                                                                                               Path("foo/l1c_map")], )

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.hi_l1c_1, sentinel.glows_l3e_1),
                                                                 (sentinel.hi_l1c_2, sentinel.glows_l3e_2),
                                                                 (sentinel.hi_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="h90-ena-h-sf-sp-ram-hae-4deg-6mo",
                                       )

        computed_survival_probabilities = rng.random((1, 9, 90, 45))
        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset({
            "exposure_weighted_survival_probabilities": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L2.value,
                    CoordNames.ELEVATION_L2.value,
                ],
                computed_survival_probabilities
            )
        },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY.value: rng.random((9,)),
                CoordNames.AZIMUTH_L2.value: rng.random((90,)),
                CoordNames.ELEVATION_L2.value: rng.random((45,)),
            })

        processor = HiProcessor(sentinel.dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_data, sentinel.hi_l1c_data)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.hi_l1c_1, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_1,
                 sentinel.hi_l2_energies),
            call(sentinel.hi_l1c_2, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_2,
                 sentinel.hi_l2_energies),
            call(sentinel.hi_l1c_3, sentinel.l2_sensor, sentinel.l2_spin, sentinel.glows_l3e_3, sentinel.hi_l2_energies)
        ])

        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     expected_grid_size,
                                                     SpiceFrame.ECLIPJ2000)

        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        mock_save_data.assert_called_once()
        survival_data_product: HiL3IntensityDataProduct = mock_save_data.call_args_list[0].args[0]
        survival_data: HiIntensityMapData = survival_data_product.data

        self.assertEqual(input_metadata, survival_data_product.input_metadata)

        np.testing.assert_array_equal(survival_data.ena_intensity,
                                      input_map.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data.ena_intensity_stat_unc,
                                      input_map.ena_intensity_stat_unc / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data.ena_intensity_sys_err,
                                      input_map.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data.epoch, input_map.epoch)
        np.testing.assert_array_equal(survival_data.epoch_delta, input_map.epoch_delta)
        np.testing.assert_array_equal(survival_data.energy, input_map.energy)
        np.testing.assert_array_equal(survival_data.energy_delta_plus, input_map.energy_delta_plus)
        np.testing.assert_array_equal(survival_data.energy_delta_minus, input_map.energy_delta_minus)
        np.testing.assert_array_equal(survival_data.energy_label, input_map.energy_label)
        np.testing.assert_array_equal(survival_data.latitude, input_map.latitude)
        np.testing.assert_array_equal(survival_data.latitude_delta, input_map.latitude_delta)
        np.testing.assert_array_equal(survival_data.latitude_label, input_map.latitude_label)
        np.testing.assert_array_equal(survival_data.longitude, input_map.longitude)
        np.testing.assert_array_equal(survival_data.longitude_delta, input_map.longitude_delta)
        np.testing.assert_array_equal(survival_data.longitude_label, input_map.longitude_label)
        np.testing.assert_array_equal(survival_data.exposure_factor, input_map.exposure_factor)
        np.testing.assert_array_equal(survival_data.obs_date, input_map.obs_date)
        np.testing.assert_array_equal(survival_data.obs_date_range, input_map.obs_date_range)
        np.testing.assert_array_equal(survival_data.solid_angle, input_map.solid_angle)

        self.assertEqual(["l1c_map", "l2_map", "spice1"], survival_data_product.parent_file_names)

        mock_upload.assert_called_once_with(mock_save_data.return_value)

    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch("imap_l3_processing.hi.hi_processor.HiL3SingleSensorFullSpinDependencies.fetch_dependencies")
    @patch("imap_l3_processing.hi.hi_processor.HiProcessor.process_survival_probabilities")
    @patch("imap_l3_processing.hi.hi_processor.combine_maps")
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    @patch('imap_l3_processing.hi.hi_processor.upload')
    def test_process_full_spin_single_sensor_map(self, mock_upload, mock_save_data, mock_combine_maps,
                                                 mock_process_survival_prob,
                                                 mock_fetch_full_spin_single_sensor_dependencies,
                                                 mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["ram_map", "antiram_map"]
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="h90-ena-h-sf-sp-full-hae-4deg-6mo",
                                       )

        full_spin_dependencies: HiL3SingleSensorFullSpinDependencies = mock_fetch_full_spin_single_sensor_dependencies.return_value
        full_spin_dependencies.dependency_file_paths = [
            Path("folder/ram_map", Path("folder/antiram_map"), Path("folder/l1c"))]

        mock_process_survival_prob.side_effect = [
            sentinel.survival_corrected_ram,
            sentinel.survival_corrected_antiram,
        ]

        processor = HiProcessor(sentinel.dependencies, input_metadata)
        processor.process()

        mock_fetch_full_spin_single_sensor_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_process_survival_prob.assert_has_calls([
            call(full_spin_dependencies.ram_dependencies),
            call(full_spin_dependencies.antiram_dependencies)
        ])

        mock_combine_maps.assert_called_once_with([
            sentinel.survival_corrected_ram,
            sentinel.survival_corrected_antiram,
        ])

        mock_save_data.assert_called_once_with(HiL3IntensityDataProduct(
            input_metadata=input_metadata,
            parent_file_names=["antiram_map", "l1c", "ram_map"],
            data=mock_combine_maps.return_value))
        mock_upload.assert_called_once_with(mock_save_data.return_value)

    @patch('imap_l3_processing.hi.hi_processor.combine_maps')
    @patch('imap_l3_processing.hi.hi_processor.upload')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch("imap_l3_processing.hi.hi_processor.HiL3CombinedMapDependencies.fetch_dependencies")
    def test_process_combined_sensor_calls_with_correct_descriptor(self, mock_fetch_dependencies,
                                                                   mock_get_parent_file_names,
                                                                   mock_save_data,
                                                                   mock_upload,
                                                                   mock_combine_maps):
        valid_descriptors = [
            "hic-ena-h-hf-sp-full-hae-4deg-1yr",
            "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
            "hic-ena-h-hf-sp-full-hae-6deg-1yr",
            "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
        ]

        mock_get_parent_file_names.return_value = ["spice_file"]

        for descriptor in valid_descriptors:
            with self.subTest(descriptor=descriptor):
                input_metadata = InputMetadata(instrument="hi",
                                               data_level="l3",
                                               start_date=datetime.now(),
                                               end_date=datetime.now() + timedelta(days=1),
                                               version="",
                                               descriptor=descriptor,
                                               )

                processor = HiProcessor(sentinel.dependencies, input_metadata)
                processor.process()

                mock_fetch_dependencies.assert_called_with(sentinel.dependencies)
                mock_upload.assert_called_with(mock_save_data.return_value)

                mock_combine_maps.assert_called_with(mock_fetch_dependencies.return_value.maps)
                mock_save_data.assert_called_with(
                    HiL3IntensityDataProduct(
                        data=mock_combine_maps.return_value,
                        input_metadata=input_metadata,
                        parent_file_names=["spice_file"],
                    ),
                )

    def test_raises_error_for_currently_unimplemented_maps(self):
        cases = [
            "hic-ena-h-sf-nsp-full-hae-6deg-6mo",
            "hic-ena-h-sf-sp-full-hae-6deg-6mo",
            "hic-ena-h-sf-sp-ram-hae-6deg-6mo",
            "h45-ena-h-hf-nsp-full-hae-6deg-6mo",
        ]
        for descriptor in cases:
            with self.subTest(descriptor):
                input_metadata = InputMetadata(instrument="hi",
                                               data_level="l3",
                                               start_date=datetime.now(),
                                               end_date=datetime.now() + timedelta(days=1),
                                               version="",
                                               descriptor=descriptor,
                                               )

                processor = HiProcessor(ProcessingInputCollection(), input_metadata)
                with self.assertRaises(NotImplementedError):
                    processor.process()

    def test_integration_uses_fill_values_for_missing_l3e_data(self):
        t1 = datetime(2025, 4, 29, 12)
        t2 = datetime(2025, 5, 7, 12)
        t3 = datetime(2025, 5, 8, 12)
        t4 = datetime(2025, 5, 15, 12)
        l1c_psets = [create_l1c_pset(t1), create_l1c_pset(t2), create_l1c_pset(t3), create_l1c_pset(t4)]
        l3e_psets = [create_l3e_pset(t1), create_l3e_pset(t3), create_l3e_pset(t4)]
        l2_intensity_map = _create_h1_l3_data()

        descriptor = parse_map_descriptor("h90-ena-h-sf-nsp-ram-hae-4deg-3mo")
        survival_dependencies = HiL3SurvivalDependencies(l2_intensity_map, l1c_psets, l3e_psets, descriptor)

        processor = HiProcessor(sentinel.dependencies, sentinel.input_metadata)
        output_map = processor.process_survival_probabilities(survival_dependencies)
        np.testing.assert_equal(output_map.ena_intensity[0, 0, 76, :], np.full(45, 2.0))
        np.testing.assert_equal(output_map.ena_intensity[0, 0, 78, :], np.full(45, np.nan))
        np.testing.assert_equal(output_map.ena_intensity[0, 0, 80, :], np.full(45, 2.0))


def create_l1c_pset(epoch: datetime) -> HiL1cData:
    epoch_j2000 = np.array([spice_wrapper.spice.datetime2et(epoch)]) * 1e9
    energy_steps = np.array([1])
    exposures = np.full(shape=(1, energy_steps.shape[0], 3600), fill_value=1.)
    return HiL1cData(epoch, epoch_j2000, exposures, energy_steps)


def create_l3e_pset(epoch) -> HiGlowsL3eData:
    energy_steps = np.array([0.5, 5.0, 12.0])
    spin_angle = np.arange(0, 360)
    sp = np.full(shape=(1, len(energy_steps), len(spin_angle)), fill_value=0.5)
    return HiGlowsL3eData(epoch, energy_steps, spin_angle, sp)


def _create_h1_l3_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None,
                       intensity_stat_unc=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(lon), len(lat)), fill_value=1)
    intensity_stat_unc = intensity_stat_unc if intensity_stat_unc is not None else np.full(
        (len(epoch), len(energy), len(lon), len(lat)),
        fill_value=1)

    if isinstance(flux, np.ndarray):
        more_real_flux = flux
    else:
        more_real_flux = np.full((len(epoch), 9, len(lon), len(lat)), fill_value=1)

    return HiIntensityMapData(
        epoch=epoch,
        epoch_delta=np.ma.array([0]),
        energy=energy,
        energy_delta_plus=energy_delta,
        energy_delta_minus=energy_delta,
        energy_label=np.array(["energy"]),
        latitude=lat,
        latitude_delta=np.full_like(lat, 0),
        latitude_label=lat.astype(str),
        longitude=lon,
        longitude_delta=np.full_like(lon, 0),
        longitude_label=lon.astype(str),
        exposure_factor=np.full_like(flux, 0),
        obs_date=np.ma.array(np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1))),
        obs_date_range=np.ma.array(np.full_like(more_real_flux, 0)),
        solid_angle=np.full_like(more_real_flux, 0),
        ena_intensity=flux,
        ena_intensity_stat_unc=intensity_stat_unc,
        ena_intensity_sys_err=np.full_like(flux, 0),
    )
