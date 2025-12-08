import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, call, sentinel, Mock

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralIndexDependencies
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiL3SingleSensorFullSpinDependencies, \
    HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_models import RectangularSpectralIndexDataProduct, RectangularIntensityDataProduct
from imap_l3_processing.models import InputMetadata, Instrument
from tests.maps.test_builders import create_rectangular_intensity_map_data
from tests.test_helpers import get_test_data_path, run_periodically


class TestHiProcessor(unittest.TestCase):
    @patch('imap_l3_processing.hi.hi_processor.MapProcessor.get_parent_file_names')
    @patch('imap_l3_processing.hi.hi_processor.HiL3SpectralIndexDependencies.fetch_dependencies')
    @patch('imap_l3_processing.maps.spectral_fit.fit_arrays_to_power_law')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    def test_process_spectral_fit(self, mock_save_data, mock_spectral_fit,
                                  mock_fetch_dependencies,
                                  mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l2_or_l3_map"]
        lat = np.array([0, 45])
        long = np.array([0, 45, 90])
        energy = np.array([1, 15]) + 0.5
        energy_delta = np.array([0.5, 0.5])
        epoch = np.array([datetime.now()])
        flux = np.full((1, 2, 3, 2), 1)
        intensity_stat_unc = 5

        hi_l3_data = create_rectangular_intensity_map_data(lat=lat, lon=long, energy=energy, epoch=epoch, flux=flux,
                                                           intensity_stat_uncert=intensity_stat_unc,
                                                           energy_delta=energy_delta)
        intensity_data = hi_l3_data.intensity_map_data
        intensity_data.exposure_factor = np.full_like(flux, 1)
        dependencies = HiL3SpectralIndexDependencies(map_data=hi_l3_data)

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
        product = processor.process()

        mock_fetch_dependencies.assert_called_with(upstream_dependencies)
        mock_spectral_fit.assert_called_once_with(intensity_data.ena_intensity,
                                                  intensity_data.ena_intensity_stat_uncert,
                                                  intensity_data.energy)

        mock_save_data.assert_called_once()
        actual_hi_data_product = mock_save_data.call_args_list[0].args[0]
        self.assertIsInstance(actual_hi_data_product, RectangularSpectralIndexDataProduct)
        self.assertEqual(input_metadata, actual_hi_data_product.input_metadata)
        actual_data = actual_hi_data_product.data
        spectral_index_data = actual_data.spectral_index_map_data
        spectral_index_coords = actual_data.coords
        expected_energy_delta_minus = np.array([3])
        expected_energy_delta_plus = np.array([12])
        expected_energy = np.array([4])
        expected_exposure = np.full((1, 1, 3, 2), 2)
        np.testing.assert_array_equal(expected_energy_delta_minus, spectral_index_data.energy_delta_minus)
        np.testing.assert_array_equal(expected_energy_delta_plus, spectral_index_data.energy_delta_plus)
        np.testing.assert_array_equal(expected_index, spectral_index_data.ena_spectral_index)
        np.testing.assert_array_equal(expected_unc,
                                      spectral_index_data.ena_spectral_index_stat_uncert)
        np.testing.assert_array_equal(spectral_index_data.energy, expected_energy)
        np.testing.assert_array_equal(spectral_index_data.latitude, intensity_data.latitude)
        np.testing.assert_array_equal(spectral_index_data.longitude, intensity_data.longitude)
        np.testing.assert_array_equal(spectral_index_data.epoch, intensity_data.epoch)
        np.testing.assert_array_equal(spectral_index_data.exposure_factor, expected_exposure)
        self.assertEqual(["l2_or_l3_map"], actual_hi_data_product.parent_file_names)
        self.assertEqual(hi_l3_data.coords, spectral_index_coords)

        self.assertEqual([mock_save_data.return_value], product)

    @run_periodically(timedelta(days=3))
    def test_spectral_fit_against_validation_data(self):
        test_cases = [
            ("hi45", "hi/fake_l2_maps/hi45-6months.cdf",
             "hi/validation/spectral_index/IMAP-Hi45_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/spectral_index/IMAP-Hi45_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi90", "hi/fake_l2_maps/hi90-6months.cdf",
             "hi/validation/spectral_index/IMAP-Hi90_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/spectral_index/IMAP-Hi90_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi45-zirnstein-mondel", "hi/fake_l2_maps/hi45-zirnstein-mondel-6months.cdf",
             "hi/validation/spectral_index/IMAP-Hi45_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/spectral_index/IMAP-Hi45_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi90-zirnstein-mondel", "hi/fake_l2_maps/hi90-zirnstein-mondel-6months.cdf",
             "hi/validation/spectral_index/IMAP-Hi90_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/spectral_index/IMAP-Hi90_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam_sig.csv"),
        ]

        for name, input_file_path, expected_gamma_path, expected_sigma_path in test_cases:
            with self.subTest(name):
                dependencies = HiL3SpectralIndexDependencies.from_file_paths(
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
                np.testing.assert_allclose(output_data.spectral_index_map_data.ena_spectral_index_stat_uncert[0, 0],
                                           expected_gamma_sigma, atol=1e-3)

    def test_spectral_fit_fields_other_than_fit_fields(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])
        lat = np.arange(-90, 90, 45)
        lon = np.arange(0, 360, 45)

        input_map = create_rectangular_intensity_map_data(energy=input_energies, energy_delta=input_deltas, lat=lat,
                                                          lon=lon)
        intensity_data = input_map.intensity_map_data
        intensity_data.obs_date[0, 0] = datetime(2025, 1, 1)
        intensity_data.obs_date[0, 1] = datetime(2025, 1, 1)
        intensity_data.obs_date[0, 2] = datetime(2027, 1, 1)

        intensity_data.exposure_factor[0, 0] = 1.0
        intensity_data.exposure_factor[0, 1] = 2.0
        intensity_data.exposure_factor[0, 2] = 3.0

        intensity_data.obs_date_range[0, 0] = 1
        intensity_data.obs_date_range[0, 1] = 1
        intensity_data.obs_date_range[0, 2] = 3

        processor = HiProcessor(sentinel.input_metadata, sentinel.dependencies)

        dependencies = HiL3SpectralIndexDependencies(input_map)

        output = processor.process_spectral_fit_index(dependencies)

        self.assertEqual(output.spectral_index_map_data.energy.shape[0], 1)
        self.assertEqual(output.spectral_index_map_data.energy[0], 10)
        np.testing.assert_allclose(output.spectral_index_map_data.energy_delta_minus, np.array([9]))
        np.testing.assert_allclose(output.spectral_index_map_data.energy_delta_plus, np.array([90]))
        self.assertEqual(output.spectral_index_map_data.energy_label.shape[0], 1)
        self.assertEqual("1.0 - 100.0", output.spectral_index_map_data.energy_label[0])

        expected_ena_shape = np.array([1, 1, len(lon), len(lat)])
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date,
                                      np.full(expected_ena_shape, datetime(2026, 1, 1)))
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date_range, np.full(expected_ena_shape, 2))
        np.testing.assert_array_equal(output.spectral_index_map_data.exposure_factor, np.full(expected_ena_shape, 6))

    def test_spectral_fit_other_fields_with_one_zero_exposure(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])
        lat = np.arange(-90, 90, 45)
        lon = np.arange(0, 360, 45)

        input_map = create_rectangular_intensity_map_data(energy=input_energies, energy_delta=input_deltas, lat=lat,
                                                          lon=lon)
        intensity_data = input_map.intensity_map_data
        intensity_data.obs_date[0, 0] = datetime(2025, 1, 1)
        intensity_data.obs_date[0, 1] = datetime(2025, 1, 1)
        intensity_data.obs_date[0, 2] = datetime(2027, 1, 1)
        intensity_data.obs_date.mask = np.ma.getmaskarray(intensity_data.obs_date)
        intensity_data.obs_date.mask[0, 1] = True

        intensity_data.exposure_factor[0, 0] = 1.0
        intensity_data.exposure_factor[0, 1] = 0.0
        intensity_data.exposure_factor[0, 2] = 1.0

        intensity_data.obs_date_range[0, 0] = 1
        intensity_data.obs_date_range[0, 1] = 1
        intensity_data.obs_date_range[0, 2] = 3
        intensity_data.obs_date_range.mask = np.ma.getmaskarray(intensity_data.obs_date_range)
        intensity_data.obs_date_range.mask[0, 1] = True

        processor = HiProcessor(sentinel.input_metadata, sentinel.dependencies)

        dependencies = HiL3SpectralIndexDependencies(input_map)

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

        input_map = create_rectangular_intensity_map_data(energy=input_energies, energy_delta=input_deltas, lat=lat,
                                                          lon=lon)
        intensity_data = input_map.intensity_map_data
        intensity_data.obs_date.mask = np.ma.getmaskarray(intensity_data.obs_date)
        intensity_data.obs_date.mask[:] = True

        intensity_data.exposure_factor[:] = 0.0

        intensity_data.obs_date_range.mask = np.ma.getmaskarray(intensity_data.obs_date_range)
        intensity_data.obs_date_range.mask[:] = True

        processor = HiProcessor(sentinel.input_metadata, sentinel.dependencies)

        dependencies = HiL3SpectralIndexDependencies(input_map)

        output = processor.process_spectral_fit_index(dependencies)

        expected_ena_shape = np.array([1, 1, len(lon), len(lat)])

        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date.mask, np.full(expected_ena_shape, True),
                                      strict=True)
        np.testing.assert_array_equal(output.spectral_index_map_data.obs_date_range.mask,
                                      np.full(expected_ena_shape, True), strict=True)

    @patch('imap_l3_processing.hi.hi_processor.MapProcessor.get_parent_file_names')
    @patch("imap_l3_processing.hi.hi_processor.HiLoL3SurvivalDependencies.fetch_dependencies")
    @patch("imap_l3_processing.hi.hi_processor.process_survival_probabilities")
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    def test_process_ram_or_antiram_spin(self, mock_save_data,
                                         mock_process_survival_prob,
                                         mock_fetch_survival_dependencies, mock_get_parent_file_names):

        mock_get_parent_file_names.return_value = ["somewhere"]

        cases = [
            ("ram", "h90-ena-h-sf-sp-ram-hae-4deg-6mo", SpiceFrame.ECLIPJ2000),
            ("anti", "h90-ena-h-sf-sp-anti-hae-4deg-6mo", SpiceFrame.IMAP_GCS),
            ("anti", "h90-ena-h-hf-sp-anti-hae-4deg-6mo", SpiceFrame.IMAP_HNU)
        ]

        for case, descriptor, spice_frame_name in cases:
            with self.subTest(case):
                mock_save_data.reset_mock()
                mock_fetch_survival_dependencies.reset_mock()
                mock_process_survival_prob.reset_mock()

                input_metadata = InputMetadata(instrument="hi",
                                               data_level="l3",
                                               start_date=datetime.now(),
                                               end_date=datetime.now() + timedelta(days=1),
                                               version="",
                                               descriptor=descriptor,
                                               )

                dependencies: HiLoL3SurvivalDependencies = mock_fetch_survival_dependencies.return_value
                dependencies.dependency_file_paths = [
                    Path("folder/map"), Path("folder/l1c")]

                mock_process_survival_prob.return_value = sentinel.survival_probabilities

                processor = HiProcessor(sentinel.input_dependencies, input_metadata)
                product = processor.process(spice_frame_name=spice_frame_name)

                mock_fetch_survival_dependencies.assert_called_once_with(sentinel.input_dependencies,
                                                                         Instrument.IMAP_HI)

                mock_process_survival_prob.assert_called_once_with(dependencies, spice_frame_name)

                mock_save_data.assert_called_once_with(RectangularIntensityDataProduct(
                    input_metadata=input_metadata,
                    parent_file_names=["l1c", "map", "somewhere"],
                    data=sentinel.survival_probabilities))
                self.assertEqual([mock_save_data.return_value], product)

    @patch('imap_l3_processing.hi.hi_processor.MapProcessor.get_parent_file_names')
    @patch("imap_l3_processing.hi.hi_processor.HiL3SingleSensorFullSpinDependencies.fetch_dependencies")
    @patch("imap_l3_processing.hi.hi_processor.process_survival_probabilities")
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    @patch('imap_l3_processing.hi.hi_processor.UnweightedCombination')
    def test_process_full_spin_single_sensor_map(self, mock_unweighted_combination_class, mock_save_data,
                                                 mock_process_survival_prob,
                                                 mock_fetch_full_spin_single_sensor_dependencies,
                                                 mock_get_parent_file_names):
        mock_unweighted_combination = Mock()
        mock_unweighted_combination_class.return_value = mock_unweighted_combination

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
        product = processor.process(spice_frame_name=sentinel.spice_frame)

        mock_fetch_full_spin_single_sensor_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_process_survival_prob.assert_has_calls([
            call(full_spin_dependencies.ram_dependencies, sentinel.spice_frame),
            call(full_spin_dependencies.antiram_dependencies, sentinel.spice_frame)
        ])

        mock_unweighted_combination.combine_rectangular_intensity_map_data.assert_called_once_with([
            sentinel.survival_corrected_ram, sentinel.survival_corrected_antiram])

        mock_save_data.assert_called_once_with(RectangularIntensityDataProduct(
            input_metadata=input_metadata,
            parent_file_names=["antiram_map", "l1c", "ram_map"],
            data=mock_unweighted_combination.combine_rectangular_intensity_map_data.return_value))
        self.assertEqual([mock_save_data.return_value], product)

    def test_raises_error_for_currently_unimplemented_maps(self):
        cases = [
            "hic-ena-h-sf-sp-ram-hae-6deg-6mo",
            "h45-ena-h-hf-nsp-full-hae-6deg-6mo",
            "h90-ena-h-hf-sp-full-hae-6deg-6mo",
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
                    product = processor.process()
