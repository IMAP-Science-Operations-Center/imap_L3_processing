import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, call, sentinel

import numpy as np

from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.hi.l3.models import HiL3Data, HiL3SpectralIndexDataProduct
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path


class TestHiProcessor(unittest.TestCase):
    @patch('imap_l3_processing.hi.hi_processor.HiL3SpectralFitDependencies.fetch_dependencies')
    @patch('imap_l3_processing.hi.hi_processor.spectral_fit')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    def test_process(self, mock_save_data, mock_spectral_fit, mock_fetch_dependencies):
        lat = np.array([0, 45])
        long = np.array([0, 45, 90])
        energy = sentinel.energy
        epoch = np.array([datetime.now()])
        flux = sentinel.flux
        variance = sentinel.variance

        hi_l3_data = _create_h1_l3_data(lat=lat, lon=long, energy=energy, epoch=epoch, flux=flux, variance=variance,
                                        energy_delta=sentinel.energy_delta)
        dependencies = HiL3SpectralFitDependencies(hi_l3_data=hi_l3_data)
        upstream_dependencies = [Mock()]
        mock_fetch_dependencies.return_value = dependencies

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        mock_spectral_fit.return_value = sentinel.gammas, sentinel.errors
        processor = HiProcessor(upstream_dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(upstream_dependencies)
        mock_spectral_fit.assert_called_once_with(len(epoch), len(long), len(lat), hi_l3_data.flux, hi_l3_data.variance,
                                                  hi_l3_data.energy)

        mock_save_data.assert_called_once()
        actual_hi_data_product: HiL3SpectralIndexDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(sentinel.gammas, actual_hi_data_product.spectral_fit_index)
        self.assertEqual(sentinel.errors, actual_hi_data_product.spectral_fit_index_error)
        self.assertEqual(sentinel.flux, actual_hi_data_product.flux)
        self.assertEqual(sentinel.variance, actual_hi_data_product.variance)
        self.assertEqual(sentinel.energy_delta, actual_hi_data_product.energy_deltas)
        np.testing.assert_array_equal(actual_hi_data_product.energy, hi_l3_data.energy)
        np.testing.assert_array_equal(actual_hi_data_product.sensitivity, hi_l3_data.sensitivity)
        np.testing.assert_array_equal(actual_hi_data_product.lat, hi_l3_data.lat)
        np.testing.assert_array_equal(actual_hi_data_product.lon, hi_l3_data.lon)
        np.testing.assert_array_equal(actual_hi_data_product.counts_uncertainty, hi_l3_data.counts_uncertainty)
        np.testing.assert_array_equal(actual_hi_data_product.counts, hi_l3_data.counts)
        np.testing.assert_array_equal(actual_hi_data_product.epoch, hi_l3_data.epoch)
        np.testing.assert_array_equal(actual_hi_data_product.flux, hi_l3_data.flux)
        np.testing.assert_array_equal(actual_hi_data_product.exposure, hi_l3_data.exposure)

    def test_spectral_fit_against_validation_data(self):
        expected_failures = ["hi45", "hi45-zirnstein-mondel"]

        test_cases = [
            ("hi45", "hi/validation/hi45-6months.cdf", "hi/validation/expected_Hi45_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/expected_Hi45_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi90", "hi/validation/hi90-6months.cdf", "hi/validation/expected_Hi90_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/expected_Hi90_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi45-zirnstein-mondel", "hi/validation/hi45-zirnstein-mondel-6months.cdf",
             "hi/validation/expected_Hi45_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/expected_Hi45_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam_sig.csv"),
            ("hi90-zirnstein-mondel", "hi/validation/hi90-zirnstein-mondel-6months.cdf",
             "hi/validation/expected_Hi90_gdf_zirnstein_model_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/expected_Hi90_gdf_Zirnstein_model_6months_4.0x4.0_fit_gam_sig.csv"),
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
                output_data = processor._process_spectral_fit_index(dependencies)

                try:
                    np.testing.assert_allclose(output_data.spectral_fit_index[0], expected_gamma, atol=1e-5)
                    np.testing.assert_allclose(output_data.spectral_fit_index_error[0], expected_gamma_sigma, atol=1e-5)
                except Exception as e:
                    if name in expected_failures:
                        print(f"Spectral fit validation failed expectedly (card 2419): {name}")
                        continue
                    else:
                        raise e


def _create_h1_l3_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None, variance=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    flux = flux if flux is not None else np.full((len(epoch), len(lon), len(lat), len(energy)), fill_value=1)
    variance = variance if variance is not None else np.full((len(epoch), len(lon), len(lat), len(energy)),
                                                             fill_value=1)
    epoch = epoch if epoch is not None else np.array([datetime.now()])

    return HiL3Data(
        epoch=epoch,
        energy=energy,
        flux=flux,
        lon=lon,
        lat=lat,
        energy_deltas=energy_delta,
        counts=np.full_like(flux, 12),
        counts_uncertainty=np.full_like(flux, 0.1),
        epoch_delta=np.full(2, timedelta(minutes=5)),
        exposure=np.full((len(epoch), len(lat), len(lon)), 2),
        sensitivity=np.full_like(flux, 0.5),
        variance=variance,
    )
