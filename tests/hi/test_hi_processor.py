import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, call, sentinel

import numpy as np

from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_dependencies import HiL3Dependencies
from imap_l3_processing.hi.l3.models import HiL3Data, HiL3SpectralIndexDataProduct
from imap_l3_processing.models import InputMetadata


class TestHiProcessor(unittest.TestCase):
    @patch('imap_l3_processing.hi.hi_processor.HiL3Dependencies.fetch_dependencies')
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
        dependencies = HiL3Dependencies(hi_l3_data=hi_l3_data)
        upstream_dependencies = [Mock()]
        mock_fetch_dependencies.return_value = dependencies

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        mock_spectral_fit.return_value = sentinel.gammas
        processor = HiProcessor(upstream_dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(upstream_dependencies)
        mock_spectral_fit.assert_called_once_with(len(epoch), len(long), len(lat), hi_l3_data.flux, hi_l3_data.variance,
                                                  hi_l3_data.energy)

        mock_save_data.assert_called_once()
        actual_hi_data_product: HiL3SpectralIndexDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(sentinel.gammas, actual_hi_data_product.spectral_fit_index)
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
