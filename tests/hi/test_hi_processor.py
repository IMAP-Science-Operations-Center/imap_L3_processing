import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, call

import numpy as np

from imap_processing.hi.hi_processor import HiProcessor
from imap_processing.hi.l3.hi_l3_dependencies import HiL3Dependencies
from imap_processing.hi.l3.models import HiL3Data
from imap_processing.models import InputMetadata


class TestHiProcessor(unittest.TestCase):
    @patch('imap_processing.hi.hi_processor.HiL3Dependencies.fetch_dependencies')
    @patch('imap_processing.hi.hi_processor.mpfit')
    def test_process(self, mock_mpfit, mock_fetch_dependencies):
        lat = np.array([0, 45])
        long = np.array([0, 45, 90])
        energy = np.array([10, 20, 30])
        epoch = np.array([datetime.now()])
        flux = np.array(
            [[
                [[1, 2, 3], [10, 20, 30]],
                [[100, 200, 300], [1000, 2000, 3000]],
                [[150, 300, 450], [600, 750, 900]]
            ]]
        )
        variance = np.array(
            [[
                [[1, 1, 1], [10, 10, 10]],
                [[100, 100, 100], [1000, 1000, 1000]],
                [[20, 20, 20], [40, 40, 40]]
            ]]
        )

        hi_l3_data = _create_h1_l3_data(lat=lat, lon=long, energy=energy, epoch=epoch)
        dependencies = HiL3Dependencies(hi_l3_data=hi_l3_data)
        upstream_dependencies = [Mock()]
        mock_fetch_dependencies.return_value = dependencies

        initial_parameters = (1, 1)

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        processor = HiProcessor(upstream_dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(upstream_dependencies)
        self.assertEqual(mock_mpfit.call_count, len(hi_l3_data.lon) * len(hi_l3_data.lat))
        mock_mpfit.has_calls(
            [
                call(HiProcessor._fit_function, initial_parameters,
                     {"xval": energy, "yval": flux[0][0][0], 'errval': variance[0][0][0]}),
                call(HiProcessor._fit_function, initial_parameters,
                     {"xval": energy, "yval": flux[0][1][1], 'errval': variance[0][1][1]}),
                call(HiProcessor._fit_function, initial_parameters,
                     {"xval": energy, "yval": flux[0][2][0], 'errval': variance[0][2][0]}),
                call(HiProcessor._fit_function, initial_parameters,
                     {"xval": energy, "yval": flux[0][0][1], 'errval': variance[0][0][1]}),
                call(HiProcessor._fit_function, initial_parameters,
                     {"xval": energy, "yval": flux[0][1][0], 'errval': variance[0][1][0]}),
                call(HiProcessor._fit_function, initial_parameters,
                     {"xval": energy, "yval": flux[0][2][1], 'errval': variance[0][2][1]}),

            ]
        )
        mock_fetch_dependencies.return_value = dependencies

    def test_power_law_function(self):
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )
        upstream_dependencies = [Mock()]
        processor = HiProcessor(upstream_dependencies, input_metadata)

        params = (2, -2)
        x = np.array([1, 2, 3])
        y = np.array([4, 10, 22])
        err = np.array([2, 2, 2])
        keywords = {'xval': x, 'yval': y, 'errval': err}

        expected_residual = np.array([1, 1, 2])
        status, actual_residuals = processor._fit_function(params, **keywords)

        np.testing.assert_array_equal(actual_residuals, expected_residual)
        self.assertEqual(status, 0)

    def test_finds_best_fit(self):
        np.random.seed(42)
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)

        lat = np.array([0])
        long = np.array([0])
        epoch = np.array([datetime.now()])
        energy = energies
        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        hi_l3_data = _create_h1_l3_data(lat=lat, lon=long, energy=energy, epoch=epoch, flux=flux, variance=variance)

        dependency = HiL3Dependencies(hi_l3_data=hi_l3_data)
        processor = HiProcessor(Mock(), Mock())

        result = processor._process_spectral_fit_index(dependency)
        np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, 1))


def _create_h1_l3_data(epoch=None, lon=None, lat=None, energy=None, flux=None, variance=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
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
        energy_deltas=np.full((len(energy), 2), 1),
        counts=np.full_like(flux, np.nan),
        counts_uncertainty=np.full_like(flux, np.nan),
        epoch_delta=np.full(2, timedelta(minutes=5)),
        exposure=np.full((len(epoch), len(lat), len(lon)), np.nan),
        sensitivity=np.full_like(flux, np.nan),
        variance=variance,
    )
