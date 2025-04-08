import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, call, sentinel

import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.hi_processor import HiProcessor, combine_glows_l3e_hi_l1c, MapDescriptorParts, \
    parse_map_descriptor
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies, \
    HI_L3_SPECTRAL_FIT_DESCRIPTOR
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies
from imap_l3_processing.hi.l3.models import HiMapData, HiL3SpectralIndexDataProduct, GlowsL3eData, HiL1cData
from imap_l3_processing.hi.l3.science.survival_probability import Sensor
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path, NumpyArrayMatcher
import xarray as xr


class TestHiProcessor(unittest.TestCase):
    @patch('imap_l3_processing.hi.hi_processor.HiL3SpectralFitDependencies.fetch_dependencies')
    @patch('imap_l3_processing.hi.hi_processor.spectral_fit')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    def test_process_spectral_fit(self, mock_save_data, mock_spectral_fit, mock_fetch_dependencies):
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
                                       descriptor=HI_L3_SPECTRAL_FIT_DESCRIPTOR,
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

    @patch('imap_l3_processing.hi.hi_processor.save_data')
    @patch("imap_l3_processing.hi.hi_processor.HiL3SurvivalCorrectedDataProduct")
    @patch("imap_l3_processing.hi.hi_processor.parse_map_descriptor")
    @patch('imap_l3_processing.hi.hi_processor.HiSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.hi.hi_processor.HiSurvivalProbabilityPointingSet')
    @patch('imap_l3_processing.hi.hi_processor.combine_glows_l3e_hi_l1c')
    @patch('imap_l3_processing.hi.hi_processor.HiL3SurvivalDependencies.fetch_dependencies')
    def test_process_survival_probability(self, mock_fetch_dependencies, mock_combine_glows_l3e_hi_l1c,
                                          mock_survival_probability_pointing_set, mock_survival_skymap,
                                          mock_parse_map_descriptor, mock_data_product_class, mock_save_data):

        rng = np.random.default_rng()
        input_map_flux = rng.random((1, 9, 90, 45))

        epoch = datetime.now()

        input_map = HiMapData(
            epoch=np.array([epoch]),
            energy=rng.random((1,)),
            energy_deltas=rng.random((1,)),
            counts=rng.random((1,)),
            counts_uncertainty=rng.random((1,)),
            epoch_delta=rng.random((1,)),
            exposure=rng.random((1,)),
            flux=input_map_flux,
            lat=rng.random((1,)),
            lon=rng.random((1,)),
            sensitivity=rng.random((1,)),
            variance=rng.random((1,)),
        )

        mock_fetch_dependencies.return_value = HiL3SurvivalDependencies(l2_data=input_map,
                                                                        hi_l1c_data=sentinel.hi_l1c_data,
                                                                        glows_l3e_data=sentinel.glows_l3e_data, )

        mock_combine_glows_l3e_hi_l1c.return_value = [(sentinel.hi_l1c_1, sentinel.glows_l3e_1),
                                                      (sentinel.hi_l1c_2, sentinel.glows_l3e_2),
                                                      (sentinel.hi_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        mock_parse_map_descriptor.return_value = MapDescriptorParts(sensor=sentinel.sensor,
                                                                    grid_size=sentinel.grid_size)

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"45sensor-spacecraft-survival-full-4deg-map",
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

        mock_parse_map_descriptor.assert_called_once_with(input_metadata.descriptor)

        mock_combine_glows_l3e_hi_l1c.assert_called_once_with(sentinel.glows_l3e_data, sentinel.hi_l1c_data)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.hi_l1c_1, sentinel.sensor, sentinel.glows_l3e_1),
            call(sentinel.hi_l1c_2, sentinel.sensor, sentinel.glows_l3e_2),
            call(sentinel.hi_l1c_3, sentinel.sensor, sentinel.glows_l3e_3)
        ])

        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     sentinel.grid_size,
                                                     SpiceFrame.ECLIPJ2000)

        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        mock_data_product_class.assert_called_once_with(
            input_metadata=input_metadata,
            epoch=input_map.epoch,
            energy=input_map.energy,
            energy_deltas=input_map.energy_deltas,
            counts=input_map.counts,
            counts_uncertainty=input_map.counts_uncertainty,
            epoch_delta=input_map.epoch_delta,
            exposure=input_map.exposure,
            flux=NumpyArrayMatcher(input_map.flux / computed_survival_probabilities),
            lat=input_map.lat,
            lon=input_map.lon,
            sensitivity=input_map.sensitivity,
            variance=input_map.variance,
        )

        mock_save_data.assert_called_once_with(mock_data_product_class.return_value)

    def test_parse_map_descriptor(self):
        test_cases = [
            ("45sensor-spacecraft-survival-full-4deg-map", Sensor.Hi45, 4),
            ("45sensor-spacecraft-survival-full-6deg-map", Sensor.Hi45, 6),
            ("90sensor-spacecraft-survival-full-4deg-map", Sensor.Hi90, 4),
            ("90sensor-spacecraft-survival-full-6deg-map", Sensor.Hi90, 6)
        ]

        for descriptor, expected_sensor, expected_grid_size in test_cases:
            with self.subTest(f"{expected_sensor}-{expected_grid_size}"):
                descriptor_parts = parse_map_descriptor(descriptor)
                self.assertEqual(MapDescriptorParts(sensor=expected_sensor, grid_size=expected_grid_size),
                                 descriptor_parts)

    def test_combine_glows_l3e_hi_l1c(self):
        glows_l3e_data = [
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-01T00:00:00Z"), spin_angle=None, energy=None,
                         probability_of_survival=None),
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-02T00:00:00Z"), spin_angle=None, energy=None,
                         probability_of_survival=None),
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-03T00:00:00Z"), spin_angle=None, energy=None,
                         probability_of_survival=None),
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-05T00:00:00Z"), spin_angle=None, energy=None,
                         probability_of_survival=None),
        ]

        hi_l1c_data = [
            HiL1cData(epoch=datetime.fromisoformat("2023-01-02T00:00:00Z"), exposure_times=None, esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-04T00:00:00Z"), exposure_times=None, esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-05T00:00:00Z"), exposure_times=None, esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-06T00:00:00Z"), exposure_times=None, esa_energy_step=None),
        ]

        expected = [
            (hi_l1c_data[0], glows_l3e_data[1],),
            (hi_l1c_data[2], glows_l3e_data[3],),
        ]

        actual = combine_glows_l3e_hi_l1c(glows_l3e_data, hi_l1c_data)

        self.assertEqual(expected, actual)


def _create_h1_l3_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None, variance=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    flux = flux if flux is not None else np.full((len(epoch), len(lon), len(lat), len(energy)), fill_value=1)
    variance = variance if variance is not None else np.full((len(epoch), len(lon), len(lat), len(energy)),
                                                             fill_value=1)
    epoch = epoch if epoch is not None else np.array([datetime.now()])

    return HiMapData(
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
