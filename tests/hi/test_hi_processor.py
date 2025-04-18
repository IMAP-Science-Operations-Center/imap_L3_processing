import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, call, sentinel

import numpy as np
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.hi_processor import HiProcessor, combine_glows_l3e_hi_l1c
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies
from imap_l3_processing.hi.l3.models import HiL3SpectralIndexDataProduct, GlowsL3eData, HiL1cData, \
    HiL3SurvivalCorrectedDataProduct, HiIntensityMapData
from imap_l3_processing.hi.l3.utils import MapDescriptorParts, MapQuantity
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path


class TestHiProcessor(unittest.TestCase):
    @patch("imap_l3_processing.hi.hi_processor.parse_map_descriptor")
    @patch('imap_l3_processing.hi.hi_processor.HiL3SpectralFitDependencies.fetch_dependencies')
    @patch('imap_l3_processing.hi.hi_processor.spectral_fit')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    def test_process_spectral_fit(self, mock_save_data, mock_spectral_fit,
                                  mock_fetch_dependencies, mock_parse_map_descriptor):
        lat = np.array([0, 45])
        long = np.array([0, 45, 90])
        energy = sentinel.energy
        epoch = np.array([datetime.now()])
        flux = sentinel.flux
        intensity_stat_unc = 5

        mock_parse_map_descriptor.return_value = MapDescriptorParts(
            sensor=sentinel.sensor,
            cg_correction=sentinel.cg_correction,
            survival_correction=sentinel.survival_correction,
            spin_phase=sentinel.spin_phase,
            duration=sentinel.duration,
            grid_size=sentinel.grid_size,
            quantity=MapQuantity.SpectralIndex
        )

        hi_l3_data = _create_h1_l3_data(lat=lat, lon=long, energy=energy, epoch=epoch, flux=flux,
                                        intensity_stat_unc=intensity_stat_unc,
                                        energy_delta=sentinel.energy_delta)
        dependencies = HiL3SpectralFitDependencies(hi_l3_data=hi_l3_data)
        upstream_dependencies = [Mock()]
        mock_fetch_dependencies.return_value = dependencies

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="h45-hf-sp-hae-4deg-6mo-spectral",
                                       )

        mock_spectral_fit.return_value = sentinel.gammas, sentinel.errors
        processor = HiProcessor(upstream_dependencies, input_metadata)
        processor.process()

        mock_parse_map_descriptor.assert_called_once_with(input_metadata.descriptor)

        mock_fetch_dependencies.assert_called_with(upstream_dependencies)
        mock_spectral_fit.assert_called_once_with(len(epoch), len(long), len(lat), hi_l3_data.ena_intensity,
                                                  np.square(hi_l3_data.ena_intensity_stat_unc),
                                                  hi_l3_data.energy)

        mock_save_data.assert_called_once()
        actual_hi_data_product: HiL3SpectralIndexDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(input_metadata.to_upstream_data_dependency(input_metadata.descriptor),
                         actual_hi_data_product.input_metadata)
        self.assertEqual(sentinel.gammas, actual_hi_data_product.ena_spectral_index)
        self.assertEqual(sentinel.errors, actual_hi_data_product.ena_spectral_index_stat_unc)
        self.assertEqual(sentinel.energy_delta, actual_hi_data_product.energy_delta_minus)
        self.assertEqual(sentinel.energy_delta, actual_hi_data_product.energy_delta_plus)
        np.testing.assert_array_equal(actual_hi_data_product.energy, hi_l3_data.energy)
        np.testing.assert_array_equal(actual_hi_data_product.latitude, hi_l3_data.latitude)
        np.testing.assert_array_equal(actual_hi_data_product.longitude, hi_l3_data.longitude)
        np.testing.assert_array_equal(actual_hi_data_product.epoch, hi_l3_data.epoch)
        np.testing.assert_array_equal(actual_hi_data_product.exposure_factor, hi_l3_data.exposure_factor)

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
                output_data = processor._process_spectral_fit_index(dependencies)

                np.testing.assert_allclose(output_data.ena_spectral_index[0],
                                           expected_gamma, atol=1e-3)
                np.testing.assert_allclose(output_data.ena_spectral_index_stat_unc[0],
                                           expected_gamma_sigma, atol=1e-3)

    @patch('imap_l3_processing.hi.hi_processor.upload')
    @patch('imap_l3_processing.hi.hi_processor.save_data')
    @patch("imap_l3_processing.hi.hi_processor.parse_map_descriptor")
    @patch('imap_l3_processing.hi.hi_processor.HiSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.hi.hi_processor.HiSurvivalProbabilityPointingSet')
    @patch('imap_l3_processing.hi.hi_processor.combine_glows_l3e_hi_l1c')
    @patch('imap_l3_processing.hi.hi_processor.HiL3SurvivalDependencies.fetch_dependencies')
    def test_process_survival_probability(self, mock_fetch_dependencies, mock_combine_glows_l3e_hi_l1c,
                                          mock_survival_probability_pointing_set, mock_survival_skymap,
                                          mock_parse_map_descriptor, mock_save_data, mock_upload):
        rng = np.random.default_rng()
        input_map_flux = rng.random((1, 9, 90, 45))
        epoch = datetime.now()

        input_map: HiIntensityMapData = _create_h1_l3_data(epoch=[epoch], flux=input_map_flux)

        input_map.energy = sentinel.hi_l2_energies

        mock_fetch_dependencies.return_value = HiL3SurvivalDependencies(l2_data=input_map,
                                                                        hi_l1c_data=sentinel.hi_l1c_data,
                                                                        glows_l3e_data=sentinel.glows_l3e_data, )

        mock_combine_glows_l3e_hi_l1c.return_value = [(sentinel.hi_l1c_1, sentinel.glows_l3e_1),
                                                      (sentinel.hi_l1c_2, sentinel.glows_l3e_2),
                                                      (sentinel.hi_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        mock_parse_map_descriptor.return_value = MapDescriptorParts(
            sensor=sentinel.sensor,
            cg_correction=sentinel.cg_correction,
            survival_correction=sentinel.survival_correction,
            spin_phase=sentinel.spin_phase,
            duration=sentinel.duration,
            grid_size=sentinel.grid_size,
            quantity=MapQuantity.Intensity
        )

        input_metadata = InputMetadata(instrument="hi",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"h90-sf-sp-hae-4deg-6mo",
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
            call(sentinel.hi_l1c_1, sentinel.sensor, sentinel.glows_l3e_1, sentinel.hi_l2_energies),
            call(sentinel.hi_l1c_2, sentinel.sensor, sentinel.glows_l3e_2, sentinel.hi_l2_energies),
            call(sentinel.hi_l1c_3, sentinel.sensor, sentinel.glows_l3e_3, sentinel.hi_l2_energies)
        ])

        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     sentinel.grid_size,
                                                     SpiceFrame.ECLIPJ2000)

        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        mock_save_data.assert_called_once()
        survival_data_product: HiL3SurvivalCorrectedDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(input_metadata.to_upstream_data_dependency(input_metadata.descriptor),
                         survival_data_product.input_metadata)

        np.testing.assert_array_equal(survival_data_product.ena_intensity,
                                      input_map.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data_product.ena_intensity_stat_unc,
                                      input_map.ena_intensity_stat_unc / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data_product.ena_intensity_sys_err,
                                      input_map.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data_product.epoch, input_map.epoch)
        np.testing.assert_array_equal(survival_data_product.epoch_delta, input_map.epoch_delta)
        np.testing.assert_array_equal(survival_data_product.energy, input_map.energy)
        np.testing.assert_array_equal(survival_data_product.energy_delta_plus, input_map.energy_delta_plus)
        np.testing.assert_array_equal(survival_data_product.energy_delta_minus, input_map.energy_delta_minus)
        np.testing.assert_array_equal(survival_data_product.energy_label, input_map.energy_label)
        np.testing.assert_array_equal(survival_data_product.latitude, input_map.latitude)
        np.testing.assert_array_equal(survival_data_product.latitude_delta, input_map.latitude_delta)
        np.testing.assert_array_equal(survival_data_product.latitude_label, input_map.latitude_label)
        np.testing.assert_array_equal(survival_data_product.longitude, input_map.longitude)
        np.testing.assert_array_equal(survival_data_product.longitude_delta, input_map.longitude_delta)
        np.testing.assert_array_equal(survival_data_product.longitude_label, input_map.longitude_label)
        np.testing.assert_array_equal(survival_data_product.exposure_factor, input_map.exposure_factor)
        np.testing.assert_array_equal(survival_data_product.obs_date, input_map.obs_date)
        np.testing.assert_array_equal(survival_data_product.obs_date_range, input_map.obs_date_range)
        np.testing.assert_array_equal(survival_data_product.solid_angle, input_map.solid_angle)

        mock_upload.assert_called_once_with(mock_save_data.return_value)

    def test_combine_glows_l3e_hi_l1c(self):
        glows_l3e_data = [
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-01T00:00:00Z"), spin_angle=None,
                         energy=None, probability_of_survival=None),
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-02T00:00:00Z"), spin_angle=None,
                         energy=None, probability_of_survival=None),
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-03T00:00:00Z"), spin_angle=None,
                         energy=None, probability_of_survival=None),
            GlowsL3eData(epoch=datetime.fromisoformat("2023-01-05T00:00:00Z"), spin_angle=None,
                         energy=None, probability_of_survival=None),
        ]

        hi_l1c_data = [
            HiL1cData(epoch=datetime.fromisoformat("2023-01-02T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-04T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-05T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-06T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
        ]

        expected = [
            (hi_l1c_data[0], glows_l3e_data[1],),
            (hi_l1c_data[2], glows_l3e_data[3],),
        ]

        actual = combine_glows_l3e_hi_l1c(glows_l3e_data, hi_l1c_data)

        self.assertEqual(expected, actual)


def _create_h1_l3_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None,
                       intensity_stat_unc=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    flux = flux if flux is not None else np.full((len(epoch), len(lon), len(lat), len(energy)), fill_value=1)
    intensity_stat_unc = intensity_stat_unc if intensity_stat_unc is not None else np.full(
        (len(epoch), len(lon), len(lat), len(energy)),
        fill_value=1)
    epoch = epoch if epoch is not None else np.array([datetime.now()])

    if isinstance(flux, np.ndarray):
        more_real_flux = flux
    else:
        more_real_flux = np.full((len(epoch), len(lon), len(lat), 9), fill_value=1)

    return HiIntensityMapData(
        epoch=epoch,
        epoch_delta=np.array([0]),
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
        obs_date=np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1)),
        obs_date_range=np.full_like(more_real_flux, 0),
        solid_angle=np.full_like(more_real_flux, 0),
        ena_intensity=flux,
        ena_intensity_stat_unc=intensity_stat_unc,
        ena_intensity_sys_err=np.full_like(flux, 0),
    )
