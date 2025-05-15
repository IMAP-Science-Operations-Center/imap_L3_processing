import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
import xarray as xr
from imap_data_access.processing_input import AncillaryInput, ScienceInput, ProcessingInputCollection
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.map_models import HealPixIntensityMapData, IntensityMapData, HealPixCoords, \
    HealPixIntensityDataProduct, HealPixSpectralIndexDataProduct, SpectralIndexMapData, HealPixSpectralIndexMapData
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies, UltraL3SpectralIndexDependencies
from imap_l3_processing.ultra.l3.ultra_processor import UltraProcessor
from tests.test_helpers import get_test_data_path


class TestUltraProcessor(unittest.TestCase):

    @patch('imap_l3_processing.ultra.l3.ultra_processor.upload')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbability')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3Dependencies.fetch_dependencies')
    def test_process_survival_probability(self, mock_fetch_dependencies, mock_combine_glows_l3e_with_l1c_pointing,
                                          mock_survival_probability_pointing_set, mock_survival_skymap,
                                          mock_save_data, mock_upload):
        rng = np.random.default_rng()
        healpix_indices = np.arange(12)
        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()

        input_l2_map = _create_ultra_l2_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)

        input_l2_map.intensity_map_data.energy = sentinel.ultra_l2_energies

        mock_fetch_dependencies.return_value = UltraL3Dependencies(
            ultra_l2_map=input_l2_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp)

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.ultra_l1c_1, sentinel.glows_l3e_1),
                                                                 (sentinel.ultra_l1c_2, sentinel.glows_l3e_2),
                                                                 (sentinel.ultra_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"u90-ena-h-sf-sp-full-hae-nside8-6mo"
                                       )

        computed_survival_probabilities = rng.random((1, 9, healpix_indices.shape[0]))

        mock_survival_skymap.return_value.to_dataset.return_value = xr.Dataset({
            "exposure_weighted_survival_probabilities": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                computed_survival_probabilities
            )
        },
            coords={
                CoordNames.TIME.value: [epoch],
                CoordNames.ENERGY.value: rng.random((9,)),
                CoordNames.HEALPIX_INDEX.value: healpix_indices,
            })

        processor = UltraProcessor(sentinel.dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(sentinel.dependencies)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_sp, sentinel.ultra_l1c_pset)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.ultra_l1c_1, sentinel.glows_l3e_1),
            call(sentinel.ultra_l1c_2, sentinel.glows_l3e_2),
            call(sentinel.ultra_l1c_3, sentinel.glows_l3e_3)
        ])
        intensity_data = input_l2_map.intensity_map_data
        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     SpiceFrame.ECLIPJ2000, input_l2_map.coords.nside)

        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        mock_save_data.assert_called_once()
        survival_data_product: HealPixIntensityDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertIsInstance(survival_data_product, HealPixIntensityDataProduct)
        self.assertEqual(input_metadata.to_upstream_data_dependency(input_metadata.descriptor),
                         survival_data_product.input_metadata)

        intensity_map_data = survival_data_product.data.intensity_map_data
        np.testing.assert_array_equal(intensity_map_data.ena_intensity,
                                      intensity_data.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(intensity_map_data.ena_intensity_stat_unc,
                                      intensity_data.ena_intensity_stat_unc / computed_survival_probabilities)
        np.testing.assert_array_equal(intensity_map_data.ena_intensity_sys_err,
                                      intensity_data.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(intensity_map_data.epoch, intensity_data.epoch)
        np.testing.assert_array_equal(intensity_map_data.epoch_delta, intensity_data.epoch_delta)
        np.testing.assert_array_equal(intensity_map_data.energy, intensity_data.energy)
        np.testing.assert_array_equal(intensity_map_data.energy_delta_plus, intensity_data.energy_delta_plus)
        np.testing.assert_array_equal(intensity_map_data.energy_delta_minus, intensity_data.energy_delta_minus)
        np.testing.assert_array_equal(intensity_map_data.energy_label, intensity_data.energy_label)
        np.testing.assert_array_equal(intensity_map_data.latitude, intensity_data.latitude)
        np.testing.assert_array_equal(intensity_map_data.longitude, intensity_data.longitude)
        np.testing.assert_array_equal(intensity_map_data.exposure_factor, intensity_data.exposure_factor)
        np.testing.assert_array_equal(intensity_map_data.obs_date, intensity_data.obs_date)
        np.testing.assert_array_equal(intensity_map_data.obs_date_range, intensity_data.obs_date_range)
        np.testing.assert_array_equal(intensity_map_data.solid_angle, intensity_data.solid_angle)

        mock_upload.assert_called_once_with(mock_save_data.return_value)

    @patch('imap_l3_processing.processor.spiceypy')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.upload')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.calculate_spectral_index_for_multiple_ranges')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def test_process_spectral_index(self, mock_fetch_dependencies, mock_calculate_spectral_index, mock_save_data,
                                    mock_upload, mock_spiceypy):
        mock_spiceypy.ktotal.return_value = 0

        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        energy_range_file_name = 'imap_ultra_energy-range-descriptor_20250601_v000.dat'
        input_deps = ProcessingInputCollection(ScienceInput(map_file_name), AncillaryInput(energy_range_file_name))

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-nside8-6mo")
        input_map_data = HealPixIntensityMapData(Mock(), Mock())
        dependencies = UltraL3SpectralIndexDependencies(input_map_data, sentinel.energy_ranges)
        mock_fetch_dependencies.return_value = dependencies

        mock_spectral_index_map_data = Mock(spec=SpectralIndexMapData)
        mock_calculate_spectral_index.return_value = mock_spectral_index_map_data

        expected_healpix_spectral_index_map_data = HealPixSpectralIndexMapData(
            spectral_index_map_data=mock_spectral_index_map_data,
            coords=input_map_data.coords)

        expected_parent_file_names = [map_file_name, energy_range_file_name]

        expected_spectral_index_data_product = HealPixSpectralIndexDataProduct(
            data=expected_healpix_spectral_index_map_data, input_metadata=input_metadata,
            parent_file_names=expected_parent_file_names)

        processor = UltraProcessor(input_deps, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(input_deps)
        mock_calculate_spectral_index.assert_called_once_with(dependencies.map_data.intensity_map_data,
                                                              sentinel.energy_ranges)
        mock_save_data.assert_called_once_with(expected_spectral_index_data_product)

        mock_upload.assert_called_once_with(mock_save_data.return_value)

    @patch('imap_l3_processing.ultra.l3.ultra_processor.upload')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3SpectralIndexDependencies.fetch_dependencies')
    def test_process_spectral_index_validating_output_values(self, mock_fetch_dependencies, mock_save_data,
                                                             _):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="v000",
                                       descriptor=f"u90-spx-h-sf-sp-full-hae-nside8-6mo")
        input_map_path = get_test_data_path('ultra/fake_ultra_map_data_with_breakpoint_at_15keV.cdf')
        fit_energy_ranges_path = get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250507_v000.txt')
        dependencies = UltraL3SpectralIndexDependencies.from_file_paths(input_map_path, fit_energy_ranges_path)
        mock_fetch_dependencies.return_value = dependencies

        expected_ena_spectral_index = np.array([2] * 48 + [3.5] * 48).reshape(1, 2, 48)

        processing_input_collection = ProcessingInputCollection()
        processor = UltraProcessor(processing_input_collection, input_metadata)
        processor.process()

        actual_data_product: HealPixSpectralIndexDataProduct = mock_save_data.call_args[0][0]

        np.testing.assert_array_almost_equal(actual_data_product.data.spectral_index_map_data.ena_spectral_index,
                                             expected_ena_spectral_index)


def _create_ultra_l2_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None,
                          intensity_stat_unc=None, healpix_indices=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    healpix_indices = healpix_indices if healpix_indices is not None else np.arange(12)
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(healpix_indices)), fill_value=1)
    intensity_stat_unc = intensity_stat_unc if intensity_stat_unc is not None else np.full(
        flux.shape,
        fill_value=1)
    epoch = epoch if epoch is not None else np.array([datetime.now()])

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
            ena_intensity_stat_unc=intensity_stat_unc,
            ena_intensity_sys_err=np.full_like(flux, 0),
        ),
        HealPixCoords(
            pixel_index=healpix_indices,
            pixel_index_label=np.full(healpix_indices.shape, "healpix index label")
        )
    )
