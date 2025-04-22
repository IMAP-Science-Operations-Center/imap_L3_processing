import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, sentinel, call

import numpy as np
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.l3.models import UltraL3SurvivalCorrectedDataProduct, UltraL2Map
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies
from imap_l3_processing.ultra.l3.ultra_processor import UltraProcessor, UltraMapDescriptorParts


class TestHiProcessor(unittest.TestCase):

    @patch('imap_l3_processing.ultra.l3.ultra_processor.upload')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.save_data')
    @patch("imap_l3_processing.ultra.l3.ultra_processor.parse_map_descriptor")
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbabilitySkyMap')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraSurvivalProbability')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.combine_glows_l3e_with_l1c_pointing')
    @patch('imap_l3_processing.ultra.l3.ultra_processor.UltraL3Dependencies.fetch_dependencies')
    def test_process_survival_probability(self, mock_fetch_dependencies, mock_combine_glows_l3e_with_l1c_pointing,
                                          mock_survival_probability_pointing_set, mock_survival_skymap,
                                          mock_parse_map_descriptor, mock_save_data, mock_upload):
        rng = np.random.default_rng()
        healpix_indices = np.arange(12)
        input_map_flux = rng.random((1, 9, 12))
        epoch = datetime.now()

        input_l2_map = _create_ultra_l2_data(epoch=[epoch], flux=input_map_flux, healpix_indices=healpix_indices)

        input_l2_map.energy = sentinel.ultra_l2_energies

        mock_fetch_dependencies.return_value = UltraL3Dependencies(
            ultra_l2_map=input_l2_map,
            ultra_l1c_pset=sentinel.ultra_l1c_pset,
            glows_l3e_sp=sentinel.glows_l3e_sp)

        mock_combine_glows_l3e_with_l1c_pointing.return_value = [(sentinel.ultra_l1c_1, sentinel.glows_l3e_1),
                                                                 (sentinel.ultra_l1c_2, sentinel.glows_l3e_2),
                                                                 (sentinel.ultra_l1c_3, sentinel.glows_l3e_3)]

        mock_survival_probability_pointing_set.side_effect = [sentinel.pset_1, sentinel.pset_2, sentinel.pset_3]

        mock_parse_map_descriptor.return_value = UltraMapDescriptorParts(grid_size=sentinel.grid_size)

        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor=f"45sensor-spacecraft-survival-full-4deg-map",
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

        mock_parse_map_descriptor.assert_called_once_with(input_metadata.descriptor)

        mock_combine_glows_l3e_with_l1c_pointing.assert_called_once_with(sentinel.glows_l3e_sp, sentinel.ultra_l1c_pset)

        mock_survival_probability_pointing_set.assert_has_calls([
            call(sentinel.ultra_l1c_1, sentinel.glows_l3e_1, sentinel.ultra_l2_energies),
            call(sentinel.ultra_l1c_2, sentinel.glows_l3e_2, sentinel.ultra_l2_energies),
            call(sentinel.ultra_l1c_3, sentinel.glows_l3e_3, sentinel.ultra_l2_energies)
        ])

        mock_survival_skymap.assert_called_once_with([sentinel.pset_1, sentinel.pset_2, sentinel.pset_3],
                                                     SpiceFrame.ECLIPJ2000)

        mock_survival_skymap.return_value.to_dataset.assert_called_once_with()

        mock_save_data.assert_called_once()
        survival_data_product: UltraL3SurvivalCorrectedDataProduct = mock_save_data.call_args_list[0].args[0]

        self.assertEqual(input_metadata.to_upstream_data_dependency(input_metadata.descriptor),
                         survival_data_product.input_metadata)

        np.testing.assert_array_equal(survival_data_product.ena_intensity,
                                      input_l2_map.ena_intensity / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data_product.ena_intensity_stat_unc,
                                      input_l2_map.ena_intensity_stat_unc / computed_survival_probabilities)
        np.testing.assert_array_equal(survival_data_product.ena_intensity_sys_err,
                                      input_l2_map.ena_intensity_sys_err / computed_survival_probabilities)

        np.testing.assert_array_equal(survival_data_product.epoch, input_l2_map.epoch)
        np.testing.assert_array_equal(survival_data_product.epoch_delta, input_l2_map.epoch_delta)
        np.testing.assert_array_equal(survival_data_product.energy, input_l2_map.energy)
        np.testing.assert_array_equal(survival_data_product.energy_delta_plus, input_l2_map.energy_delta_plus)
        np.testing.assert_array_equal(survival_data_product.energy_delta_minus, input_l2_map.energy_delta_minus)
        np.testing.assert_array_equal(survival_data_product.energy_label, input_l2_map.energy_label)
        np.testing.assert_array_equal(survival_data_product.latitude, input_l2_map.latitude)
        np.testing.assert_array_equal(survival_data_product.longitude, input_l2_map.longitude)
        np.testing.assert_array_equal(survival_data_product.exposure_factor, input_l2_map.exposure_factor)
        np.testing.assert_array_equal(survival_data_product.obs_date, input_l2_map.obs_date)
        np.testing.assert_array_equal(survival_data_product.obs_date_range, input_l2_map.obs_date_range)
        np.testing.assert_array_equal(survival_data_product.solid_angle, input_l2_map.solid_angle)

        mock_upload.assert_called_once_with(mock_save_data.return_value)


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

    return UltraL2Map(
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
        pixel_index=healpix_indices,
        pixel_index_label=np.full(healpix_indices.shape, "healpix index label")
    )
