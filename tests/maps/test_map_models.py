import dataclasses
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import sentinel, Mock, patch

import numpy as np
from imap_processing.ena_maps.utils.spatial_utils import build_solid_angle_map
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable_and_mask_fill_values
from imap_l3_processing.constants import ONE_SECOND_IN_NANOSECONDS, SECONDS_PER_DAY, FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.maps import map_models
from imap_l3_processing.maps.map_models import RectangularCoords, SpectralIndexMapData, RectangularSpectralIndexMapData, \
    RectangularSpectralIndexDataProduct, RectangularIntensityMapData, IntensityMapData, RectangularIntensityDataProduct, \
    combine_rectangular_intensity_map_data, combine_intensity_map_data, HealPixIntensityMapData, \
    HealPixSpectralIndexMapData, HealPixCoords, HealPixSpectralIndexDataProduct, HealPixIntensityDataProduct
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import get_test_data_folder


class TestMapModels(unittest.TestCase):

    def test_rectangular_spectral_index_to_data_product_variables(self):
        input_metadata = Mock()

        spectral_index_data_product = RectangularSpectralIndexDataProduct(
            input_metadata=input_metadata,
            data=RectangularSpectralIndexMapData(
                spectral_index_map_data=SpectralIndexMapData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    latitude=sentinel.latitude,
                    longitude=sentinel.longitude,
                    exposure_factor=sentinel.exposure_factor,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                    solid_angle=sentinel.solid_angle,
                    ena_spectral_index=sentinel.ena_spectral_index,
                    ena_spectral_index_stat_unc=sentinel.ena_spectral_index_stat_unc
                ),
                coords=RectangularCoords(
                    latitude_delta=sentinel.latitude_delta,
                    latitude_label=sentinel.latitude_label,
                    longitude_delta=sentinel.longitude_delta,
                    longitude_label=sentinel.longitude_label,
                )
            )
        )

        actual_variables = spectral_index_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(map_models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(map_models.ENA_SPECTRAL_INDEX_VAR_NAME, sentinel.ena_spectral_index),
            DataProductVariable(map_models.ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, sentinel.ena_spectral_index_stat_unc),
            DataProductVariable(map_models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta),
            DataProductVariable(map_models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label),
            DataProductVariable(map_models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta),
            DataProductVariable(map_models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_rectangular_intensity_to_data_product_variables(self):
        input_metadata = sentinel.input_metadata

        data_product = RectangularIntensityDataProduct(
            input_metadata=input_metadata,
            data=RectangularIntensityMapData(
                intensity_map_data=IntensityMapData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    latitude=sentinel.latitude,
                    longitude=sentinel.longitude,
                    exposure_factor=sentinel.exposure_factor,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                    solid_angle=sentinel.solid_angle,
                    ena_intensity=sentinel.ena_intensity,
                    ena_intensity_stat_unc=sentinel.ena_intensity_stat_unc,
                    ena_intensity_sys_err=sentinel.ena_intensity_sys_err,
                ),
                coords=RectangularCoords(
                    longitude_delta=sentinel.longitude_delta,
                    longitude_label=sentinel.longitude_label,
                    latitude_delta=sentinel.latitude_delta,
                    latitude_label=sentinel.latitude_label,

                )
            ),
        )
        actual_variables = data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(map_models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(map_models.ENA_INTENSITY_VAR_NAME, sentinel.ena_intensity),
            DataProductVariable(map_models.ENA_INTENSITY_STAT_UNC_VAR_NAME, sentinel.ena_intensity_stat_unc),
            DataProductVariable(map_models.ENA_INTENSITY_SYS_ERR_VAR_NAME, sentinel.ena_intensity_sys_err),
            DataProductVariable(map_models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta),
            DataProductVariable(map_models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label),
            DataProductVariable(map_models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta),
            DataProductVariable(map_models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_healpix_spectral_index_to_data_product_variables(self):
        input_metadata = Mock()

        spectral_index_data_product = HealPixSpectralIndexDataProduct(
            input_metadata=input_metadata,
            data=HealPixSpectralIndexMapData(
                spectral_index_map_data=SpectralIndexMapData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    latitude=sentinel.latitude,
                    longitude=sentinel.longitude,
                    exposure_factor=sentinel.exposure_factor,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                    solid_angle=sentinel.solid_angle,
                    ena_spectral_index=sentinel.ena_spectral_index,
                    ena_spectral_index_stat_unc=sentinel.ena_spectral_index_stat_unc
                ),
                coords=HealPixCoords(
                    pixel_index=sentinel.pixel_index,
                    pixel_index_label=sentinel.pixel_index_label,
                )
            )
        )

        actual_variables = spectral_index_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(map_models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(map_models.ENA_SPECTRAL_INDEX_VAR_NAME, sentinel.ena_spectral_index),
            DataProductVariable(map_models.ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, sentinel.ena_spectral_index_stat_unc),
            DataProductVariable(map_models.PIXEL_INDEX_VAR_NAME, sentinel.pixel_index),
            DataProductVariable(map_models.PIXEL_INDEX_LABEL_VAR_NAME, sentinel.pixel_index_label),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_healpix_intensity_to_data_product_variables(self):
        input_metadata = Mock()

        data_product = HealPixIntensityDataProduct(
            input_metadata=input_metadata,
            data=HealPixIntensityMapData(
                intensity_map_data=IntensityMapData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    latitude=sentinel.latitude,
                    longitude=sentinel.longitude,
                    exposure_factor=sentinel.exposure_factor,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                    solid_angle=sentinel.solid_angle,
                    ena_intensity=sentinel.ena_intensity,
                    ena_intensity_stat_unc=sentinel.ena_intensity_stat_unc,
                    ena_intensity_sys_err=sentinel.ena_intensity_sys_err,
                ),
                coords=HealPixCoords(
                    pixel_index=sentinel.pixel_index,
                    pixel_index_label=sentinel.pixel_index_label,
                )
            )
        )

        actual_variables = data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(map_models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(map_models.ENA_INTENSITY_VAR_NAME, sentinel.ena_intensity),
            DataProductVariable(map_models.ENA_INTENSITY_STAT_UNC_VAR_NAME, sentinel.ena_intensity_stat_unc),
            DataProductVariable(map_models.ENA_INTENSITY_SYS_ERR_VAR_NAME, sentinel.ena_intensity_sys_err),
            DataProductVariable(map_models.PIXEL_INDEX_VAR_NAME, sentinel.pixel_index),
            DataProductVariable(map_models.PIXEL_INDEX_LABEL_VAR_NAME, sentinel.pixel_index_label),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def construct_intensity_data_with_all_zero_fields(self) -> IntensityMapData:

        return IntensityMapData(
            epoch=np.array([0]),
            epoch_delta=np.array([0]),
            energy=np.array([0]),
            energy_delta_plus=np.array([0]),
            energy_delta_minus=np.array([0]),
            energy_label=np.array([0]),
            latitude=np.array([0]),
            longitude=np.array([0]),
            exposure_factor=np.array([1]),
            obs_date=np.array([datetime(2025, 5, 6)]),
            obs_date_range=np.array([0]),
            solid_angle=np.array([0]),
            ena_intensity=np.array([0]),
            ena_intensity_stat_unc=np.array([0]),
            ena_intensity_sys_err=np.array([0]),
        )

    def test_combine_maps_does_nothing_when_passed_a_single_map(self):
        map_1 = self.construct_intensity_data_with_all_zero_fields()

        combine_one = combine_intensity_map_data([map_1])
        np.testing.assert_equal(dataclasses.asdict(combine_one), dataclasses.asdict(map_1))

    def test_combine_maps_throws_exception_when_fields_vary_that_should_not(self):
        map_1 = self.construct_intensity_data_with_all_zero_fields()

        fields_which_may_differ = {"ena_intensity", "ena_intensity_stat_unc", "ena_intensity_sys_err",
                                   "exposure_factor", "obs_date", "obs_date_range"}

        alternate_values_by_type = {datetime: datetime(2025, 5, 6), str: "label"}
        generic_value = 10
        for field in dataclasses.fields(map_1):
            replacement_value = alternate_values_by_type.get(type(getattr(map_1, field.name)[0]), generic_value)
            map_with_difference = dataclasses.replace(map_1, **{field.name: np.array([replacement_value])})
            if field.name not in fields_which_may_differ:
                with self.assertRaises(AssertionError, msg=field.name):
                    combine_intensity_map_data([map_1, map_with_difference])
            else:
                try:
                    combine_intensity_map_data([map_1, map_with_difference])
                except:
                    self.fail(f"Differences in other fields should be alright: {field.name}")

    def test_combine_maps_does_a_time_weighted_average_of_intensity(self):
        map_1 = self.construct_intensity_data_with_all_zero_fields()
        map_1.ena_intensity = np.array([1, np.nan, 3, 4, np.nan])
        map_1.exposure_factor = np.array([1, 0, 5, 6, 0])
        map_1.ena_intensity_sys_err = np.array([1, np.nan, 10, 100, np.nan])
        map_1.ena_intensity_stat_unc = np.array([10, np.nan, 10, 10, np.nan])
        DATETIME_FILL = datetime(9999, 12, 31, 23, 59, 59, 999999)
        map_1.obs_date = np.ma.masked_equal(
            [datetime(2025, 5, 5),
             DATETIME_FILL,
             datetime(2025, 5, 7),
             datetime(2025, 5, 8),
             DATETIME_FILL],
            DATETIME_FILL)

        map_2 = self.construct_intensity_data_with_all_zero_fields()
        map_2.ena_intensity = np.array([5, 6, 7, 8, np.nan])
        map_2.exposure_factor = np.array([3, 1, 5, 2, 0])
        map_2.ena_intensity_sys_err = np.array([9, 4, 2, 0, np.nan])
        map_2.ena_intensity_stat_unc = np.array([1, 2, 3, 4, np.nan])
        map_2.obs_date = np.ma.masked_equal(
            [datetime(2025, 5, 9),
             datetime(2025, 5, 10),
             datetime(2025, 5, 11),
             datetime(2025, 5, 12),
             DATETIME_FILL],
            DATETIME_FILL)

        expected_combined_exposure = [4, 1, 10, 8, 0]
        expected_combined_intensity = [4, 6, 5, 5, np.nan]
        expected_sys_err = [7, 4, 6, 75, np.nan]
        expected_stat_unc = [np.sqrt((1 * 100 + 9 * 1) / 16), 2, np.sqrt((25 * 100 + 25 * 9) / 100),
                             np.sqrt((36 * 100 + 16 * 4) / 64), np.nan]
        expected_obs_date = np.ma.array(
            [datetime(2025, 5, 8),
             datetime(2025, 5, 10),
             datetime(2025, 5, 9),
             datetime(2025, 5, 9), np.ma.masked])

        combine_two = combine_intensity_map_data([map_1, map_2])
        np.testing.assert_equal(combine_two.ena_intensity, expected_combined_intensity)
        np.testing.assert_equal(combine_two.ena_intensity_sys_err, expected_sys_err)
        np.testing.assert_equal(combine_two.ena_intensity_stat_unc, expected_stat_unc)
        np.testing.assert_equal(combine_two.exposure_factor, expected_combined_exposure)
        np.testing.assert_equal(combine_two.obs_date.mask, expected_obs_date.mask)
        np.testing.assert_equal(combine_two.obs_date, expected_obs_date)

    @patch('imap_l3_processing.maps.map_models.combine_intensity_map_data')
    def test_combine_rectangular_intensity_map_data(self, mock_combine_intensity_map_data):

        expected_coords = RectangularCoords(latitude_delta=np.array([1]), longitude_delta=np.array([1]),
                                            latitude_label=np.array(["one"]), longitude_label=np.array(["one"]), )
        base_map = RectangularIntensityMapData(
            intensity_map_data=sentinel.data_1,
            coords=expected_coords
        )

        second_map = RectangularIntensityMapData(
            intensity_map_data=sentinel.data_2,
            coords=expected_coords
        )

        maps = [base_map, second_map]
        combined_map = combine_rectangular_intensity_map_data(maps)
        mock_combine_intensity_map_data.assert_called_with([sentinel.data_1, sentinel.data_2])

        self.assertEqual(combined_map.intensity_map_data, mock_combine_intensity_map_data.return_value)
        self.assertEqual(combined_map.coords, expected_coords)

    def test_combine_rectangular_intensity_map_data_errors_if_coords_not_matching(self):
        delta_array = np.array([1])
        label_array = np.array(["one"])

        def make_data(lat_delta=delta_array, lon_delta=delta_array, lat_label=label_array, lon_label=label_array):
            return RectangularIntensityMapData(
                intensity_map_data=self.construct_intensity_data_with_all_zero_fields(),
                coords=RectangularCoords(
                    latitude_delta=lat_delta,
                    longitude_delta=lon_delta,
                    latitude_label=lat_label,
                    longitude_label=lon_label,
                )
            )

        base_map = make_data()
        cases = [
            ("lat_delta", make_data(lat_delta=np.array([2]))),
            ("lon_delta", make_data(lon_delta=np.array([2]))),
            ("lat_label", make_data(lat_label=np.array(["two"]))),
            ("lon_label", make_data(lon_label=np.array(["two"]))),
        ]
        for name, not_matching_map in cases:
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    combine_rectangular_intensity_map_data([base_map, not_matching_map])

    def test_healpix_map_nside_property(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l2_maps' / 'test_l2_map.cdf'
        actual = HealPixIntensityMapData.read_from_path(path_to_cdf)
        expected_nside = 2

        self.assertIsInstance(actual.coords.nside, int)
        self.assertEqual(expected_nside, actual.coords.nside)

    def test_ultra_l2_map_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l2_maps' / 'ultra45-6months-copied-from-hi.cdf'

        map_data = HealPixIntensityMapData.read_from_path(path_to_cdf)
        actual_intensity_data = map_data.intensity_map_data
        actual_coords = map_data.coords

        expected_epoch = datetime(2025, 4, 18, 15, 51, 57, 171732)
        with CDF(str(path_to_cdf)) as expected:
            date_range_var = read_variable_and_mask_fill_values(expected["obs_date_range"])
            obs_date = read_variable_and_mask_fill_values(expected["obs_date"])

            self.assertEqual(expected_epoch, actual_intensity_data.epoch[0])
            np.testing.assert_array_equal(expected["epoch_delta"][...], actual_intensity_data.epoch_delta)
            np.testing.assert_array_equal(expected["energy"][...], actual_intensity_data.energy)
            np.testing.assert_array_equal(expected["energy_delta_plus"][...], actual_intensity_data.energy_delta_plus)
            np.testing.assert_array_equal(expected["energy_delta_minus"][...], actual_intensity_data.energy_delta_minus)
            np.testing.assert_array_equal(expected["energy_label"][...], actual_intensity_data.energy_label)
            np.testing.assert_array_equal(expected["latitude"][...], actual_intensity_data.latitude)
            np.testing.assert_array_equal(expected["longitude"][...], actual_intensity_data.longitude)
            np.testing.assert_array_equal(expected["exposure_factor"][...], actual_intensity_data.exposure_factor)
            np.testing.assert_array_equal(obs_date, actual_intensity_data.obs_date)
            np.testing.assert_array_equal(date_range_var, actual_intensity_data.obs_date_range)
            np.testing.assert_array_equal(expected["solid_angle"][...], actual_intensity_data.solid_angle)
            np.testing.assert_array_equal(expected["ena_intensity"][...], actual_intensity_data.ena_intensity)
            np.testing.assert_array_equal(expected["ena_intensity_stat_unc"][...],
                                          actual_intensity_data.ena_intensity_stat_unc)
            np.testing.assert_array_equal(expected["ena_intensity_sys_err"][...],
                                          actual_intensity_data.ena_intensity_sys_err)
            np.testing.assert_array_equal(expected["pixel_index"][...], actual_coords.pixel_index)
            np.testing.assert_array_equal(expected["pixel_index_label"][...], actual_coords.pixel_index_label)

    def test_read_intensity_map_with_rectangular_cords_data_from_cdf(self):

        rng = np.random.default_rng()
        with tempfile.TemporaryDirectory() as temp_dir:
            pathname = os.path.join(temp_dir, "test_cdf")
            with CDF(pathname, '') as cdf:
                cdf.col_major(True)

                ena_intensity = rng.random((1, 9, 90, 45))
                energy = rng.random(9)
                energy_delta_plus = rng.random(9)
                energy_delta_minus = rng.random(9)
                energy_label = energy.astype(str)
                ena_intensity_stat_unc = rng.random(ena_intensity.shape)
                ena_intensity_sys_err = rng.random(ena_intensity.shape)

                epoch = np.array([datetime.now()])
                epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
                exposure = np.full(ena_intensity.shape[:-1], 1.0)
                lat = np.arange(-88.0, 92.0, 4.0)
                lat_delta = np.full(lat.shape, 2.0)
                lat_label = [f"{x} deg" for x in lat]
                lon = np.arange(0.0, 360.0, 4.0)
                lon_delta = np.full(lon.shape, 2.0)
                lon_label = [f"{x} deg" for x in lon]

                obs_date = np.full(ena_intensity.shape, datetime.now())
                obs_date_range = np.full(ena_intensity.shape, ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY * 2)
                solid_angle = build_solid_angle_map(4)
                solid_angle = solid_angle[np.newaxis, ...]

                cdf.new("epoch", epoch)
                cdf.new("energy", energy, recVary=False)
                cdf.new("latitude", lat, recVary=False)
                cdf.new("latitude_delta", lat_delta, recVary=False)
                cdf.new("latitude_label", lat_label, recVary=False)
                cdf.new("longitude", lon, recVary=False)
                cdf.new("longitude_delta", lon_delta, recVary=False)
                cdf.new("longitude_label", lon_label, recVary=False)
                cdf.new("ena_intensity", ena_intensity, recVary=True)
                cdf.new("ena_intensity_stat_unc", ena_intensity_stat_unc, recVary=True)
                cdf.new("ena_intensity_sys_err", ena_intensity_sys_err, recVary=True)
                cdf.new("exposure_factor", exposure, recVary=True)
                cdf.new("obs_date", obs_date, recVary=True)
                cdf.new("obs_date_range", obs_date_range, recVary=True)
                cdf.new("solid_angle", solid_angle, recVary=True)
                cdf.new("epoch_delta", epoch_delta, recVary=True)
                cdf.new("energy_delta_plus", energy_delta_plus, recVary=False)
                cdf.new("energy_delta_minus", energy_delta_minus, recVary=False)
                cdf.new("energy_label", energy_label, recVary=False)

                for var in cdf:
                    cdf[var].attrs['FILLVAL'] = 1000000

            for path in [pathname, Path(pathname)]:
                with self.subTest(path=path):
                    result = RectangularIntensityMapData.read_from_path(path)
                    self.assertIsInstance(result, RectangularIntensityMapData)

                    rectangular_coords = result.coords
                    map_data = result.intensity_map_data

                    np.testing.assert_array_equal(epoch, map_data.epoch)
                    np.testing.assert_array_equal(epoch_delta, map_data.epoch_delta)
                    np.testing.assert_array_equal(energy, map_data.energy)
                    np.testing.assert_array_equal(energy_delta_plus, map_data.energy_delta_plus)
                    np.testing.assert_array_equal(energy_delta_minus, map_data.energy_delta_minus)
                    np.testing.assert_array_equal(energy_label, map_data.energy_label)
                    np.testing.assert_array_equal(lat, map_data.latitude)
                    np.testing.assert_array_equal(lat_delta, rectangular_coords.latitude_delta)
                    np.testing.assert_array_equal(lat_label, rectangular_coords.latitude_label)
                    np.testing.assert_array_equal(lon, map_data.longitude)
                    np.testing.assert_array_equal(lon_delta, rectangular_coords.longitude_delta)
                    np.testing.assert_array_equal(lon_label, rectangular_coords.longitude_label)
                    np.testing.assert_array_equal(ena_intensity, map_data.ena_intensity)
                    np.testing.assert_array_equal(ena_intensity_stat_unc, map_data.ena_intensity_stat_unc)
                    np.testing.assert_array_equal(ena_intensity_sys_err, map_data.ena_intensity_sys_err)
                    np.testing.assert_array_equal(exposure, map_data.exposure_factor)
                    np.testing.assert_array_equal(obs_date, map_data.obs_date)
                    np.testing.assert_array_equal(obs_date_range, map_data.obs_date_range)
                    np.testing.assert_array_equal(solid_angle, map_data.solid_angle)

    def test_fill_values_in_read_rectangular_intensity_map_data_from_cdf(self):
        path = get_test_data_folder() / 'hi' / 'fake_l2_maps' / 'l2_map_with_fill_values.cdf'
        result = RectangularIntensityMapData.read_from_path(path)
        map_data = result.intensity_map_data
        coords = result.coords

        with CDF(str(path)) as cdf:
            np.testing.assert_array_equal(map_data.epoch, cdf["epoch"], )

            self.assertTrue(np.all(map_data.epoch_delta.mask))
            self.assertTrue(np.all(map_data.obs_date.mask))
            self.assertTrue(np.all(map_data.obs_date_range.mask))

            np.testing.assert_array_equal(map_data.energy, np.full_like(cdf["energy"], np.nan))
            np.testing.assert_array_equal(map_data.energy_delta_plus, np.full_like(cdf["energy_delta_plus"], np.nan))
            np.testing.assert_array_equal(map_data.energy_delta_minus, np.full_like(cdf["energy_delta_minus"], np.nan))
            np.testing.assert_array_equal(map_data.latitude, np.full_like(cdf["latitude"], np.nan))
            np.testing.assert_array_equal(coords.latitude_delta, np.full_like(cdf["latitude_delta"], np.nan))
            np.testing.assert_array_equal(map_data.longitude, np.full_like(cdf["longitude"], np.nan))
            np.testing.assert_array_equal(coords.longitude_delta, np.full_like(cdf["longitude_delta"], np.nan))
            np.testing.assert_array_equal(map_data.ena_intensity, np.full_like(cdf["ena_intensity"], np.nan))
            np.testing.assert_array_equal(map_data.ena_intensity_stat_unc,
                                          np.full_like(cdf["ena_intensity_stat_unc"], np.nan))
            np.testing.assert_array_equal(map_data.ena_intensity_sys_err,
                                          np.full_like(cdf["ena_intensity_sys_err"], np.nan))
            np.testing.assert_array_equal(map_data.exposure_factor, np.full_like(cdf["exposure_factor"], np.nan))
            np.testing.assert_array_equal(map_data.solid_angle, np.full_like(cdf["solid_angle"], np.nan))


if __name__ == '__main__':
    unittest.main()
