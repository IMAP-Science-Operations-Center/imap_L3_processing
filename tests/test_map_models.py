import dataclasses
import unittest
from datetime import datetime
from unittest.mock import sentinel, Mock, patch

import numpy as np

from imap_l3_processing import map_models
from imap_l3_processing.map_models import RectangularCoords, SpectralIndexMapData, RectangularSpectralIndexMapData, \
    RectangularSpectralIndexDataProduct, RectangularIntensityMapData, IntensityMapData, RectangularIntensityDataProduct, \
    combine_rectangular_intensity_map_data, combine_intensity_map_data
from imap_l3_processing.models import DataProductVariable
from imap_l3_processing.utils import read_healpix_intensity_map_data_from_cdf
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

    def test_intensity_to_data_product_variables(self):
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

    @patch('imap_l3_processing.map_models.combine_intensity_map_data')
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
        actual = read_healpix_intensity_map_data_from_cdf(path_to_cdf)
        expected_nside = 2

        self.assertIsInstance(actual.coords.nside, int)
        self.assertEqual(expected_nside, actual.coords.nside)


if __name__ == '__main__':
    unittest.main()
