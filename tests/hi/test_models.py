import dataclasses
import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel

import numpy as np

from imap_l3_processing.hi.l3 import models
from imap_l3_processing.hi.l3.models import HiL3SpectralIndexDataProduct, HiL3SurvivalCorrectedDataProduct, \
    combine_maps
from imap_l3_processing.models import DataProductVariable, InputMetadata


class TestModels(unittest.TestCase):
    def test_spectral_index_to_data_products(self):
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        hi_l3_spectral_index_data_product = HiL3SpectralIndexDataProduct(
            input_metadata=input_metadata,
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=sentinel.energy,
            energy_delta_plus=sentinel.energy_delta_plus,
            energy_delta_minus=sentinel.energy_delta_minus,
            energy_label=sentinel.energy_label,
            latitude=sentinel.latitude,
            latitude_delta=sentinel.latitude_delta,
            latitude_label=sentinel.latitude_label,
            longitude=sentinel.longitude,
            longitude_delta=sentinel.longitude_delta,
            longitude_label=sentinel.longitude_label,
            exposure_factor=sentinel.exposure_factor,
            obs_date=sentinel.obs_date,
            obs_date_range=sentinel.obs_date_range,
            solid_angle=sentinel.solid_angle,
            ena_spectral_index=sentinel.ena_spectral_index,
            ena_spectral_index_stat_unc=sentinel.ena_spectral_index_stat_unc,
        )

        actual_variables = hi_l3_spectral_index_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta),
            DataProductVariable(models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label),
            DataProductVariable(models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta),
            DataProductVariable(models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label),
            DataProductVariable(models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(models.ENA_SPECTRAL_INDEX_VAR_NAME, sentinel.ena_spectral_index),
            DataProductVariable(models.ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, sentinel.ena_spectral_index_stat_unc),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_survival_probability_to_data_product_variables(self):
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        hi_l3_survival_corrected_data_product = HiL3SurvivalCorrectedDataProduct(
            input_metadata=input_metadata,
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=sentinel.energy,
            energy_delta_plus=sentinel.energy_delta_plus,
            energy_delta_minus=sentinel.energy_delta_minus,
            energy_label=sentinel.energy_label,
            latitude=sentinel.latitude,
            latitude_delta=sentinel.latitude_delta,
            latitude_label=sentinel.latitude_label,
            longitude=sentinel.longitude,
            longitude_delta=sentinel.longitude_delta,
            longitude_label=sentinel.longitude_label,
            exposure_factor=sentinel.exposure_factor,
            obs_date=sentinel.obs_date,
            obs_date_range=sentinel.obs_date_range,
            solid_angle=sentinel.solid_angle,
            ena_intensity=sentinel.ena_intensity,
            ena_intensity_stat_unc=sentinel.ena_intensity_stat_unc,
            ena_intensity_sys_err=sentinel.ena_intensity_sys_err,
        )
        actual_variables = hi_l3_survival_corrected_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta),
            DataProductVariable(models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label),
            DataProductVariable(models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta),
            DataProductVariable(models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label),
            DataProductVariable(models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),

            DataProductVariable(models.ENA_INTENSITY_VAR_NAME, sentinel.ena_intensity),
            DataProductVariable(models.ENA_INTENSITY_STAT_UNC_VAR_NAME, sentinel.ena_intensity_stat_unc),
            DataProductVariable(models.ENA_INTENSITY_SYS_ERR_VAR_NAME, sentinel.ena_intensity_sys_err),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def construct_map_data_product_with_all_zero_fields(self) -> HiL3SurvivalCorrectedDataProduct:
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        return HiL3SurvivalCorrectedDataProduct(
            input_metadata=input_metadata,
            epoch=np.array([0]),
            epoch_delta=np.array([0]),
            energy=np.array([0]),
            energy_delta_plus=np.array([0]),
            energy_delta_minus=np.array([0]),
            energy_label=np.array([0]),
            latitude=np.array([0]),
            latitude_delta=np.array([0]),
            latitude_label=np.array([0]),
            longitude=np.array([0]),
            longitude_delta=np.array([0]),
            longitude_label=np.array([0]),
            exposure_factor=np.array([1]),
            obs_date=np.array([0]),
            obs_date_range=np.array([0]),
            solid_angle=np.array([0]),
            ena_intensity=np.array([0]),
            ena_intensity_stat_unc=np.array([0]),
            ena_intensity_sys_err=np.array([0]),
        )

    def test_combine_maps_does_nothing_when_passed_a_single_map(self):
        map_1 = self.construct_map_data_product_with_all_zero_fields()

        combine_one = combine_maps([map_1])
        np.testing.assert_equal(dataclasses.asdict(combine_one), dataclasses.asdict(map_1))

    def test_combine_maps_throws_exception_when_fields_vary_that_should_not(self):
        map_1 = self.construct_map_data_product_with_all_zero_fields()

        fields_which_may_differ = {"ena_intensity", "ena_intensity_stat_unc", "ena_intensity_sys_err",
                                   "exposure_factor", "obs_date", "obs_date_range", "parent_file_names",
                                   "input_metadata"}
        for field in dataclasses.fields(HiL3SurvivalCorrectedDataProduct):
            map_with_difference = dataclasses.replace(map_1, **{field.name: np.array([10])})
            if field.name not in fields_which_may_differ:
                with self.assertRaises(AssertionError, msg=field.name):
                    combine_maps([map_1, map_with_difference])
            else:
                try:
                    combine_maps([map_1, map_with_difference])
                except:
                    self.fail(f"Differences in other fields should be alright: {field.name}")

    def test_combine_maps_does_a_time_weighted_average_of_intensity(self):
        map_1 = self.construct_map_data_product_with_all_zero_fields()
        map_1.ena_intensity = np.array([1, np.nan, 3, 4, np.nan])
        map_1.exposure_factor = np.array([1, 0, 5, 6, 0])
        map_1.ena_intensity_sys_err = np.array([1, np.nan, 10, 100, np.nan])
        map_1.ena_intensity_stat_unc = np.array([10, np.nan, 10, 10, np.nan])

        map_2 = self.construct_map_data_product_with_all_zero_fields()
        map_2.ena_intensity = np.array([5, 6, 7, 8, np.nan])
        map_2.exposure_factor = np.array([3, 1, 5, 2, 0])
        map_2.ena_intensity_sys_err = np.array([9, 4, 2, 0, np.nan])
        map_2.ena_intensity_stat_unc = np.array([1, 2, 3, 4, np.nan])

        expected_combined_exposure = [4, 1, 10, 8, 0]
        expected_combined_intensity = [4, 6, 5, 5, np.nan]
        expected_sys_err = [7, 4, 6, 75, np.nan]
        expected_stat_unc = [np.sqrt((1 * 100 + 9 * 1) / 16), 2, np.sqrt((25 * 100 + 25 * 9) / 100),
                             np.sqrt((36 * 100 + 16 * 4) / 64), np.nan]

        combine_two = combine_maps([map_1, map_2])
        np.testing.assert_equal(combine_two.ena_intensity, expected_combined_intensity)
        np.testing.assert_equal(combine_two.ena_intensity_sys_err, expected_sys_err)
        np.testing.assert_equal(combine_two.ena_intensity_stat_unc, expected_stat_unc)
        np.testing.assert_equal(combine_two.exposure_factor, expected_combined_exposure)
