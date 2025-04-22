import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable_and_mask_fill_values
from imap_l3_processing.models import InputMetadata, DataProductVariable
from imap_l3_processing.ultra.l3.models import UltraGlowsL3eData, UltraL1CPSet, UltraL3SurvivalCorrectedDataProduct, \
    UltraL2Map
from tests.test_helpers import get_test_data_folder


class TestModels(unittest.TestCase):

    def test_to_dataproduct_variables(self):
        input_metadata = InputMetadata(instrument="ultra",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        sp_corrected_ultra_dataproduct = UltraL3SurvivalCorrectedDataProduct(
            input_metadata=input_metadata,
            epoch=sentinel.epoch,
            epoch_delta=sentinel.epoch_delta,
            energy=sentinel.energy,
            energy_label=sentinel.energy_label,
            energy_delta_minus=sentinel.energy_delta_minus,
            energy_delta_plus=sentinel.energy_delta_plus,
            pixel_index=sentinel.pixel_index,
            pixel_index_label=sentinel.pixel_index_label,
            latitude=sentinel.latitude,
            longitude=sentinel.longitude,
            ena_intensity=sentinel.ena_intensity,
            ena_intensity_stat_unc=sentinel.ena_intensity_stat_unc,
            ena_intensity_sys_err=sentinel.ena_intensity_sys_err,
            exposure_factor=sentinel.exposure_factor,
            obs_date=sentinel.obs_date,
            obs_date_range=sentinel.obs_date_range,
            solid_angle=sentinel.solid_angle,
        )

        from imap_l3_processing.ultra.l3 import models
        expected_dataproduct_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(models.PIXEL_INDEX_VAR_NAME, sentinel.pixel_index),
            DataProductVariable(models.PIXEL_INDEX_LABEL_VAR_NAME, sentinel.pixel_index_label),
            DataProductVariable(models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(models.ENA_INTENSITY_VAR_NAME, sentinel.ena_intensity),
            DataProductVariable(models.ENA_INTENSITY_STAT_UNC_VAR_NAME, sentinel.ena_intensity_stat_unc),
            DataProductVariable(models.ENA_INTENSITY_SYS_ERR_VAR_NAME, sentinel.ena_intensity_sys_err),
            DataProductVariable(models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
        ]

        actual_variables = sp_corrected_ultra_dataproduct.to_data_product_variables()

        self.assertEqual(expected_dataproduct_variables, actual_variables)

    def test_glows_l3e_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l3e_survival_probabilities' / 'imap_glows_l3e_survival-probabilities-ultra_20250416_v001.cdf'

        actual = UltraGlowsL3eData.read_from_path(path_to_cdf)

        with CDF(str(path_to_cdf)) as expected:
            expected_epoch = datetime(2025, 4, 16, 12, 0)
            self.assertEqual(expected_epoch, actual.epoch)
            np.testing.assert_array_equal(expected['energy'][...], actual.energy)
            np.testing.assert_array_equal(expected['latitude'][...], actual.latitude)
            np.testing.assert_array_equal(expected['longitude'][...], actual.longitude)
            np.testing.assert_array_equal(expected['healpix_index'][...], actual.healpix_index)
            np.testing.assert_array_equal(expected['probability_of_survival'][...], actual.survival_probability)

    def test_ultra_l1c_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l1c_psets' / 'imap_ultra_l1c_45sensor-pset_20250101_v001.cdf'

        actual = UltraL1CPSet.read_from_path(path_to_cdf)

        expected_epoch = datetime(2025, 1, 1, 0, 0)
        with CDF(str(path_to_cdf)) as expected:
            self.assertEqual(expected_epoch, actual.epoch)
            np.testing.assert_array_equal(expected["counts"][...], actual.counts)
            np.testing.assert_array_equal(expected["energy"][...], actual.energy)
            np.testing.assert_array_equal(expected["exposure_time"][...], actual.exposure)
            np.testing.assert_array_equal(expected["healpix_index"][...], actual.healpix_index)
            np.testing.assert_array_equal(expected["latitude"][...], actual.latitude)
            np.testing.assert_array_equal(expected["longitude"][...], actual.longitude)
            np.testing.assert_array_equal(expected["sensitivity"][...], actual.sensitivity)

    def test_ultra_l2_map_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l2_maps' / 'hi45-6months.cdf'

        actual = UltraL2Map.read_from_path(path_to_cdf)

        expected_epoch = datetime(2025, 4, 18, 15, 51, 57, 171732)
        with CDF(str(path_to_cdf)) as expected:
            date_range_var = read_variable_and_mask_fill_values(expected["obs_date_range"])
            obs_date = read_variable_and_mask_fill_values(expected["obs_date"])

            self.assertEqual(expected_epoch, actual.epoch[0])
            np.testing.assert_array_equal(expected["epoch_delta"][...], actual.epoch_delta)
            np.testing.assert_array_equal(expected["energy"][...], actual.energy)
            np.testing.assert_array_equal(expected["energy_delta_plus"][...], actual.energy_delta_plus)
            np.testing.assert_array_equal(expected["energy_delta_minus"][...], actual.energy_delta_minus)
            np.testing.assert_array_equal(expected["energy_label"][...], actual.energy_label)
            np.testing.assert_array_equal(expected["latitude"][...], actual.latitude)
            np.testing.assert_array_equal(expected["longitude"][...], actual.longitude)
            np.testing.assert_array_equal(expected["exposure_factor"][...], actual.exposure_factor)
            np.testing.assert_array_equal(obs_date, actual.obs_date)
            np.testing.assert_array_equal(date_range_var, actual.obs_date_range)
            np.testing.assert_array_equal(expected["solid_angle"][...], actual.solid_angle)
            np.testing.assert_array_equal(expected["ena_intensity"][...], actual.ena_intensity)
            np.testing.assert_array_equal(expected["ena_intensity_stat_unc"][...], actual.ena_intensity_stat_unc)
            np.testing.assert_array_equal(expected["ena_intensity_sys_err"][...], actual.ena_intensity_sys_err)
            np.testing.assert_array_equal(expected["pixel_index"][...], actual.pixel_index)
            np.testing.assert_array_equal(expected["pixel_index_label"][...], actual.pixel_index_label)
