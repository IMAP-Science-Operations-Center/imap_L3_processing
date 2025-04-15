import dataclasses
import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel

import numpy as np
from spacepy import pycdf

from imap_l3_processing.hi.l3 import models
from imap_l3_processing.hi.l3.models import HiL3SpectralIndexDataProduct, HiL3SurvivalCorrectedDataProduct, \
    HiDataProduct
from imap_l3_processing.models import InputMetadata, DataProductVariable, UpstreamDataDependency


class TestModels(unittest.TestCase):
    def test_spectral_index_to_data_products(self):
        input_metadata = UpstreamDataDependency(instrument="hi",
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
            ena_spectral_index_sys_err=sentinel.ena_spectral_index_sys_err
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
            DataProductVariable(models.ENA_SPECTRAL_INDEX_SYS_ERR_VAR_NAME, sentinel.ena_spectral_index_sys_err),
        ]

        self.assertEqual(expected_variables, actual_variables)

    def test_survival_probability_to_data_product_variables(self):
        input_metadata = UpstreamDataDependency(instrument="hi",
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
