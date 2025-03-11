import unittest
from datetime import datetime, timedelta

import numpy as np
from spacepy import pycdf

from imap_processing.hi.l3 import models
from imap_processing.hi.l3.models import HiL3SpectralIndexDataProduct
from imap_processing.models import InputMetadata, DataProductVariable


class TestModels(unittest.TestCase):
    def test_read_from_cdf(self):
        pass

    def test_spectral_index_to_data_products(self):
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        lat = np.array([0, 45])
        long = np.array([0, 45])
        energy = np.array([10, 20])
        epoch = np.array([datetime.now()])
        flux = np.array([[[[1, 2], [10, 20]], [[100, 200], [1000, 2000]]]])
        variance = np.ones_like(flux)
        spectral_index = np.ones([len(epoch), len(long), len(lat)])

        expected_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(models.LAT_VAR_NAME, lat, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.LONG_VAR_NAME, long, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.ENERGY_VAR_NAME, energy, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.FLUX_VAR_NAME, flux, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.VARIANCE_VAR_NAME, variance, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.SPECTRAL_INDEX_VAR_NAME, spectral_index, cdf_data_type=pycdf.const.CDF_DOUBLE),
        ]

        spectral_index_product = HiL3SpectralIndexDataProduct(
            input_metadata=input_metadata,
            epoch=epoch,
            energy=energy,
            variance=variance,
            flux=flux,
            lat=lat,
            lon=long,
            spectral_fit_index=spectral_index,
            energy_deltas=None,
            counts=None,
            counts_uncertainty=None,
            epoch_delta=None,
            exposure=None,
            sensitivity=None,
        )

        actual_variables = spectral_index_product.to_data_product_variables()

        self.assertEqual(expected_variables, actual_variables)
