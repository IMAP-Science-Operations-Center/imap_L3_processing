import unittest
from datetime import datetime, timedelta

import numpy as np
from spacepy import pycdf

from imap_l3_processing.hi.l3 import models
from imap_l3_processing.hi.l3.models import HiL3SpectralIndexDataProduct, HiL3SurvivalCorrectedDataProduct
from imap_l3_processing.models import InputMetadata, DataProductVariable


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
        spectral_index_error = np.full_like(spectral_index, 4)
        energy_deltas = np.arange(2 * 2).reshape(2, 2)
        counts = np.arange(2 * 2 * 2).reshape(1, 2, 2, 2) * 104.3
        counts_uncertainty = np.arange(2 * 2 * 2).reshape(1, 2, 2, 2) * 56.8
        epoch_delta = np.arange(1) * timedelta(hours=1)
        exposure = np.arange(2 * 2).reshape(1, 2, 2) * 20.4
        sensitivity = np.arange(2 * 2 * 2).reshape(1, 2, 2, 2)

        expected_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(models.LAT_VAR_NAME, lat, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.LONG_VAR_NAME, long, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.ENERGY_VAR_NAME, energy, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.FLUX_VAR_NAME, flux, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.VARIANCE_VAR_NAME, variance, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.ENERGY_DELTAS_VAR_NAME, energy_deltas, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.COUNTS_VAR_NAME, counts, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.COUNTS_UNCERTAINTY_VAR_NAME, counts_uncertainty,
                                cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.EPOCH_DELTA_VAR_NAME, epoch_delta, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.EXPOSURE_VAR_NAME, exposure, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.SENSITIVITY_VAR_NAME, sensitivity, cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.SPECTRAL_FIT_INDEX_VAR_NAME, spectral_index,
                                cdf_data_type=pycdf.const.CDF_DOUBLE),
            DataProductVariable(models.SPECTRAL_FIT_INDEX_ERROR_VAR_NAME, spectral_index_error,
                                cdf_data_type=pycdf.const.CDF_DOUBLE),
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
            energy_deltas=energy_deltas,
            counts=counts,
            counts_uncertainty=counts_uncertainty,
            epoch_delta=epoch_delta,
            exposure=exposure,
            sensitivity=sensitivity,
            spectral_fit_index_error=spectral_index_error
        )

        actual_variables = spectral_index_product.to_data_product_variables()

        self.assertEqual(expected_variables, actual_variables)

    def test_survival_probability_to_data_product_variables(self):
        input_metadata = InputMetadata(instrument="hi",
                                       data_level="",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="",
                                       )

        flux = np.array([[[[1, 2], [10, 20]], [[100, 200], [1000, 2000]]]])

        energy = np.array([10, 20])
        lat_data = np.array([0, 45])
        lon_data = np.array([0, 90])
        input_data = {
            "lat": lat_data,
            "lon": lon_data,
            "energy": energy,
            "epoch": np.array([datetime.now()]),
            "flux": flux,
            "variance": np.ones_like(flux),
            "energy_deltas": np.arange(2 * 2).reshape(2, 2),
            "counts": np.arange(2 * 2 * 2).reshape(1, 2, 2, 2) * 104.3,
            "counts_uncertainty": np.arange(2 * 2 * 2).reshape(1, 2, 2, 2) * 56.8,
            "epoch_delta": np.arange(1) * timedelta(hours=1),
            "exposure": np.arange(2 * 2).reshape(1, 2, 2) * 20.4,
            "sensitivity": np.arange(2 * 2 * 2).reshape(1, 2, 2, 2),
        }

        hi_l3_survival_corrected_data_product = HiL3SurvivalCorrectedDataProduct(input_metadata=input_metadata,
                                                                                 **input_data)

        actual_variables = hi_l3_survival_corrected_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(models.EPOCH_VAR_NAME, input_data["epoch"]),
            DataProductVariable(models.ENERGY_VAR_NAME, input_data["energy"]),
            DataProductVariable(models.ENERGY_DELTAS_VAR_NAME, input_data["energy_deltas"],
                                ),
            DataProductVariable(models.COUNTS_VAR_NAME, input_data["counts"]),
            DataProductVariable(models.COUNTS_UNCERTAINTY_VAR_NAME, input_data["counts_uncertainty"],
                                ),
            DataProductVariable(models.EPOCH_DELTA_VAR_NAME, input_data["epoch_delta"],
                                ),
            DataProductVariable(models.EXPOSURE_VAR_NAME, input_data["exposure"]),
            DataProductVariable(models.FLUX_VAR_NAME, flux),
            DataProductVariable(models.LAT_VAR_NAME, input_data["lat"]),
            DataProductVariable(models.LONG_VAR_NAME, input_data["lon"]),
            DataProductVariable(models.SENSITIVITY_VAR_NAME, input_data["sensitivity"],
                                ),
            DataProductVariable(models.VARIANCE_VAR_NAME, input_data["variance"]),
            DataProductVariable(models.ENERGY_LABEL_VAR_NAME,
                                [f"Energy Bin {e}" for e in energy]),
            DataProductVariable(models.LON_LABEL_VAR_NAME, [f"Lon {lon}" for lon in lon_data]),
            DataProductVariable(models.LAT_LABEL_VAR_NAME, [f"Lat {lat}" for lat in lat_data]),
        ]

        self.assertEqual(expected_variables, actual_variables)
