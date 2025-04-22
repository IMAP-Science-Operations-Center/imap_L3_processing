import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel, Mock

import numpy as np

from imap_l3_processing.models import DataProductVariable

from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from tests.test_helpers import get_test_instrument_team_data_path


class TestL3eUltraModel(unittest.TestCase):
    def test_l3e_ultra_model_to_data_product_variables(self):
        l3e_ultra: GlowsL3EUltraData = GlowsL3EUltraData(
            Mock(),
            sentinel.epochs,
            sentinel.epoch_deltas,
            sentinel.energy,
            sentinel.latitude,
            sentinel.longitude,
            sentinel.healpix_index,
            sentinel.probability_of_survival
        )

        data_products = l3e_ultra.to_data_product_variables()
        expected_data_products = [
            DataProductVariable("epoch", sentinel.epochs),
            DataProductVariable("epoch_delta", sentinel.epoch_deltas),
            DataProductVariable("energy", sentinel.energy),
            DataProductVariable("latitude", sentinel.latitude),
            DataProductVariable("longitude", sentinel.longitude),
            DataProductVariable("healpix_index", sentinel.healpix_index),
            DataProductVariable("probability_of_survival", sentinel.probability_of_survival),
        ]

        self.assertEqual(expected_data_products, data_products)

    def test_convert_dat_to_glows_l3e_ul_product(self):
        ul_file_path = get_test_instrument_team_data_path("glows/probSur.Imap.Ul_20090101_010101_2009.000.txt")
        expected_epoch = np.array(datetime(year=2009, month=1, day=1))
        expected_time_delta = np.array(timedelta(hours=12))

        expected_energy = np.array(
            [2.3751086, 3.0917682, 4.0246710, 5.2390656, 6.8198887, 8.8777057, 11.5564435, 15.0434572, 19.5826341,
             25.4914514, 33.1831812, 43.1957952, 56.2295915, 73.1961745, 95.2822137, 124.0324417, 161.4576950,
             210.1755551, 273.5934261, 356.1468544])

        row_1_expected_latitude = 87.07582
        row_1_expected_longitude = 45.00000
        row_1_probability_of_survival = np.array(
            [0.86450643E+00, 0.87973395E+00, 0.89379058E+00, 0.90690377E+00, 0.91915602E+00, 0.93067239E+00,
             0.94156318E+00, 0.95180802E+00, 0.96145060E+00, 0.97033853E+00, 0.97807026E+00, 0.98418278E+00,
             0.98846694E+00, 0.99120292E+00, 0.99291842E+00, 0.99405812E+00, 0.99489599E+00, 0.99556132E+00,
             0.99611769E+00, 0.99659974E+00])

        row_915_expected_latitude = np.nan
        row_915_expected_longitude = np.nan

        row_915_probability_of_survival = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan])

        row_3071_expected_latitude = -87.07582
        row_3071_expected_longitude = 315.00000
        row_3071_expected_probability_of_survival = np.array([
            0.87503997, 0.88938004, 0.90261715, 0.91481741, 0.92617171, 0.93674534, 0.94667530, 0.95597378, 0.96467314,
            0.97260995, 0.97946208, 0.98487928, 0.98871799, 0.99122513, 0.99284000, 0.99395244, 0.99478699, 0.99545964,
            0.99602705, 0.99651490
        ])

        expected_survival_probability_shape = (1, 20, 3072)

        expected_heal_pix = np.arange(0, 3072)
        mock_metadata = Mock()

        l3e_ul_product: GlowsL3EUltraData = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(mock_metadata,
                                                                                                  ul_file_path,
                                                                                                  expected_epoch,
                                                                                                  expected_time_delta)
        self.assertEqual(expected_epoch, l3e_ul_product.epoch)
        self.assertEqual(expected_time_delta, l3e_ul_product.epoch_delta)

        np.testing.assert_array_equal(l3e_ul_product.energy, expected_energy)

        np.testing.assert_array_equal(expected_heal_pix, l3e_ul_product.healpix_index)

        self.assertEqual(expected_survival_probability_shape, l3e_ul_product.probability_of_survival.shape)

        np.testing.assert_array_equal(l3e_ul_product.longitude[0], row_1_expected_longitude)
        np.testing.assert_array_equal(l3e_ul_product.latitude[0], row_1_expected_latitude)
        np.testing.assert_array_equal(l3e_ul_product.probability_of_survival[0].T[0, :], row_1_probability_of_survival)

        np.testing.assert_array_equal(l3e_ul_product.longitude[915], row_915_expected_longitude)
        np.testing.assert_array_equal(l3e_ul_product.latitude[915], row_915_expected_latitude)
        np.testing.assert_array_equal(l3e_ul_product.probability_of_survival[0].T[915, :],
                                      row_915_probability_of_survival)

        np.testing.assert_array_equal(l3e_ul_product.longitude[3071], row_3071_expected_longitude)
        np.testing.assert_array_equal(l3e_ul_product.latitude[3071], row_3071_expected_latitude)
        np.testing.assert_array_equal(l3e_ul_product.probability_of_survival[0].T[3071, :],
                                      row_3071_expected_probability_of_survival)
