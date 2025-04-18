import unittest
from unittest.mock import sentinel, Mock

from imap_l3_processing.models import DataProductVariable

from imap_l3_processing.glows.l3e.l3e_glows_ultra_model import GlowsL3EUltraData


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
