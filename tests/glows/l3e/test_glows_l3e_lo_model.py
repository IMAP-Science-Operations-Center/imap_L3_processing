import unittest
from unittest.mock import sentinel, Mock

from imap_l3_processing.glows.l3e.l3e_glows_lo_model import GlowsL3ELoData
from imap_l3_processing.models import DataProductVariable


class TestL3eLoModel(unittest.TestCase):
    def test_l3e_lo_model_to_data_product_variables(self):
        l3e_lo: GlowsL3ELoData = GlowsL3ELoData(
            Mock(),
            sentinel.epochs,
            sentinel.epoch_deltas,
            sentinel.energy,
            sentinel.spin_angle,
            sentinel.probability_of_survival
        )

        data_products = l3e_lo.to_data_product_variables()
        expected_data_products = [
            DataProductVariable("epoch", sentinel.epochs),
            DataProductVariable("epoch_delta", sentinel.epoch_deltas),
            DataProductVariable("energy", sentinel.energy),
            DataProductVariable("spin_angle", sentinel.spin_angle),
            DataProductVariable("probability_of_survival", sentinel.probability_of_survival),
        ]

        self.assertEqual(expected_data_products, data_products)
