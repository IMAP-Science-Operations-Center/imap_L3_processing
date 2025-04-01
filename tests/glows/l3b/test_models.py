import unittest
from unittest.mock import sentinel

from imap_l3_processing.glows.l3b.models import GlowsL3BIonizationRate
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_to_data_product_variables(self):
        data = GlowsL3BIonizationRate(input_metadata=sentinel.input_metadata,
                                      epoch=sentinel.epoch,
                                      epoch_delta=sentinel.epoch_delta,
                                      cr=sentinel.cr,
                                      uv_anisotropy_factor=sentinel.uv_anisotropy_factor,
                                      lat_grid=sentinel.lat_grid,
                                      lat_grid_delta=sentinel.lat_grid_delta,
                                      sum_rate=sentinel.sum_rate,
                                      ph_rate=sentinel.ph_rate,
                                      cx_rate=sentinel.cx_rate,
                                      sum_uncert=sentinel.sum_uncert,
                                      ph_uncert=sentinel.ph_uncert,
                                      cx_uncert=sentinel.cx_uncert,
                                      lat_grid_label=sentinel.lat_grid_label,
                                      )

        variables = data.to_data_product_variables()
        self.assertEqual(13, len(variables))

        variables = iter(variables)
        self.assert_variable_attributes(next(variables), sentinel.epoch, "epoch")
        self.assert_variable_attributes(next(variables), sentinel.epoch_delta, "epoch_delta")
        self.assert_variable_attributes(next(variables), sentinel.cr, "cr")
        self.assert_variable_attributes(next(variables), sentinel.uv_anisotropy_factor, "uv_anisotropy_factor")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid, "lat_grid")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_delta, "lat_grid_delta")
        self.assert_variable_attributes(next(variables), sentinel.sum_rate, "sum_rate")
        self.assert_variable_attributes(next(variables), sentinel.ph_rate, "ph_rate")
        self.assert_variable_attributes(next(variables), sentinel.cx_rate, "cx_rate")
        self.assert_variable_attributes(next(variables), sentinel.sum_uncert, "sum_uncert")
        self.assert_variable_attributes(next(variables), sentinel.ph_uncert, "ph_uncert")
        self.assert_variable_attributes(next(variables), sentinel.cx_uncert, "cx_uncert")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_label, "lat_grid_label")


if __name__ == '__main__':
    unittest.main()
