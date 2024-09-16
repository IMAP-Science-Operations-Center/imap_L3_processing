from unittest.mock import Mock, call

import numpy as np
from spacepy import pycdf

from imap_processing.cdf.cdf_utils import write_cdf
from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.models import DataProduct, DataProductVariable
from tests.temp_file_test_case import TempFileTestCase


class TestCdfUtils(TempFileTestCase):
    def test_write_cdf(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        regular_var, time_var, non_rec_varying_var = data.to_data_product_variables()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"variable_attr1": "var_val1", "variable_attr2": "var_val2"},
            {"variable_attr3": "var_val3", "variable_attr4": "var_val4"},
            {"variable_attr5": "var_val5", "variable_attr6": "var_val6"},
        ]

        write_cdf(path, data, attribute_manager)

        attribute_manager.get_global_attributes.assert_called_once()
        attribute_manager.get_variable_attributes.assert_has_calls(
            [call(var.name) for var in data.to_data_product_variables()]
        )

        with pycdf.CDF(path) as actual_cdf:
            self.assertTrue(actual_cdf.col_major())
            self.assertEqual('global_val1', actual_cdf.attrs['global1'][...][0])
            self.assertEqual('global_val2', actual_cdf.attrs['global2'][...][0])

            np.testing.assert_array_equal(regular_var.value, actual_cdf[regular_var.name][...])
            self.assertEqual('var_val1', actual_cdf[regular_var.name].attrs['variable_attr1'])
            self.assertEqual('var_val2', actual_cdf[regular_var.name].attrs['variable_attr2'])
            self.assertEqual(pycdf.const.CDF_INT8.value, actual_cdf[regular_var.name].type())
            self.assertTrue(actual_cdf[regular_var.name].rv())
            np.testing.assert_array_equal(time_var.value, actual_cdf.raw_var(time_var.name))

            self.assertEqual('var_val3', actual_cdf[time_var.name].attrs['variable_attr3'])
            self.assertEqual('var_val4', actual_cdf[time_var.name].attrs['variable_attr4'])
            self.assertEqual(pycdf.const.CDF_TIME_TT2000.value, actual_cdf[time_var.name].type())
            self.assertTrue(actual_cdf[time_var.name].rv())

            self.assertEqual(non_rec_varying_var.value, actual_cdf[non_rec_varying_var.name][...])
            self.assertEqual('var_val5', actual_cdf[non_rec_varying_var.name].attrs['variable_attr5'])
            self.assertEqual('var_val6', actual_cdf[non_rec_varying_var.name].attrs['variable_attr6'])
            self.assertEqual(pycdf.const.CDF_BYTE.value, actual_cdf[non_rec_varying_var.name].type())
            self.assertFalse(actual_cdf[non_rec_varying_var.name].rv())


class TestDataProduct(DataProduct):
    def __init__(self):
        pass
    
    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable("var1", np.arange(0, 10)),
            DataProductVariable("var2", np.arange(10, 20), pycdf.const.CDF_TIME_TT2000),
            DataProductVariable("var3", 100, record_varying=False)
        ]
