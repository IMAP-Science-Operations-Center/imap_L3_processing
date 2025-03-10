from datetime import datetime
from unittest.mock import Mock, call

import numpy as np
from spacepy import pycdf

from imap_processing.cdf.cdf_utils import write_cdf, read_variable
from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.models import DataProduct, DataProductVariable, InputMetadata
from tests.temp_file_test_case import TempFileTestCase


class TestCdfUtils(TempFileTestCase):
    def test_write_cdf(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        var_without_explicit_type, int_var, float_var, time_var, non_rec_varying_var = data.to_data_product_variables()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"variable_attr1": "var_val1"},
            {"variable_attr1": "var_val1", "FILLVAL": -9223372036854775808},
            {"variable_attr1": "var_val1", "FILLVAL": -1e31},
            {"variable_attr3": "var_val3", "FILLVAL": -9223372036854775808},
            {"variable_attr5": "var_val5", "variable_attr6": "var_val6"},
        ]

        write_cdf(path, data, attribute_manager)

        expected_data_product_logical_source = "imap_instrument_data-level_descriptor"
        attribute_manager.get_global_attributes.assert_called_once_with(expected_data_product_logical_source)
        attribute_manager.get_variable_attributes.assert_has_calls(
            [call(var.name) for var in data.to_data_product_variables()]
        )

        with pycdf.CDF(path) as actual_cdf:
            self.assertTrue(actual_cdf.col_major())
            self.assertEqual('global_val1', actual_cdf.attrs['global1'][...][0])
            self.assertEqual('global_val2', actual_cdf.attrs['global2'][...][0])

            np.testing.assert_array_equal(var_without_explicit_type.value,
                                          actual_cdf[var_without_explicit_type.name][...])
            self.assertEqual('var_val1', actual_cdf[var_without_explicit_type.name].attrs['variable_attr1'])
            self.assertEqual(pycdf.const.CDF_INT8.value, actual_cdf[var_without_explicit_type.name].type())
            self.assertTrue(actual_cdf[var_without_explicit_type.name].rv())

            np.testing.assert_array_equal(int_var.value, actual_cdf.raw_var(int_var.name))
            self.assertEqual('var_val1', actual_cdf[int_var.name].attrs['variable_attr1'])
            self.assertEqual(-9223372036854775808, actual_cdf[int_var.name].attrs['FILLVAL'])
            self.assertEqual(pycdf.const.CDF_INT8.value, actual_cdf[int_var.name].attrs.type("FILLVAL"))
            self.assertEqual(pycdf.const.CDF_INT8.value, actual_cdf[int_var.name].type())
            self.assertTrue(actual_cdf[int_var.name].rv())

            np.testing.assert_array_equal(float_var.value, actual_cdf.raw_var(float_var.name))
            self.assertEqual('var_val1', actual_cdf[float_var.name].attrs['variable_attr1'])
            self.assertEqual(-1e31, actual_cdf[float_var.name].attrs['FILLVAL'])
            self.assertEqual(pycdf.const.CDF_REAL4.value, actual_cdf[float_var.name].attrs.type("FILLVAL"))
            self.assertEqual(pycdf.const.CDF_REAL4.value, actual_cdf[float_var.name].type())
            self.assertTrue(actual_cdf[float_var.name].rv())

            np.testing.assert_array_equal(time_var.value, actual_cdf.raw_var(time_var.name))
            self.assertEqual('var_val3', actual_cdf[time_var.name].attrs['variable_attr3'])
            self.assertEqual(datetime(9999, 12, 31, 23, 59, 59, 999999), actual_cdf[time_var.name].attrs['FILLVAL'])
            self.assertEqual(pycdf.const.CDF_TIME_TT2000.value, actual_cdf[time_var.name].attrs.type("FILLVAL"))
            self.assertEqual(pycdf.const.CDF_TIME_TT2000.value, actual_cdf[time_var.name].type())
            self.assertTrue(actual_cdf[time_var.name].rv())

            self.assertEqual(non_rec_varying_var.value, actual_cdf[non_rec_varying_var.name][...])
            self.assertEqual('var_val5', actual_cdf[non_rec_varying_var.name].attrs['variable_attr5'])
            self.assertEqual('var_val6', actual_cdf[non_rec_varying_var.name].attrs['variable_attr6'])
            self.assertEqual(pycdf.const.CDF_BYTE.value, actual_cdf[non_rec_varying_var.name].type())
            self.assertFalse(actual_cdf[non_rec_varying_var.name].rv())

    def test_write_cdf_trims_numbers_in_logical_source_when_fetching_global_metadata(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.side_effect = [KeyError("Logical Source Not Found"),
                                                               {"global1": "global_val1", "global2": "global_val2"}]
        attribute_manager.get_variable_attributes.return_value = {}
        expected_data_product_logical_source = "imap_instrument_data-level_descriptor-10100"
        data.input_metadata.descriptor = "descriptor-10100"

        write_cdf(path, data, attribute_manager)

        attribute_manager.get_global_attributes.assert_has_calls(
            [call(expected_data_product_logical_source), call("imap_instrument_data-level_descriptor-")])

        with pycdf.CDF(path) as actual_cdf:
            self.assertTrue(actual_cdf.col_major())
            self.assertEqual('global_val1', str(actual_cdf.attrs['global1']))
            self.assertEqual('global_val2', str(actual_cdf.attrs['global2']))

    def test_write_cdf_replaces_nan_with_fill_value(self):
        epoch_data = [datetime(2025, 3, 7, 17, 0)]

        class DataProductWithNan(DataProduct):
            def __init__(self):
                self.input_metadata = Mock()

            def to_data_product_variables(self) -> list[DataProductVariable]:
                return [
                    DataProductVariable("epoch", epoch_data, pycdf.const.CDF_TIME_TT2000),
                    DataProductVariable("float_var", np.array([3, 5, np.nan, 9, np.nan, np.nan]),
                                        pycdf.const.CDF_REAL4),
                ]

        path = str(self.temp_directory / "write_cdf.cdf")
        data = DataProductWithNan()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {}
        attribute_manager.get_variable_attributes.side_effect = [
            {"VAR_NAME": "epoch", "FILLVAL": datetime.fromisoformat("9999-12-31T23:59:59.999999999")},
            {"VAR_NAME": "float_var", "FILLVAL": -1e31},
        ]

        write_cdf(path, data, attribute_manager)
        with pycdf.CDF(path) as actual_cdf:
            self.assertFalse(np.any(np.isnan(actual_cdf["float_var"][...])))
            np.testing.assert_array_equal(np.array([3, 5, -1e31, 9, -1e31, -1e31], dtype=np.float32),
                                          actual_cdf["float_var"][...], strict=True)
            np.testing.assert_array_equal(actual_cdf["epoch"][...], np.array(epoch_data), strict=True)

    def test_does_not_write_depend_0_variable_attribute_if_it_is_empty(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        _, _, regular_var, time_var, non_rec_varying_var = data.to_data_product_variables()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"ignored_attr": "var_val3", "variable_attr4": "var_val4", "DEPEND_0": ""},
            {"ignored_attr": "var_val3", "variable_attr4": "var_val4", "DEPEND_0": ""},
            {"variable_attr1": "var_val1", "variable_attr2": "var_val2", "DEPEND_0": "epoch"},
            {"variable_attr3": "var_val3", "variable_attr4": "var_val4", "DEPEND_0": ""},
            {"variable_attr5": "var_val5", "variable_attr6": "var_val6", "DEPEND_0": ""},
        ]

        write_cdf(path, data, attribute_manager)

        with pycdf.CDF(path) as actual_cdf:
            np.testing.assert_array_equal(regular_var.value, actual_cdf[regular_var.name][...])
            self.assertEqual('epoch', actual_cdf[regular_var.name].attrs['DEPEND_0'])
            self.assertFalse('DEPEND_0' in actual_cdf[time_var.name].attrs)
            self.assertFalse('DEPEND_0' in actual_cdf[non_rec_varying_var.name].attrs)

    def test_read_variable_replaces_fill_values_with_nan(self):
        path = str(self.temp_directory / "cdf.cdf")
        with pycdf.CDF(path, create=True) as actual_cdf:
            actual_cdf['var'] = np.array([1, 2, -1e31, 4, 5])
            actual_cdf['var'].attrs['FILLVAL'] = -1e31

            data = read_variable(actual_cdf['var'])

        np.testing.assert_equal(data, np.array([1, 2, np.nan, 4, 5]))


class TestDataProduct(DataProduct):
    def __init__(self):
        input_metadata = InputMetadata("instrument", "data-level", datetime(year=2025, month=5, day=10),
                                       datetime(year=2025, month=5, day=12), "version")

        self.input_metadata = input_metadata.to_upstream_data_dependency("descriptor")

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable("var_without_explicit_type", np.arange(0, 10)),
            DataProductVariable("int_var", np.arange(0, 10), pycdf.const.CDF_INT8),
            DataProductVariable("float_var", np.arange(0, 10), pycdf.const.CDF_REAL4),
            DataProductVariable("time_var", np.arange(10, 20), pycdf.const.CDF_TIME_TT2000),
            DataProductVariable("non_record_varying", 100, pycdf.const.CDF_BYTE, record_varying=False)
        ]
