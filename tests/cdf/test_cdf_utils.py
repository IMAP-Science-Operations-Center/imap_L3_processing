from datetime import datetime
from unittest import mock
from unittest.mock import Mock, call, patch, sentinel

import numpy as np
from spacepy import pycdf

from imap_l3_processing.cdf.cdf_utils import write_cdf, read_variable_and_mask_fill_values, read_numeric_variable
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata
from tests.temp_file_test_case import TempFileTestCase


class TestCdfUtils(TempFileTestCase):
    def test_write_cdf(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        var_without_explicit_type, int_var, float_var, time_var, non_rec_varying_var = data.to_data_product_variables()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"variable_attr1": "var_val1", "DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "rV"},
            {"variable_attr1": "var_val1", "FILLVAL": -9223372036854775808, "DATA_TYPE": "CDF_INT8",
             "RECORD_VARYING": "RV"},
            {"variable_attr1": "var_val1", "FILLVAL": -1e31, "DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "RV"},
            {"variable_attr3": "var_val3", "FILLVAL": -9223372036854775808, "DATA_TYPE": "CDF_TIME_TT2000",
             "RECORD_VARYING": "NRV"},
            {"variable_attr5": "var_val5", "variable_attr6": "var_val6", "DATA_TY"
                                                                         "PE": "CDF_INT4",
             "RECORD_VARYING": "NRV"},
        ]

        write_cdf(path, data, attribute_manager)

        expected_data_product_logical_source = "imap_glows_l3_descriptor"
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
            self.assertEqual(pycdf.const.CDF_REAL4.value, actual_cdf[var_without_explicit_type.name].type())
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
            self.assertFalse(actual_cdf[time_var.name].rv())

            self.assertEqual('var_val5', actual_cdf[non_rec_varying_var.name].attrs['variable_attr5'])
            self.assertEqual('var_val6', actual_cdf[non_rec_varying_var.name].attrs['variable_attr6'])
            self.assertEqual(non_rec_varying_var.value, actual_cdf[non_rec_varying_var.name][...])
            self.assertEqual(pycdf.const.CDF_INT4.value, actual_cdf[non_rec_varying_var.name].type())
            self.assertFalse(actual_cdf[non_rec_varying_var.name].rv())

            actual_var_attributes = [attribute for k, v in actual_cdf.items() for attribute in v.attrs]
            self.assertNotIn("DATA_TYPE", actual_var_attributes)
            self.assertNotIn("RECORD_VARYING", actual_var_attributes)

    @patch("imap_l3_processing.cdf.cdf_utils.CDF")
    def test_write_cdf_compresses_variables(self, mock_new_cdf):
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"variable_attr1": "var_val1", "DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "rV"},
            {"variable_attr1": "var_val1", "FILLVAL": -9223372036854775808, "DATA_TYPE": "CDF_INT8",
             "RECORD_VARYING": "RV"},
            {"variable_attr1": "var_val1", "FILLVAL": -1e31, "DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "RV"},
            {"variable_attr3": "var_val3", "FILLVAL": -9223372036854775808, "DATA_TYPE": "CDF_TIME_TT2000",
             "RECORD_VARYING": "NRV"},
            {"variable_attr5": "var_val5", "variable_attr6": "var_val6", "DATA_TYPE": "CDF_INT4",
             "RECORD_VARYING": "NRV"},
        ]

        expected_data_product = TestDataProduct()

        write_cdf(sentinel.filepath, expected_data_product, attribute_manager)

        mock_new_cdf.return_value.__enter__.return_value.new.assert_has_calls(
            [
                call('var_without_explicit_type', mock.ANY, recVary=True, type=pycdf.const.CDF_REAL4, dims=mock.ANY,
                     compress=pycdf.const.GZIP_COMPRESSION, compress_param=7),
                call('int_var', mock.ANY, recVary=True, type=pycdf.const.CDF_INT8, dims=mock.ANY,
                     compress=pycdf.const.GZIP_COMPRESSION, compress_param=7),
                call('float_var', mock.ANY, recVary=True, type=pycdf.const.CDF_REAL4, dims=mock.ANY,
                     compress=pycdf.const.GZIP_COMPRESSION, compress_param=7),
                call('time_var', mock.ANY, recVary=False, type=pycdf.const.CDF_TIME_TT2000, dims=mock.ANY,
                     compress=None, compress_param=None),
                call('non_record_varying', mock.ANY, recVary=False, type=pycdf.const.CDF_INT4, dims=mock.ANY,
                     compress=None, compress_param=None),
            ]
        )

    def test_write_cdf_trims_numbers_in_logical_source_when_fetching_global_metadata(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.side_effect = [KeyError("Logical Source Not Found"),
                                                               {"global1": "global_val1", "global2": "global_val2"}]
        attribute_manager.get_variable_attributes.return_value = {"DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "NRV"}
        expected_data_product_logical_source = "imap_glows_l3_descriptor-10100"
        data.input_metadata.descriptor = "descriptor-10100"

        write_cdf(path, data, attribute_manager)

        attribute_manager.get_global_attributes.assert_has_calls(
            [call(expected_data_product_logical_source), call("imap_glows_l3_descriptor-")])

        with pycdf.CDF(path) as actual_cdf:
            self.assertTrue(actual_cdf.col_major())
            self.assertEqual('global_val1', str(actual_cdf.attrs['global1']))
            self.assertEqual('global_val2', str(actual_cdf.attrs['global2']))

    def test_write_cdf_replaces_nan_and_masked_with_fill_value(self):
        epoch_fillval = datetime.fromisoformat("9999-12-31T23:59:59.999999")

        epoch_data = np.ma.masked_array([datetime(2025, 3, 7, 17, 0), None], mask=[False, True])
        expected_cdf_epoch_data = np.array([datetime(2025, 3, 7, 17, 0), epoch_fillval])

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
            {"VAR_NAME": "epoch", "FILLVAL": epoch_fillval,
             "DATA_TYPE": "CDF_TIME_TT2000", "RECORD_VARYING": "NRV"},
            {"VAR_NAME": "float_var", "FILLVAL": -1e31, "DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "NRV"},
        ]

        write_cdf(path, data, attribute_manager)
        with pycdf.CDF(path) as actual_cdf:
            self.assertFalse(np.any(np.isnan(actual_cdf["float_var"][...])))
            np.testing.assert_array_equal(np.array([3, 5, -1e31, 9, -1e31, -1e31], dtype=np.float32),
                                          actual_cdf["float_var"][...], strict=True)
            np.testing.assert_array_equal(actual_cdf["epoch"][...], expected_cdf_epoch_data, strict=True)


    def test_can_write_empty_epoch_variable(self):
        epoch_data = []
        epoch_fillval = datetime.fromisoformat("9999-12-31T23:59:59.999999")

        class DataProductWithEmptyEpoch(DataProduct):
            def __init__(self):
                self.input_metadata = Mock()

            def to_data_product_variables(self) -> list[DataProductVariable]:
                return [
                    DataProductVariable("epoch", epoch_data, pycdf.const.CDF_TIME_TT2000),
                ]

        path = str(self.temp_directory / "write_cdf.cdf")
        data = DataProductWithEmptyEpoch()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {}
        attribute_manager.get_variable_attributes.side_effect = [
            {"VAR_NAME": "epoch", "FILLVAL": epoch_fillval,
             "DATA_TYPE": "CDF_TIME_TT2000", "RECORD_VARYING": "RV"},
            {"VAR_NAME": "float_var", "FILLVAL": -1e31, "DATA_TYPE": "CDF_REAL4", "RECORD_VARYING": "NRV"},
        ]

        write_cdf(path, data, attribute_manager)
        with pycdf.CDF(path) as actual_cdf:
            np.testing.assert_array_equal(actual_cdf["epoch"][...], np.array([], dtype=object), strict=True)



    def test_can_write_empty_variables_of_various_shapes(self):
        cases=[
            (np.zeros((0,5), dtype=object), "CDF_TIME_TT2000", "RV", datetime.fromisoformat("9999-12-31T23:59:59.999999")),
            (np.zeros((0), dtype=object), "CDF_TIME_TT2000", "RV", datetime.fromisoformat("9999-12-31T23:59:59.999999")),
            (np.zeros((0,5), dtype=float), "CDF_REAL8", "RV", -1e31),
            (np.zeros((0), dtype=float), "CDF_REAL8", "RV", -1e31),
        ]
        for input_data, data_type, record_varying, fill_val in cases:
            with self.subTest(input_data=input_data, data_type=data_type, rv=record_varying):
                class DataProductWithEmptyEpoch(DataProduct):
                    def __init__(self):
                        self.input_metadata = Mock()

                    def to_data_product_variables(self) -> list[DataProductVariable]:
                        return [
                            DataProductVariable("var", input_data, pycdf.const.CDF_REAL8),
                        ]

                path = self.temp_directory / "write_cdf.cdf"
                data = DataProductWithEmptyEpoch()
                attribute_manager = Mock(spec=ImapAttributeManager)
                attribute_manager.get_global_attributes.return_value = {}
                attribute_manager.get_variable_attributes.side_effect = [
                    {"VAR_NAME": "var", "FILLVAL": fill_val, "DATA_TYPE": data_type, "RECORD_VARYING": record_varying},
                ]

                write_cdf(str(path), data, attribute_manager)
                with pycdf.CDF(str(path)) as actual_cdf:
                    np.testing.assert_array_equal(actual_cdf["var"][...], input_data, strict=True)

                path.unlink()


    def test_does_not_write_depend_0_variable_attribute_if_it_is_empty(self):
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestDataProduct()
        _, _, regular_var, time_var, non_rec_varying_var = data.to_data_product_variables()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"ignored_attr": "var_val3", "variable_attr4": "var_val4", "DATA_TYPE": "CDF_REAL4",
             "RECORD_VARYING": "NRV",
             "DEPEND_0": ""},
            {"ignored_attr": "var_val3", "variable_attr4": "var_val4", "DATA_TYPE": "CDF_REAL4",
             "RECORD_VARYING": "NRV",
             "DEPEND_0": ""},
            {"variable_attr1": "var_val1", "variable_attr2": "var_val2", "DATA_TYPE": "CDF_REAL4",
             "RECORD_VARYING": "RV", "DEPEND_0": "epoch"},
            {"variable_attr3": "var_val3", "variable_attr4": "var_val4", "DATA_TYPE": "CDF_REAL4",
             "RECORD_VARYING": "NRV", "DEPEND_0": ""},
            {"variable_attr5": "var_val5", "variable_attr6": "var_val6", "DATA_TYPE": "CDF_REAL4",
             "RECORD_VARYING": "NRV", "DEPEND_0": ""},
        ]

        write_cdf(path, data, attribute_manager)

        with pycdf.CDF(path) as actual_cdf:
            np.testing.assert_array_equal(regular_var.value, actual_cdf[regular_var.name][...])
            self.assertEqual('epoch', actual_cdf[regular_var.name].attrs['DEPEND_0'])
            self.assertFalse('DEPEND_0' in actual_cdf[time_var.name].attrs)
            self.assertFalse('DEPEND_0' in actual_cdf[non_rec_varying_var.name].attrs)

    def test_read_variable_and_mask_fill_values(self):
        path = str(self.temp_directory / "cdf.cdf")
        with pycdf.CDF(path, create=True) as actual_cdf:
            datetime_fill = datetime(9999, 12, 31, 23, 59, 59, 999999)
            actual_cdf['date_var'] = [datetime(2025, 4, 16), datetime_fill]
            actual_cdf['date_var'].attrs['FILLVAL'] = datetime_fill

            int_fillval = -9223372036854775808
            actual_cdf['int_var'] = np.array([-1, 2, int_fillval, int_fillval + 1], dtype=np.int64)
            actual_cdf['int_var'].attrs['FILLVAL'] = int_fillval

            masked_datetime_data: np.ma.masked_array = read_variable_and_mask_fill_values(actual_cdf['date_var'])
            masked_int_data: np.ma.masked_array = read_variable_and_mask_fill_values(actual_cdf['int_var'])
        np.testing.assert_array_equal(masked_datetime_data.data,
                                      [datetime(2025, 4, 16), datetime_fill])
        np.testing.assert_equal(masked_datetime_data.mask, [False, True])

        np.testing.assert_equal(masked_int_data.data, np.array([-1, 2, int_fillval, int_fillval + 1], dtype=np.int64))
        np.testing.assert_equal(masked_int_data.mask, np.array([False, False, True, False]))

    def test_read_variable_replaces_fill_values_with_nan(self):
        path = str(self.temp_directory / "cdf.cdf")
        with pycdf.CDF(path, create=True) as actual_cdf:
            actual_cdf['var'] = np.array([1, 2, -1e31, 4, 5], dtype=np.float64)
            actual_cdf['var'].attrs['FILLVAL'] = -1e31
            actual_cdf['int_var'] = np.array([1, 2, 3, 4, 5], dtype=np.int32)
            actual_cdf['int_var'].attrs['FILLVAL'] = 5

            data = read_numeric_variable(actual_cdf['var'])
            int_data = read_numeric_variable(actual_cdf['int_var'])

        np.testing.assert_equal(data, np.array([1, 2, np.nan, 4, 5]))
        np.testing.assert_equal(int_data, np.array([1, 2, 3, 4, np.nan]))

    def test_write_map_cdf(self):
        class TestMapDataProduct(DataProduct):
            def __init__(self):
                self.input_metadata = InputMetadata("hi", "l3", datetime(year=2025, month=1, day=1),
                                                    datetime(year=2025, month=5, day=31), "v001",
                                                    "h45-ena-h-sf-sp-ram-hae-6deg-6mo")
                self.parent_file_names = []

            def to_data_product_variables(self) -> list[DataProductVariable]:
                return [
                    DataProductVariable("ena_intensity", np.arange(0, 10)),
                ]
        path = str(self.temp_directory / "write_cdf.cdf")
        data = TestMapDataProduct()
        ena_intensity, = data.to_data_product_variables()
        attribute_manager = Mock(spec=ImapAttributeManager)
        attribute_manager.get_global_attributes.return_value = {"global1": "global_val1", "global2": "global_val2"}
        attribute_manager.get_variable_attributes.side_effect = [
            {"CATDESC": "This should be ignored", "DATA_TYPE": "CDF_REAL4", "FILLVAL": -1e31, "RECORD_VARYING": "rV"},
        ]

        write_cdf(path, data, attribute_manager)

        expected_data_product_logical_source = "imap_hi_l3_h45-ena-h-sf-sp-ram-hae-6deg-6mo"
        attribute_manager.get_global_attributes.assert_called_once_with(expected_data_product_logical_source)
        attribute_manager.get_variable_attributes.assert_has_calls(
            [call(var.name) for var in data.to_data_product_variables()]
        )

        with pycdf.CDF(path) as actual_cdf:
            self.assertEqual("IMAP Hi45 Inten H, HAE SC Frame, Surv Corr, Ram, 6 deg, 6 Mon",
                             actual_cdf['ena_intensity'].attrs['CATDESC'])


class TestDataProduct(DataProduct):
    def __init__(self):
        self.input_metadata = InputMetadata("glows", "l3", datetime(year=2025, month=5, day=10),
                                            datetime(year=2025, month=5, day=12), "v003", "descriptor")
        self.parent_file_names = []

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable("var_without_explicit_type", np.arange(0, 10)),
            DataProductVariable("int_var", np.arange(0, 10)),
            DataProductVariable("float_var", np.arange(0, 10)),
            DataProductVariable("time_var", np.arange(10, 20)),
            DataProductVariable("non_record_varying", 100)
        ]
