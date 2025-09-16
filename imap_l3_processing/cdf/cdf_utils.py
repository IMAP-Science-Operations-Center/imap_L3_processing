import re

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.swapi.l3a.models import DataProduct


def write_cdf(file_path: str, data: DataProduct, attribute_manager: ImapAttributeManager):
    with CDF(file_path, '') as cdf:
        cdf.col_major(True)
        try:
            global_attrs = attribute_manager.get_global_attributes(data.input_metadata.logical_source)
        except KeyError:
            trimmed_source = re.sub(r"-\d+$", "-", data.input_metadata.logical_source)
            global_attrs = attribute_manager.get_global_attributes(trimmed_source)

        for k, v in global_attrs.items():
            cdf.attrs[k] = v

        for data_product in data.to_data_product_variables():
            var_name = data_product.name
            variable_attributes = attribute_manager.get_variable_attributes(var_name)
            data_type = getattr(pycdf.const, variable_attributes["DATA_TYPE"])
            data_array = np.asanyarray(data_product.value)

            record_varying = variable_attributes["RECORD_VARYING"].lower() == "rv"
            if record_varying:
                dims = data_array.shape[1:]
            else:
                dims = data_array.shape

            if data_array.size == 0:
                data_array = None
            else:
                if 'FILLVAL' in variable_attributes:
                    if np.issubdtype(data_array.dtype, np.floating):
                        data_array = np.ma.masked_invalid(data_array)
                    data_array = np.ma.filled(data_array, variable_attributes['FILLVAL'])
                else:
                    assert not np.ma.isMaskedArray(data_array)

            cdf.new(var_name, data_array,
                    recVary=record_varying,
                    type=data_type,
                    dims=dims)
            for k, v in variable_attributes.items():
                if k == 'DEPEND_0' and v == '':
                    continue
                if k in ['DATA_TYPE', 'RECORD_VARYING']:
                    continue
                if k == 'FILLVAL' and data_type is not None:
                    cdf[var_name].attrs.new(k, v, data_type)
                else:
                    cdf[var_name].attrs[k] = v


def read_variable_and_mask_fill_values(var: pycdf.Var) -> np.ma.masked_array:
    return np.ma.masked_equal(var[...], var.attrs['FILLVAL'])


def read_numeric_variable(var: pycdf.Var) -> np.ndarray:
    assert np.issubdtype(var.dtype, np.number)
    return np.where(var[...] == var.attrs['FILLVAL'], np.nan, var[...])
