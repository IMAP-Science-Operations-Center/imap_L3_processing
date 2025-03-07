import re

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.swapi.l3a.models import DataProduct


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
            data_array = np.asarray(data_product.value)
            if 'FILLVAL' in variable_attributes and np.issubdtype(data_array.dtype, np.floating):
                data_array = np.where(np.isnan(data_array), variable_attributes["FILLVAL"], data_array)
            cdf.new(var_name, data_array,
                    recVary=data_product.record_varying,
                    type=data_product.cdf_data_type)
            for k, v in variable_attributes.items():
                if k == 'DEPEND_0' and v == '':
                    continue
                if k == 'FILLVAL' and data_product.cdf_data_type is not None:
                    cdf[var_name].attrs.new(k, v, data_product.cdf_data_type)
                else:
                    cdf[var_name].attrs[k] = v


def read_variable(var: pycdf.Var) -> np.ndarray:
    return np.where(var == var.attrs['FILLVAL'], np.nan, var)
