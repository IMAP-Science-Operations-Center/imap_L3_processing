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

        data_type_to_pycdf_const = {"CDF_TIME_TT2000": pycdf.const.CDF_TIME_TT2000,
                                    "CDF_FLOAT": pycdf.const.CDF_FLOAT,
                                    "CDF_INT8": pycdf.const.CDF_INT8,
                                    "CDF_UINT4": pycdf.const.CDF_UINT4,
                                    "CDF_CHAR": pycdf.const.CDF_CHAR, }
        try:
            for data_product in data.to_data_product_variables():
                var_name = data_product.name
                variable_attributes = attribute_manager.get_variable_attributes(var_name)
                data_array = np.asarray(data_product.value)

                if 'FILLVAL' in variable_attributes and np.issubdtype(data_array.dtype, np.floating):
                    data_array = np.where(np.isnan(data_array), variable_attributes["FILLVAL"], data_array)

                data_type = data_type_to_pycdf_const[variable_attributes["DATA_TYPE"]]
                record_varying = variable_attributes["RECORD_VARYING"] == "RV"
                cdf.new(var_name, data_array,
                        recVary=record_varying,
                        type=data_type)
                for k, v in variable_attributes.items():
                    if k == 'DEPEND_0' and v == '':
                        continue
                    if k == 'FILLVAL' and data_type is not None:
                        cdf[var_name].attrs.new(k, v, data_type)
                    else:
                        cdf[var_name].attrs[k] = v

                    # if k in ["LABL_PTR_1", "LABL_PTR_2", "LABL_PTR_3"]:
                    #     if v not in cdf:
                    #         required_attributes = attribute_manager.variable_attribute_schema["metadata"]
                    #         meta_data_attributes = attribute_manager.get_variable_attributes(v)
                    #         size(data_product.value)
                    #
                    #         data_type = data_type_to_pycdf_const[meta_data_attributes[
                    #             "DATA_TYPE"]] if "DATA_TYPE" in meta_data_attributes.keys() else pycdf.const.CDF_CHAR
                    #
                    #         cdf.new(v, type=data_type, dims=int(k[-1]))
                    #
                    #         for meta_data_key, meta_data_val in meta_data_attributes.items():
                    #             if meta_data_key in required_attributes:
                    #                 cdf[v].attrs.new(meta_data_key, meta_data_val, data_type)
                    # cdf[v].attrs[meta_data_key] = meta_data_val
        except Exception as e:
            debug = 0


def read_variable(var: pycdf.Var) -> np.ndarray:
    return np.where(var == var.attrs['FILLVAL'], np.nan, var)
