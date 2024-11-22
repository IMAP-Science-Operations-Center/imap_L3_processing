import re

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
            cdf.new(var_name, data=data_product.value,
                    recVary=data_product.record_varying,
                    type=data_product.cdf_data_type)
            for k, v in attribute_manager.get_variable_attributes(var_name).items():
                if var_name == 'epoch' and k == 'DEPEND_0':
                    continue
                cdf[var_name].attrs[k] = v
