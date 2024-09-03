from spacepy.pycdf import CDF

from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.swapi.l3a.models import DataProduct


def write_cdf(file_path: str, data: DataProduct, attribute_manager: ImapAttributeManager):
    with CDF(file_path, '') as cdf:
        cdf.col_major(True)
        for k, v in attribute_manager.get_global_attributes().items():
            cdf.attrs[k] = v

        for data_product in data.to_data_product_variables():
            var_name = data_product.name
            if data_product.record_varying:
                cdf[var_name] = data_product.value
            else:
                cdf.new(var_name, data_product.value, recVary=False)
            if data_product.cdf_data_type:
                cdf[var_name].type(data_product.cdf_data_type)
            for k, v in attribute_manager.get_variable_attributes(var_name).items():
                cdf[var_name].attrs[k] = v
