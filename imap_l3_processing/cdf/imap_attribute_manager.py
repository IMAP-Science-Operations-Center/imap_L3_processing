import os.path
from pathlib import Path

from sammi.cdf_attribute_manager import CdfAttributeManager


class ImapAttributeManager(CdfAttributeManager):
    def __init__(self):
        super().__init__(variable_schema_layers=[
            Path(f'{Path(__file__).parent.resolve()}/config/imap_l3_variable_cdf_attrs_schema.yaml')],
            use_defaults=True)
        self.config_folder_path = Path(f'{Path(__file__).parent.resolve()}/config')

        self.load_global_attributes(self.config_folder_path / 'imap_default_global_cdf_attrs.yaml')

    def add_instrument_attrs(self, instrument: str, level: str, descriptor: str):
        descriptor_field = ""
        if os.path.exists(self.config_folder_path / f"imap_{instrument}_{level}_{descriptor}_variable_attrs.yaml"):
            descriptor_field = f"{descriptor}_"

        self.load_variable_attributes(
            self.config_folder_path / f"imap_{instrument}_{level}_{descriptor_field}variable_attrs.yaml")
        self.load_global_attributes(
            self.config_folder_path / f"imap_{instrument}_{level}_{descriptor_field}global_cdf_attrs.yaml")

        self.load_global_attributes(self.config_folder_path / f"imap_{instrument}_global_cdf_attrs.yaml")
