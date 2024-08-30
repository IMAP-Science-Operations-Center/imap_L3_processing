from pathlib import Path

from sammi.cdf_attribute_manager import CdfAttributeManager


class ImapAttributeManager(CdfAttributeManager):
    def __init__(self):
        super().__init__(Path(f'{Path(__file__).parent.resolve()}/config'))
        self.load_global_attributes('imap_default_global_cdf_attrs.yaml')

    def add_instrument_attrs(self, instrument: str, level: str):
        self.load_global_attributes(f"imap_{instrument}_{level}_global_cdf_attrs.yaml")
        self.load_global_attributes(f"imap_{instrument}_global_cdf_attrs.yaml")
        self.load_variable_attributes(f"imap_{instrument}_{level}_variable_attrs.yaml")
