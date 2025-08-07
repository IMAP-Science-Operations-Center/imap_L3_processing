from pathlib import Path
from typing import Optional

from sammi.cdf_attribute_manager import CdfAttributeManager


class ImapAttributeManager(CdfAttributeManager):
    def __init__(self):
        super().__init__(variable_schema_layers=[
            Path(f'{Path(__file__).parent.resolve()}/config/imap_l3_variable_cdf_attrs_schema.yaml'),
            Path(f'{Path(__file__).parent.resolve()}/config/default_variable_cdf_attrs_schema.yaml')
        ],
            global_schema_layers=[Path(__file__).parent / "config/default_global_cdf_attrs_schema.yaml"],
            use_defaults=True)
        self.config_folder_path = Path(f'{Path(__file__).parent.resolve()}/config')

        self.load_global_attributes(self.config_folder_path / 'imap_default_global_cdf_attrs.yaml')

    def try_load_global_metadata(self, logical_source: str) -> Optional[dict]:
        try:
            return self.get_global_attributes(logical_source)
        except KeyError:
            return None

    def add_instrument_attrs(self, instrument: str, level: str, descriptor: str):
        self._load_variable_attributes_if_file_exists(
            self.config_folder_path / f"imap_{instrument}_{level}_variable_attrs.yaml")
        self._load_variable_attributes_if_file_exists(
            self.config_folder_path / f"imap_{instrument}_{level}_{descriptor}_variable_attrs.yaml")

        self._load_global_attributes_if_file_exists(
            self.config_folder_path / f"imap_{instrument}_global_cdf_attrs.yaml")
        self._load_global_attributes_if_file_exists(
            self.config_folder_path / f"imap_{instrument}_{level}_global_cdf_attrs.yaml")
        self._load_global_attributes_if_file_exists(
            self.config_folder_path / f"imap_{instrument}_{level}_{descriptor}_global_cdf_attrs.yaml")

    def _load_variable_attributes_if_file_exists(self, path: Path):
        if path.exists():
            self.load_variable_attributes(path)

    def _load_global_attributes_if_file_exists(self, path: Path):
        if path.exists():
            self.load_global_attributes(path)
