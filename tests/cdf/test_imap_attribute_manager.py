from pathlib import Path
from unittest import TestCase

from sammi.cdf_attribute_manager import CdfAttributeManager

import imap_l3_processing
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager


class TestImapCdfManager(TestCase):
    def setUp(self):
        self.config_folder_path = Path(imap_l3_processing.__file__).parent.resolve() / 'cdf/config'
        self.base_manager = CdfAttributeManager(
            variable_schema_layers=[self.config_folder_path / 'imap_l3_variable_cdf_attrs_schema.yaml'],
            use_defaults=True)
        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_default_global_cdf_attrs.yaml')

    def test_constructor(self):
        manager = ImapAttributeManager()
        self.assertIsInstance(manager, CdfAttributeManager)

        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())

    def test_load_instrument_and_variable_attributes_with_level(self):
        manager = ImapAttributeManager()
        manager.add_instrument_attrs('swapi', 'l3a', "descriptor")

        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_swapi_global_cdf_attrs.yaml')
        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_swapi_l3a_global_cdf_attrs.yaml')
        self.base_manager.load_variable_attributes(self.config_folder_path / 'imap_swapi_l3a_variable_attrs.yaml')

        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())
        self.assertEqual(self.base_manager.get_variable_attributes('epoch'), manager.get_variable_attributes('epoch'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_speed_delta'),
                         manager.get_variable_attributes('proton_sw_speed_delta'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_speed'),
                         manager.get_variable_attributes('proton_sw_speed'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_clock_angle'),
                         manager.get_variable_attributes('proton_sw_clock_angle'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_clock_angle_delta'),
                         manager.get_variable_attributes('proton_sw_clock_angle_delta'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_deflection_angle'),
                         manager.get_variable_attributes('proton_sw_deflection_angle'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_deflection_angle_delta'),
                         manager.get_variable_attributes('proton_sw_deflection_angle_delta'))

        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_speed'),
                         manager.get_variable_attributes('alpha_sw_speed'))
        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_speed_delta'),
                         manager.get_variable_attributes('alpha_sw_speed_delta'))

        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_density'),
                         manager.get_variable_attributes('alpha_sw_density'))
        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_density_delta'),
                         manager.get_variable_attributes('alpha_sw_density_delta'))

        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_temperature'),
                         manager.get_variable_attributes('alpha_sw_temperature'))
        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_temperature_delta'),
                         manager.get_variable_attributes('alpha_sw_temperature_delta'))

    def test_load_instrument_and_variable_attributes_with_level_and_descriptor(self):
        manager = ImapAttributeManager()
        manager.add_instrument_attrs('hit', 'l3', 'macropixel')

        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_hit_global_cdf_attrs.yaml')
        self.base_manager.load_global_attributes(
            self.config_folder_path / 'imap_hit_l3_macropixel_global_cdf_attrs.yaml')
        self.base_manager.load_variable_attributes(
            self.config_folder_path / 'imap_hit_l3_macropixel_variable_attrs.yaml')

        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())
        self.assertEqual(self.base_manager.get_variable_attributes('epoch'), manager.get_variable_attributes('epoch'))

    def test_l3b_metadata_configuration(self):
        manager = ImapAttributeManager()
        manager.add_instrument_attrs('swapi', 'l3b', "descriptor")

        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_swapi_global_cdf_attrs.yaml')
        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_swapi_l3b_global_cdf_attrs.yaml')
        self.base_manager.load_variable_attributes(self.config_folder_path / 'imap_swapi_l3b_variable_attrs.yaml')
        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())
        self.assertEqual(self.base_manager.get_variable_attributes('epoch'), manager.get_variable_attributes('epoch'))

        self.assertEqual(self.base_manager.get_variable_attributes('combined_energy'),
                         manager.get_variable_attributes('combined_energy'))
        self.assertEqual(self.base_manager.get_variable_attributes('combined_energy_delta_minus'),
                         manager.get_variable_attributes('combined_energy_delta_minus'))
        self.assertEqual(self.base_manager.get_variable_attributes('combined_energy_delta_plus'),
                         manager.get_variable_attributes('combined_energy_delta_plus'))

        self.assertEqual(self.base_manager.get_variable_attributes('combined_differential_flux'),
                         manager.get_variable_attributes('combined_differential_flux'))
        self.assertEqual(self.base_manager.get_variable_attributes('combined_differential_flux_delta'),
                         manager.get_variable_attributes('combined_differential_flux_delta'))
