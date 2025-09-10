from pathlib import Path
from unittest import TestCase

from sammi.cdf_attribute_manager import CdfAttributeManager

import imap_l3_processing
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager


class TestImapCdfManager(TestCase):
    def setUp(self):
        self.config_folder_path = Path(imap_l3_processing.__file__).parent.resolve() / 'cdf/config'
        self.base_manager = CdfAttributeManager(
            variable_schema_layers=[self.config_folder_path / 'imap_l3_variable_cdf_attrs_schema.yaml',
                                    self.config_folder_path / 'default_variable_cdf_attrs_schema.yaml'],
            global_schema_layers=[self.config_folder_path / 'default_global_cdf_attrs_schema.yaml'],
            use_defaults=True)
        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_default_global_cdf_attrs.yaml')

    def test_constructor(self):
        manager = ImapAttributeManager()
        self.assertIsInstance(manager, CdfAttributeManager)

        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())

    def test_load_instrument_and_variable_attributes_with_level(self):
        manager = ImapAttributeManager()
        manager.add_instrument_attrs('swapi', 'l3a', "descriptor")
        manager.add_global_attribute('ground_software_version', 'test version')

        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_swapi_global_cdf_attrs.yaml')
        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_swapi_l3a_global_cdf_attrs.yaml')
        self.base_manager.load_variable_attributes(self.config_folder_path / 'imap_swapi_l3a_variable_attrs.yaml')
        self.base_manager.add_global_attribute('ground_software_version', 'test version')

        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())
        self.assertEqual(self.base_manager.get_variable_attributes('epoch'), manager.get_variable_attributes('epoch'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_speed_uncert'),
                         manager.get_variable_attributes('proton_sw_speed_uncert'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_speed'),
                         manager.get_variable_attributes('proton_sw_speed'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_clock_angle'),
                         manager.get_variable_attributes('proton_sw_clock_angle'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_clock_angle_uncert'),
                         manager.get_variable_attributes('proton_sw_clock_angle_uncert'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_deflection_angle'),
                         manager.get_variable_attributes('proton_sw_deflection_angle'))
        self.assertEqual(self.base_manager.get_variable_attributes('proton_sw_deflection_angle_uncert'),
                         manager.get_variable_attributes('proton_sw_deflection_angle_uncert'))

        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_speed'),
                         manager.get_variable_attributes('alpha_sw_speed'))
        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_speed_uncert'),
                         manager.get_variable_attributes('alpha_sw_speed_uncert'))

        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_density'),
                         manager.get_variable_attributes('alpha_sw_density'))
        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_density_uncert'),
                         manager.get_variable_attributes('alpha_sw_density_uncert'))

        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_temperature'),
                         manager.get_variable_attributes('alpha_sw_temperature'))
        self.assertEqual(self.base_manager.get_variable_attributes('alpha_sw_temperature_uncert'),
                         manager.get_variable_attributes('alpha_sw_temperature_uncert'))

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
        self.assertEqual(self.base_manager.get_variable_attributes('combined_energy_stat_uncert_minus'),
                         manager.get_variable_attributes('combined_energy_stat_uncert_minus'))
        self.assertEqual(self.base_manager.get_variable_attributes('combined_energy_stat_uncert_plus'),
                         manager.get_variable_attributes('combined_energy_stat_uncert_plus'))

        self.assertEqual(self.base_manager.get_variable_attributes('combined_differential_flux'),
                         manager.get_variable_attributes('combined_differential_flux'))
        self.assertEqual(self.base_manager.get_variable_attributes('combined_differential_flux_stat_uncert'),
                         manager.get_variable_attributes('combined_differential_flux_stat_uncert'))

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

    def test_gets_global_attrs_from_global_file_when_descriptor_specific_variable_file_exists_and_global_does_not(self):
        manager = ImapAttributeManager()
        manager.add_instrument_attrs('hi', 'l3', '45sensor-spacecraft-survival-full-4deg-map')

        self.base_manager.load_global_attributes(self.config_folder_path / 'imap_hi_l3_global_cdf_attrs.yaml')
        self.base_manager.load_global_attributes(
            self.config_folder_path / 'imap_hi_global_cdf_attrs.yaml')

        self.assertEqual(self.base_manager.get_global_attributes(), manager.get_global_attributes())
