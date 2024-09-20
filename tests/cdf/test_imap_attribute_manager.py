from pathlib import Path
from unittest import TestCase

from sammi.cdf_attribute_manager import CdfAttributeManager

from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
import imap_processing


class TestImapCdfManager(TestCase):
    def test_constructor(self):
        manager = ImapAttributeManager()
        self.assertIsInstance(manager, CdfAttributeManager)

        base_manager = CdfAttributeManager(Path(f'{Path(imap_processing.__file__).parent.resolve()}/cdf/config'))

        base_manager.load_global_attributes('imap_default_global_cdf_attrs.yaml')
        self.assertEqual(base_manager.get_global_attributes(), manager.get_global_attributes())

    def test_load_instrument_and_variable_attributes_with_level(self):
        manager = ImapAttributeManager()
        manager.add_instrument_attrs('swapi', 'l3a')

        base_manager = CdfAttributeManager(Path(f'{Path(imap_processing.__file__).parent.resolve()}/cdf/config'))
        base_manager.load_global_attributes('imap_default_global_cdf_attrs.yaml')
        base_manager.load_global_attributes('imap_swapi_global_cdf_attrs.yaml')
        base_manager.load_global_attributes('imap_swapi_l3a_global_cdf_attrs.yaml')
        base_manager.load_variable_attributes('imap_swapi_l3a_variable_attrs.yaml')
        self.assertEqual(base_manager.get_global_attributes(), manager.get_global_attributes())
        self.assertEqual(base_manager.get_variable_attributes('epoch'), manager.get_variable_attributes('epoch'))
        self.assertEqual(base_manager.get_variable_attributes('proton_sw_speed_delta'),
                         manager.get_variable_attributes('proton_sw_speed_delta'))
        self.assertEqual(base_manager.get_variable_attributes('proton_sw_speed'),
                         manager.get_variable_attributes('proton_sw_speed'))
        self.assertEqual(base_manager.get_variable_attributes('proton_sw_clock_angle'),
                         manager.get_variable_attributes('proton_sw_clock_angle'))
        self.assertEqual(base_manager.get_variable_attributes('proton_sw_clock_angle_delta'),
                         manager.get_variable_attributes('proton_sw_clock_angle_delta'))
        self.assertEqual(base_manager.get_variable_attributes('proton_sw_deflection_angle'),
                         manager.get_variable_attributes('proton_sw_deflection_angle'))
        self.assertEqual(base_manager.get_variable_attributes('proton_sw_deflection_angle_delta'),
                         manager.get_variable_attributes('proton_sw_deflection_angle_delta'))
