import unittest

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments


class TestGlowsL3eCallArguments(unittest.TestCase):

    def test_to_argument_list(self):
        glows_l3e_call_args = GlowsL3eCallArguments(
            formatted_date="20100101",
            decimal_date="2010.001",
            spacecraft_radius=.54,
            spacecraft_longitude=190,
            spacecraft_latitude=-45,
            spacecraft_velocity_x=45,
            spacecraft_velocity_y=34,
            spacecraft_velocity_z=12,
            spin_axis_longitude=34,
            spin_axis_latitude=15.12345,
            elongation=20.10211,

        )

        expected_call_args = ["20100101", "2010.001", "0.54", "190", "-45", "45", "34", "12", "34", "15.1235", "20.102"]

        self.assertEqual(expected_call_args, glows_l3e_call_args.to_argument_list())
