import unittest
from datetime import datetime
from unittest.mock import sentinel, Mock, MagicMock

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import get_test_instrument_team_data_path


class TestL3eUltraModel(unittest.TestCase):
    def test_l3e_ultra_model_to_data_product_variables(self):
        l3e_ultra: GlowsL3EUltraData = GlowsL3EUltraData(
            Mock(),
            sentinel.epoch,
            sentinel.energy,
            sentinel.healpix_index,
            sentinel.probability_of_survival,
            sentinel.spin_axis_latitude,
            sentinel.spin_axis_longitude,
            sentinel.program_version,
            sentinel.spacecraft_radius,
            sentinel.spacecraft_latitude,
            sentinel.spacecraft_longitude,
            sentinel.spacecraft_velocity_x,
            sentinel.spacecraft_velocity_y,
            sentinel.spacecraft_velocity_z,
        )

        expected_energy_labels = ['Energy Label 1', 'Energy Label 2', 'Energy Label 3', 'Energy Label 4',
                                  'Energy Label 5', 'Energy Label 6', 'Energy Label 7', 'Energy Label 8',
                                  'Energy Label 9', 'Energy Label 10', 'Energy Label 11', 'Energy Label 12',
                                  'Energy Label 13', 'Energy Label 14', 'Energy Label 15', 'Energy Label 16',
                                  'Energy Label 17', 'Energy Label 18', 'Energy Label 19', 'Energy Label 20',
                                  ]

        expected_healpix_labels = [f'Heal Pixel Label {i}' for i in range(0, 3072)]

        data_products = l3e_ultra.to_data_product_variables()
        expected_data_products = [
            DataProductVariable("epoch", sentinel.epoch),
            DataProductVariable("energy_grid", sentinel.energy),
            DataProductVariable("healpix_index", sentinel.healpix_index),
            DataProductVariable("surv_prob", sentinel.probability_of_survival),
            DataProductVariable("energy_label", expected_energy_labels),
            DataProductVariable("healpix_index_label", expected_healpix_labels),
            DataProductVariable("spin_axis_latitude", np.array([sentinel.spin_axis_latitude])),
            DataProductVariable("spin_axis_longitude", np.array([sentinel.spin_axis_longitude])),
            DataProductVariable("program_version", np.array([sentinel.program_version])),
            DataProductVariable("spacecraft_radius", np.array([sentinel.spacecraft_radius])),
            DataProductVariable("spacecraft_latitude", np.array([sentinel.spacecraft_latitude])),
            DataProductVariable("spacecraft_longitude", np.array([sentinel.spacecraft_longitude])),
            DataProductVariable("spacecraft_velocity_x", np.array([sentinel.spacecraft_velocity_x])),
            DataProductVariable("spacecraft_velocity_y", np.array([sentinel.spacecraft_velocity_y])),
            DataProductVariable("spacecraft_velocity_z", np.array([sentinel.spacecraft_velocity_z])),
        ]

        self.assertEqual(expected_data_products, data_products)

    def test_convert_dat_to_glows_l3e_ul_product(self):
        ul_file_path = get_test_instrument_team_data_path("glows/probSur.Imap.Ul_20250420_000000_2025.300.txt")
        expected_epoch = datetime(year=2009, month=1, day=1)

        expected_energy = np.array(
            [2.3751086, 3.0917682, 4.0246710, 5.2390656, 6.8198887, 8.8777057, 11.5564435, 15.0434572, 19.5826341,
             25.4914514, 33.1831812, 43.1957952, 56.2295915, 73.1961745, 95.2822137, 124.0324417, 161.4576950,
             210.1755551, 273.5934261, 356.1468544])

        row_1_probability_of_survival = np.array(
            [0.84827693E+00, 0.86517360E+00, 0.88086645E+00, 0.89551288E+00, 0.90925039E+00, 0.92211605E+00,
             0.93428862E+00, 0.94576000E+00, 0.95660359E+00, 0.96660933E+00, 0.97533752E+00, 0.98226689E+00,
             0.98714107E+00, 0.99025210E+00, 0.99218849E+00, 0.99346478E+00, 0.99439230E+00, 0.99512632E+00,
             0.99574163E+00, 0.99627086E+00])

        row_804_probability_of_survival = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    np.nan, np.nan])

        row_3071_expected_probability_of_survival = np.array(
            [0.84498413E+00, 0.86231633E+00, 0.87840630E+00, 0.89344499E+00, 0.90753390E+00, 0.92079246E+00,
             0.93326641E+00, 0.94509283E+00, 0.95623858E+00, 0.96649540E+00, 0.97543873E+00, 0.98248159E+00,
             0.98738469E+00, 0.99048237E+00, 0.99239223E+00, 0.99364490E+00, 0.99455000E+00, 0.99526291E+00,
             0.99585178E+00, 0.99635733E+00])

        expected_survival_probability_shape = (1, 20, 3072)

        expected_heal_pix = np.arange(0, 3072)
        mock_metadata = Mock()

        spin_axis_lat = 45.0
        spin_axis_lon = 90.0

        args = MagicMock(spec=GlowsL3eCallArguments)
        args.spin_axis_latitude = spin_axis_lat
        args.spin_axis_longitude = spin_axis_lon

        expected_program_version = 'Ultra.v00.01'

        args.spacecraft_radius = .5
        args.spacecraft_longitude = 85.4
        args.spacecraft_latitude = 45.1

        args.spacecraft_velocity_x = 2.1
        args.spacecraft_velocity_y = 2.2
        args.spacecraft_velocity_z = 2.3

        l3e_ul_product: GlowsL3EUltraData = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(mock_metadata,
                                                                                                  ul_file_path,
                                                                                                  expected_epoch,
                                                                                                  args)

        self.assertEqual([expected_epoch], l3e_ul_product.epoch)

        np.testing.assert_array_equal(l3e_ul_product.energy, expected_energy)

        np.testing.assert_array_equal(expected_heal_pix, l3e_ul_product.healpix_index)

        self.assertEqual(expected_survival_probability_shape, l3e_ul_product.probability_of_survival.shape)

        np.testing.assert_array_equal(l3e_ul_product.probability_of_survival[0].T[0, :], row_1_probability_of_survival)

        np.testing.assert_array_equal(l3e_ul_product.probability_of_survival[0].T[804, :],
                                      row_804_probability_of_survival)

        np.testing.assert_array_equal(l3e_ul_product.probability_of_survival[0].T[3071, :],
                                      row_3071_expected_probability_of_survival)

        self.assertEqual(np.array([spin_axis_lat]), l3e_ul_product.spin_axis_lat)
        self.assertEqual(np.array([spin_axis_lon]), l3e_ul_product.spin_axis_lon)

        self.assertEqual(np.array([expected_program_version]), l3e_ul_product.program_version)

        self.assertEqual(np.array([.5]), l3e_ul_product.spacecraft_radius)
        self.assertEqual(np.array([85.4]), l3e_ul_product.spacecraft_longitude)
        self.assertEqual(np.array([45.1]), l3e_ul_product.spacecraft_latitude)

        self.assertEqual(np.array([2.1]), l3e_ul_product.spacecraft_velocity_x)
        self.assertEqual(np.array([2.2]), l3e_ul_product.spacecraft_velocity_y)
        self.assertEqual(np.array([2.3]), l3e_ul_product.spacecraft_velocity_z)
