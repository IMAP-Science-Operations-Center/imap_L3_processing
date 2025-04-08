import json
import unittest
from datetime import datetime
from unittest.mock import sentinel

import numpy as np
from spacepy import pycdf

from imap_l3_processing.constants import CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.l3bc_toolkit.l3b_CarringtonIonRate import CarringtonIonizationRate
from imap_l3_processing.glows.l3bc.l3bc_toolkit.l3c_CarringtonSolarWind import CarringtonSolarWind
from imap_l3_processing.glows.l3bc.models import GlowsL3BIonizationRate, GlowsL3CSolarWind
from tests.swapi.cdf_model_test_case import CdfModelTestCase
from tests.test_helpers import get_test_instrument_team_data_path


class TestModels(CdfModelTestCase):
    def test_l3b_to_data_product_variables(self):
        data = GlowsL3BIonizationRate(input_metadata=sentinel.input_metadata,
                                      epoch=sentinel.epoch,
                                      epoch_delta=sentinel.epoch_delta,
                                      cr=sentinel.cr,
                                      uv_anisotropy_factor=sentinel.uv_anisotropy_factor,
                                      lat_grid=sentinel.lat_grid,
                                      lat_grid_delta=sentinel.lat_grid_delta,
                                      sum_rate=sentinel.sum_rate,
                                      ph_rate=sentinel.ph_rate,
                                      cx_rate=sentinel.cx_rate,
                                      sum_uncert=sentinel.sum_uncert,
                                      ph_uncert=sentinel.ph_uncert,
                                      cx_uncert=sentinel.cx_uncert,
                                      lat_grid_label=sentinel.lat_grid_label,
                                      )

        variables = data.to_data_product_variables()
        self.assertEqual(13, len(variables))

        variables = iter(variables)
        self.assert_variable_attributes(next(variables), sentinel.epoch, "epoch")
        self.assert_variable_attributes(next(variables), sentinel.epoch_delta, "epoch_delta")
        self.assert_variable_attributes(next(variables), sentinel.cr, "cr")
        self.assert_variable_attributes(next(variables), sentinel.uv_anisotropy_factor, "uv_anisotropy_factor")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid, "lat_grid")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_delta, "lat_grid_delta")
        self.assert_variable_attributes(next(variables), sentinel.sum_rate, "sum_rate")
        self.assert_variable_attributes(next(variables), sentinel.ph_rate, "ph_rate")
        self.assert_variable_attributes(next(variables), sentinel.cx_rate, "cx_rate")
        self.assert_variable_attributes(next(variables), sentinel.sum_uncert, "sum_uncert")
        self.assert_variable_attributes(next(variables), sentinel.ph_uncert, "ph_uncert")
        self.assert_variable_attributes(next(variables), sentinel.cx_uncert, "cx_uncert")
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_label, "lat_grid_label")

    def test_l3b_from_instrument_team_dictionary(self):
        with open(get_test_instrument_team_data_path("glows/imap_glows_l3b_cr_2091_v00.json")) as f:
            instrument_team_l3b_dict = json.load(f)

        result = GlowsL3BIonizationRate.from_instrument_team_dictionary(instrument_team_l3b_dict,
                                                                        sentinel.input_metadata)

        latitude_grid = np.array(instrument_team_l3b_dict["ion_rate_profile"]["lat_grid"])
        self.assertIsInstance(result, GlowsL3BIonizationRate)

        self.assertEqual(sentinel.input_metadata, result.input_metadata)

        self.assertEqual([datetime(2009, 12, 20, 20, 14, 51, 359974)], result.epoch)
        np.testing.assert_equal([CARRINGTON_ROTATION_IN_NANOSECONDS / 2], result.epoch_delta)

        self.assertEqual([2091], result.cr)
        np.testing.assert_equal([instrument_team_l3b_dict["uv_anisotropy_factor"]], result.uv_anisotropy_factor)
        np.testing.assert_equal(latitude_grid, result.lat_grid)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["sum_rate"]], result.sum_rate)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["ph_rate"]], result.ph_rate)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["cx_rate"]], result.cx_rate)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["sum_uncert"]], result.sum_uncert)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["ph_uncert"]], result.ph_uncert)
        np.testing.assert_equal([instrument_team_l3b_dict["ion_rate_profile"]["cx_uncert"]], result.cx_uncert)
        np.testing.assert_equal(
            ['-90°', '-80°', '-70°', '-60°', '-50°', '-40°', '-30°', '-20°', '-10°', '0°', '10°', '20°', '30°', '40°',
             '50°', '60°', '70°', '80°', '90°'], result.lat_grid_label)
        self.assertEqual(latitude_grid.shape, result.lat_grid_delta.shape)
        self.assertEqual([
            "imap_glows_WawHelioIonMP_v002.json",
            "imap_glows_bad-days-list_v001.dat",
            "imap_glows_pipeline-settings-L3bc_v001.json",
            "imap_glows_uv-anisotropy-1CR_v001.json",
            "f107_fluxtable.txt",
            "imap_glows_l3a_20100101000000_orbX_modX_p_v00.json",
            "imap_glows_l3a_20100102000000_orbX_modX_p_v00.json",
            "imap_glows_l3a_20100103000000_orbX_modX_p_v00.json",
        ], result.parent_file_names)

    def test_l3c_to_data_product_variables(self):
        data = GlowsL3CSolarWind(input_metadata=sentinel.input_metadata,
                                 epoch=sentinel.epoch,
                                 epoch_delta=sentinel.epoch_delta,
                                 cr=sentinel.cr,
                                 lat_grid=sentinel.lat_grid,
                                 lat_grid_delta=sentinel.lat_grid_delta,
                                 lat_grid_label=sentinel.lat_grid_label,
                                 plasma_speed_ecliptic=sentinel.plasma_speed_ecliptic,
                                 proton_density_ecliptic=sentinel.proton_density_ecliptic,
                                 alpha_abundance_ecliptic=sentinel.alpha_abundance_ecliptic,
                                 plasma_speed_profile=sentinel.plasma_speed_profile,
                                 proton_density_profile=sentinel.proton_density_profile,
                                 )

        variables = data.to_data_product_variables()
        self.assertEqual(11, len(variables))

        variables = iter(variables)
        self.assert_variable_attributes(next(variables), sentinel.epoch, "epoch",
                                        expected_data_type=pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(next(variables), sentinel.epoch_delta, "epoch_delta",
                                        expected_data_type=pycdf.const.CDF_INT8)
        self.assert_variable_attributes(next(variables), sentinel.cr, "cr", expected_data_type=pycdf.const.CDF_INT2)
        self.assert_variable_attributes(next(variables), sentinel.lat_grid, "lat_grid",
                                        expected_data_type=pycdf.const.CDF_FLOAT, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_delta, "lat_grid_delta",
                                        expected_data_type=pycdf.const.CDF_FLOAT, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), sentinel.lat_grid_label, "lat_grid_label",
                                        expected_data_type=pycdf.const.CDF_CHAR, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), sentinel.plasma_speed_ecliptic, "plasma_speed_ecliptic",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.proton_density_ecliptic, "proton_density_ecliptic",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.alpha_abundance_ecliptic, "alpha_abundance_ecliptic",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.plasma_speed_profile, "plasma_speed_profile",
                                        expected_data_type=pycdf.const.CDF_FLOAT)
        self.assert_variable_attributes(next(variables), sentinel.proton_density_profile, "proton_density_profile",
                                        expected_data_type=pycdf.const.CDF_FLOAT)

    def test_l3c_from_instrument_team_dictionary(self):
        with open(get_test_instrument_team_data_path("glows/imap_glows_l3c_cr_2091_v00.json")) as f:
            model = json.load(f)
        latitude_grid = model["solar_wind_profile"]["lat_grid"]
        result = GlowsL3CSolarWind.from_instrument_team_dictionary(
            model, sentinel.input_metadata)

        self.assertIsInstance(result, GlowsL3CSolarWind)

        self.assertEqual(sentinel.input_metadata, result.input_metadata)
        self.assertEqual([datetime(2009, 12, 20, 20, 14, 51, 359974)], result.epoch)
        np.testing.assert_equal([CARRINGTON_ROTATION_IN_NANOSECONDS / 2], result.epoch_delta)

        self.assertEqual([2091], result.cr)
        np.testing.assert_equal(latitude_grid, result.lat_grid)
        self.assertEqual(
            ['-90°', '-80°', '-70°', '-60°', '-50°', '-40°', '-30°', '-20°', '-10°', '0°', '10°', '20°', '30°', '40°',
             '50°', '60°', '70°', '80°', '90°'], result.lat_grid_label)
        self.assertEqual((19,), result.lat_grid_delta.shape)
        np.testing.assert_equal([model["solar_wind_profile"]["plasma_speed"]], result.plasma_speed_profile)
        np.testing.assert_equal([model["solar_wind_profile"]["proton_density"]], result.proton_density_profile)

        self.assertEqual([model["solar_wind_ecliptic"]["plasma_speed"]], result.plasma_speed_ecliptic)
        self.assertEqual([model["solar_wind_ecliptic"]["proton_density"]], result.proton_density_ecliptic)
        self.assertEqual([model["solar_wind_ecliptic"]["alpha_abundance"]], result.alpha_abundance_ecliptic)

        self.assertEqual([
            "imap_glows_WawHelioIonMP_v002.json",
            "imap_glows_bad-days-list_v001.dat",
            "imap_glows_pipeline-settings-L3bc_v001.json",
            "imap_glows_uv-anisotropy-1CR_v001.json",
            "omni_2010.dat",
            "imap_glows_l3b_cr_2091_v00.json",
        ], result.parent_file_names)


if __name__ == '__main__':
    unittest.main()
