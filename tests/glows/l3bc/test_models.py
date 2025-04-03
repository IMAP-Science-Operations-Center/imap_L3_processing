import unittest
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

    def test_l3b_from_instrument_team_model(self):
        pipeline_settings_path = get_test_instrument_team_data_path("glows/imap_glows_pipeline-settings-L3bc_v001.json")
        model = CarringtonIonizationRate(l3a_fn_list=[],
                                         anc_input_from_instr_team={
                                             "pipeline_settings": pipeline_settings_path,
                                         },
                                         ext_dependencies={
                                             "f107_raw_data": []
                                         })

        model.carr_ion_rate["date"] = sentinel.date
        model.carr_ion_rate["CR"] = sentinel.CR
        latitude_grid = np.array([-90, -45, 0, 45, 90])
        model.carr_ion_rate["ion_grid"] = latitude_grid
        model.carr_ion_rate["ion_rate"] = sentinel.ion_rate
        model.carr_ion_rate["ph_rate"] = sentinel.ph_rate
        model.carr_ion_rate["cx_rate"] = sentinel.cx_rate
        model.carr_ion_rate["ion_rate_uncert"] = sentinel.ion_rate_uncert
        model.carr_ion_rate["ph_rate_uncert"] = sentinel.ph_rate_uncert
        model.carr_ion_rate["cx_rate_uncert"] = sentinel.cx_rate_uncert

        model.uv_anisotropy = sentinel.uv_anisotropy

        result = GlowsL3BIonizationRate.from_instrument_team_object(model, sentinel.input_metadata)

        self.assertIsInstance(result, GlowsL3BIonizationRate)

        self.assertEqual(sentinel.input_metadata, result.input_metadata)
        self.assertEqual([sentinel.date], result.epoch)
        np.testing.assert_equal([CARRINGTON_ROTATION_IN_NANOSECONDS / 2], result.epoch_delta)

        self.assertEqual([sentinel.CR], result.cr)
        self.assertEqual([sentinel.uv_anisotropy], result.uv_anisotropy_factor)
        np.testing.assert_equal(latitude_grid, result.lat_grid)
        self.assertEqual([sentinel.ion_rate], result.sum_rate)
        self.assertEqual([sentinel.ph_rate], result.ph_rate)
        self.assertEqual([sentinel.cx_rate], result.cx_rate)
        self.assertEqual([sentinel.ion_rate_uncert], result.sum_uncert)
        self.assertEqual([sentinel.ph_rate_uncert], result.ph_uncert)
        self.assertEqual([sentinel.cx_rate_uncert], result.cx_uncert)
        self.assertEqual(["-90°", "-45°", "0°", "45°", "90°"], result.lat_grid_label)
        self.assertEqual((5,), result.lat_grid_delta.shape)

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

    def test_l3c_from_instrument_team_model(self):
        pipeline_settings_path = get_test_instrument_team_data_path("glows/imap_glows_pipeline-settings-L3bc_v001.json")
        model = CarringtonSolarWind(anc_input_from_instr_team={
            "pipeline_settings": pipeline_settings_path,
        })

        model.sw_profile["date"] = sentinel.date
        model.sw_profile["CR"] = sentinel.CR
        latitude_grid = np.array([-90, -45, 0, 45, 90])
        model.sw_profile["grid"] = latitude_grid
        model.sw_profile["plasma_speed"] = sentinel.plasma_speed
        model.sw_profile["proton_density"] = sentinel.proton_density

        model.sw_ecliptic["mean_speed"] = sentinel.mean_speed
        model.sw_ecliptic["mean_proton_density"] = sentinel.mean_proton_density
        model.sw_ecliptic["mean_alpha_abundance"] = sentinel.mean_alpha_abundance

        result = GlowsL3CSolarWind.from_instrument_team_object(model, sentinel.input_metadata)

        self.assertIsInstance(result, GlowsL3CSolarWind)

        self.assertEqual(sentinel.input_metadata, result.input_metadata)
        self.assertEqual([sentinel.date], result.epoch)
        np.testing.assert_equal([CARRINGTON_ROTATION_IN_NANOSECONDS / 2], result.epoch_delta)

        self.assertEqual([sentinel.CR], result.cr)
        np.testing.assert_equal(latitude_grid, result.lat_grid)
        self.assertEqual(["-90°", "-45°", "0°", "45°", "90°"], result.lat_grid_label)
        self.assertEqual((5,), result.lat_grid_delta.shape)
        self.assertEqual([sentinel.plasma_speed], result.plasma_speed_profile)
        self.assertEqual([sentinel.proton_density], result.proton_density_profile)

        self.assertEqual([sentinel.mean_speed], result.plasma_speed_ecliptic)
        self.assertEqual([sentinel.mean_proton_density], result.proton_density_ecliptic)
        self.assertEqual([sentinel.mean_alpha_abundance], result.alpha_abundance_ecliptic)


if __name__ == '__main__':
    unittest.main()
