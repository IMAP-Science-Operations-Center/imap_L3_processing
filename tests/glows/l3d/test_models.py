import unittest
from unittest.mock import Mock, sentinel

from imap_l3_processing.glows.l3d.models import GlowsL3DSolarParamsHistory, LAT_GRID_CDF_VAR_NAME, CR_GRID_CDF_VAR_NAME, \
    TIME_GRID_CDF_VAR_NAME, SPEED_CDF_VAR_NAME, P_DENS_CDF_VAR_NAME, UV_ANIS_CDF_VAR_NAME, PHION_CDF_VAR_NAME, \
    LYA_CDF_VAR_NAME, E_DENS_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_glows_l3d_solar_params_history_to_data_product(self):
        input_metadata = sentinel.input_metadata
        parent_file_names = sentinel.parent_file_names
        lat_grid = sentinel.lat_grid
        cr_grid = sentinel.cr_grid
        time_grid = sentinel.time_grid
        speed = sentinel.speed
        p_dens = sentinel.p_dens
        uv_anis = sentinel.uv_anis
        phion = sentinel.phion
        lya = sentinel.lya
        e_dens = sentinel.e_dens

        data_product = GlowsL3DSolarParamsHistory(
            input_metadata=input_metadata,
            parent_file_names=parent_file_names,
            lat_grid=lat_grid,
            cr_grid=cr_grid,
            time_grid=time_grid,
            speed=speed,
            p_dens=p_dens,
            uv_anis=uv_anis,
            phion=phion,
            lya=lya,
            e_dens=e_dens,
        )

        variables = data_product.to_data_product_variables()

        self.assertEqual(9, len(variables))
        variables = iter(variables)
        self.assert_variable_attributes(next(variables), lat_grid, LAT_GRID_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), cr_grid, CR_GRID_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), time_grid, TIME_GRID_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), speed, SPEED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), p_dens, P_DENS_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), uv_anis, UV_ANIS_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), phion, PHION_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), lya, LYA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), e_dens, E_DENS_CDF_VAR_NAME)
