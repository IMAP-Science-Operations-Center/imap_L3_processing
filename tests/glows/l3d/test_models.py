from unittest.mock import sentinel

import numpy as np

from imap_l3_processing.constants import CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.glows.l3d.models import GlowsL3DSolarParamsHistory, SPEED_CDF_VAR_NAME, PHION_CDF_VAR_NAME, \
    EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, LATITUDE_CDF_CDF_VAR_NAME, \
    LATITUDE_LABEL_CDF_VAR_NAME, CR_CDF_VAR_NAME, PROTON_DENSITY_CDF_VAR_NAME, UV_ANISOTROPY_CDF_VAR_NAME, \
    LYMAN_ALPHA_CDF_VAR_NAME, ELECTRON_DENSITY_CDF_VAR_NAME, PLASMA_SPEED_FLAG_CDF_VAR_NAME, \
    UV_ANISOTROPY_FLAG_CDF_VAR_NAME, PROTON_DENSITY_FLAG_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_glows_l3d_solar_params_history_to_data_product(self):
        epoch = np.full(10, 0)
        input_metadata = sentinel.input_metadata
        parent_file_names = sentinel.parent_file_names
        latitude = np.arange(-90, 100, step=10)
        cr = sentinel.cr
        plasma_speed = sentinel.plasma_speed
        proton_density = sentinel.proton_density
        ultraviolet_anisotropy = sentinel.ultraviolet_anisotropy
        phion = sentinel.phion
        lyman_alpha = sentinel.lyman_alpha
        electron_density = sentinel.electron_density
        plasma_speed_flag = sentinel.plasma_speed_flag
        uv_anisotropy_flag = sentinel.uv_anisotropy_flag
        proton_density_flag = sentinel.proton_density_flag

        data_product = GlowsL3DSolarParamsHistory(
            input_metadata=input_metadata,
            parent_file_names=parent_file_names,
            epoch=epoch,
            latitude=latitude,
            cr=cr,
            plasma_speed=plasma_speed,
            proton_density=proton_density,
            ultraviolet_anisotropy=ultraviolet_anisotropy,
            phion=phion,
            lyman_alpha=lyman_alpha,
            electron_density=electron_density,
            plasma_speed_flag=plasma_speed_flag,
            uv_anisotropy_flag=uv_anisotropy_flag,
            proton_density_flag=proton_density_flag
        )

        variables = data_product.to_data_product_variables()

        expected_epoch_delta = np.full_like(epoch, CARRINGTON_ROTATION_IN_NANOSECONDS / 2)
        expected_latitude_label = [f'{deg:.1f} degrees' for deg in latitude]

        self.assertEqual(14, len(variables))
        variables = iter(variables)
        self.assert_variable_attributes(next(variables), epoch, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), expected_epoch_delta, EPOCH_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), latitude, LATITUDE_CDF_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), expected_latitude_label, LATITUDE_LABEL_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), cr, CR_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), plasma_speed, SPEED_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), proton_density, PROTON_DENSITY_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), ultraviolet_anisotropy, UV_ANISOTROPY_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), phion, PHION_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), lyman_alpha, LYMAN_ALPHA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), electron_density, ELECTRON_DENSITY_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), plasma_speed_flag, PLASMA_SPEED_FLAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), uv_anisotropy_flag, UV_ANISOTROPY_FLAG_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), proton_density_flag, PROTON_DENSITY_FLAG_CDF_VAR_NAME)
