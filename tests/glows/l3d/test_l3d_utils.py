import unittest

import numpy as np

from imap_l3_processing.glows.l3d.utils import create_glows_l3c_dictionary_from_cdf, \
    create_glows_l3b_dictionary_from_cdf
from tests.test_helpers import get_test_data_path, assert_dict_close


class TestL3dUtils(unittest.TestCase):
    def test_create_glows_l3c_dictionary_from_cdf(self):
        expected: dict = {
            'solar_wind_profile': {
                'proton_density': np.array([2.3197076, 2.2874057, 2.1938286, 2.5905547, 3.4460852, 4.4701824,
                                            5.5983787, 6.1791005, 7.7572236, 9.174307, 7.96888, 6.582614,
                                            5.348592, 4.2680264, 3.3404691, 2.4287271, 2.0275042, 1.9867367,
                                            1.9867367], dtype=np.float32),
                'plasma_speed': np.array([522., 526., 538., 491., 415., 351., 300., 279., 234., 204., 229.,
                                          266., 310., 362., 423., 509., 561., 567., 567.], dtype=np.float32)
            },
            'solar_wind_ecliptic': {
                'proton_density': 6.015008449554443,
                'alpha_abundance': 0.02850634977221489,
            }
        }

        l3c_path = get_test_data_path('glows/imap_glows_l3c_sw-profile_20101030_v008.cdf')

        actual: dict = create_glows_l3c_dictionary_from_cdf(l3c_path)

        self.assertEqual(expected['solar_wind_ecliptic']['proton_density'],
                         actual['solar_wind_ecliptic']['proton_density'])
        self.assertEqual(expected['solar_wind_ecliptic']['alpha_abundance'],
                         actual['solar_wind_ecliptic']['alpha_abundance'])

        np.testing.assert_array_equal(actual['solar_wind_profile']['proton_density'],
                                      expected['solar_wind_profile']['proton_density'])
        np.testing.assert_array_equal(actual['solar_wind_profile']['plasma_speed'],
                                      expected['solar_wind_profile']['plasma_speed'])

    def test_create_glows_l3d_dictionary_from_cdf(self):
        l3b_path = get_test_data_path("glows/imap_glows_l3b_ion-rate-profile_20100519_v012.cdf")

        expected: dict = {
            'uv_anisotropy_factor': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                     1., 1., 1.],
            'ion_rate_profile': {
                'lat_grid': [-90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10.,
                             20., 30., 40., 50., 60., 70., 80., 90.],
                'ph_rate': [1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07]
            }
        }

        actual: dict = create_glows_l3b_dictionary_from_cdf(l3b_path)

        assert_dict_close(actual, expected)
