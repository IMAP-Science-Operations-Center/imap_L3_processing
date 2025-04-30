import tempfile
import unittest
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3d.utils import create_glows_l3c_dictionary_from_cdf, \
    create_glows_l3b_dictionary_from_cdf, convert_json_l3d_to_cdf, get_l3a_parent_files_from_l3b
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

        self.assertEqual('imap_glows_l3c_sw-profile_20101030_v008.cdf', actual['header']['filename'])

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
            'header': {
                'filename': 'imap_glows_l3b_ion-rate-profile_20100519_v012.cdf',
                'l3a_input_files_name': [
                    'imap_glows_l3a_hist_20100518-repoint00138_v012.cdf',
                    'imap_glows_l3a_hist_20100520-repoint00139_v012.cdf',
                    'imap_glows_l3a_hist_20100521-repoint00140_v012.cdf',
                    'imap_glows_l3a_hist_20100522-repoint00141_v012.cdf',
                    'imap_glows_l3a_hist_20100523-repoint00142_v012.cdf',
                    'imap_glows_l3a_hist_20100524-repoint00143_v012.cdf',
                    'imap_glows_l3a_hist_20100525-repoint00144_v012.cdf',
                    'imap_glows_l3a_hist_20100526-repoint00145_v012.cdf',
                    'imap_glows_l3a_hist_20100527-repoint00146_v012.cdf',
                    'imap_glows_l3a_hist_20100528-repoint00147_v012.cdf',
                    'imap_glows_l3a_hist_20100529-repoint00148_v012.cdf',
                    'imap_glows_l3a_hist_20100530-repoint00149_v012.cdf',
                    'imap_glows_l3a_hist_20100531-repoint00150_v012.cdf',
                    'imap_glows_l3a_hist_20100601-repoint00151_v012.cdf',
                    'imap_glows_l3a_hist_20100602-repoint00152_v012.cdf',
                    'imap_glows_l3a_hist_20100603-repoint00153_v012.cdf',
                    'imap_glows_l3a_hist_20100604-repoint00154_v012.cdf',
                    'imap_glows_l3a_hist_20100605-repoint00155_v012.cdf',
                    'imap_glows_l3a_hist_20100606-repoint00156_v012.cdf',
                    'imap_glows_l3a_hist_20100607-repoint00157_v012.cdf',
                    'imap_glows_l3a_hist_20100608-repoint00158_v012.cdf',
                    'imap_glows_l3a_hist_20100609-repoint00159_v012.cdf',
                    'imap_glows_l3a_hist_20100610-repoint00160_v012.cdf',
                    'imap_glows_l3a_hist_20100611-repoint00161_v012.cdf',
                    'imap_glows_l3a_hist_20100612-repoint00162_v012.cdf',
                    'imap_glows_l3a_hist_20100613-repoint00163_v012.cdf',
                    'imap_glows_l3a_hist_20100614-repoint00164_v012.cdf',
                    'imap_glows_l3a_hist_20100615-repoint00165_v012.cdf',
                    'imap_glows_l3a_hist_20100616-repoint00165_v012.cdf',
                ]
            },
            'uv_anisotropy_factor': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            'ion_rate_profile': {
                'lat_grid': [-90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60.,
                             70., 80., 90.],
                'ph_rate': [1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07, 1.0643349e-07,
                            1.0643349e-07, 1.0643349e-07, 1.0643349e-07]
            }
        }

        actual: dict = create_glows_l3b_dictionary_from_cdf(l3b_path)
        np.testing.assert_array_equal(expected["header"]['l3a_input_files_name'],
                                      actual["header"]['l3a_input_files_name'])
        np.testing.assert_array_equal(expected["header"]['filename'],
                                      actual["header"]['filename'])
        np.testing.assert_array_equal(expected["uv_anisotropy_factor"],
                                      actual["uv_anisotropy_factor"])
        np.testing.assert_array_equal(expected["ion_rate_profile"]['lat_grid'],
                                      actual["ion_rate_profile"]['lat_grid'])
        np.testing.assert_array_almost_equal(expected["ion_rate_profile"]['ph_rate'],
                                             actual["ion_rate_profile"]['ph_rate'])

    def test_get_l3a_parent_files_from_l3b(self):
        expected_filenames = [
            'imap_glows_l3a_hist_20100511-repoint00131_v011.cdf',
            'imap_glows_l3a_hist_20100512-repoint00132_v011.cdf',
            'imap_glows_l3a_hist_20100513-repoint00133_v011.cdf',
            'imap_glows_l3a_hist_20100514-repoint00134_v011.cdf',
            'imap_glows_l3a_hist_20100515-repoint00135_v011.cdf',
            'imap_glows_l3a_hist_20100516-repoint00136_v011.cdf',
            'imap_glows_l3a_hist_20100517-repoint00137_v011.cdf',
            'imap_glows_l3a_hist_20100518-repoint00138_v011.cdf',
        ]

        actual_file_names = get_l3a_parent_files_from_l3b(
            get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100422_v011.cdf'))

        self.assertEqual(expected_filenames, actual_file_names)

    def test_convert_json_l3d_to_cdf(self):
        with tempfile.TemporaryDirectory() as tempdir:
            convert_json_l3d_to_cdf(get_test_data_path('glows/imap_glows_l3d_cr_2095_v00.json'), Path(tempdir))
            with CDF(str(Path(tempdir) / 'imap_glows_l3d_solar-param-hist_20100326_v000.cdf')) as cdf:
                np.testing.assert_array_equal(
                    [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                    cdf['lat_grid'][...])

                self.assertEqual(846, len(cdf['cr_grid'][...]))
                self.assertEqual(1250.5, cdf['cr_grid'][0])
                self.assertEqual(2095.5, cdf['cr_grid'][-1])

                self.assertEqual(846, len(cdf['time_grid'][...]))
                self.assertEqual("1947-03-03 07:35:32.640", cdf['time_grid'][0])
                self.assertEqual("2010-04-08 22:40:35.040", cdf['time_grid'][-1])

                self.assertEqual(846, len(cdf['speed'][...]))
                np.testing.assert_allclose(
                    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                     -1.0, -1.0, -1.0], cdf['speed'][0], rtol=1e-6)
                np.testing.assert_allclose(
                    [557.4973128404005, 559.2167280154983, 566.8138038909882, 536.7419914396169, 482.3895941633434,
                     436.7778976653025, 395.8709957198084, 373.2087723735307, 358.453450972747, 352.50397782098383,
                     353.5757902723552, 361.29520544745316, 399.6768439688289, 446.0225762645189, 490.864330739225,
                     554.518598443535, 594.9787143968655, 599.4175447470614, 599.4175447470614], cdf['speed'][-1],
                    rtol=1e-6)

                self.assertEqual(846, len(cdf['p_dens'][...]))
                np.testing.assert_allclose(
                    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                     -1.0, -1.0, -1.0], cdf['p_dens'][0], rtol=1e-6)
                np.testing.assert_allclose(
                    [2.7539010174985328, 2.737096030276837, 2.6641209660214256, 2.9648193397292495, 3.6167913862691368,
                     4.267522103200842, 4.954865982954802, 5.4089727098360765, 5.753718175867142, 5.912686119174109,
                     5.875275512094413, 5.687043093803157, 4.889661092087035, 4.129209143417914, 3.496191265226951,
                     2.764380801055958, 2.403839571256985, 2.3686555409547543, 2.3686555409547543
                     ], cdf['p_dens'][-1], rtol=1e-6)

                self.assertEqual(846, len(cdf['uv_anis'][...]))
                np.testing.assert_allclose(
                    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                     -1.0, -1.0, -1.0], cdf['uv_anis'][0], rtol=1e-6)
                np.testing.assert_allclose(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                     ], cdf['uv_anis'][-1], rtol=1e-6)

                self.assertEqual(846, len(cdf['phion'][...]))
                np.testing.assert_allclose(1.830438692701064e-07, cdf['phion'][0], rtol=1e-6)
                np.testing.assert_allclose(1.0870768678455447e-07, cdf['phion'][-1], rtol=1e-6)

                self.assertEqual(846, len(cdf['lya'][...]))
                np.testing.assert_allclose(619975763789.9209, cdf['lya'][0], rtol=1e-6)
                np.testing.assert_allclose(391561526005.0816, cdf['lya'][-1], rtol=1e-6)

                self.assertEqual(846, len(cdf['e_dens'][...]))
                self.assertEqual(-1.0, cdf['e_dens'][0])
                np.testing.assert_allclose(5.7140570690060715, cdf['e_dens'][-1], rtol=1e-6)
