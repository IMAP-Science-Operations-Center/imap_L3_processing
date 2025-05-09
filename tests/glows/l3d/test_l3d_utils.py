import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, sentinel

import numpy as np

import imap_l3_processing
from imap_l3_processing.glows.l3d.models import GlowsL3DSolarParamsHistory
from imap_l3_processing.glows.l3d.utils import create_glows_l3b_json_file_from_cdf, convert_json_to_l3d_data_product, \
    create_glows_l3c_json_file_from_cdf, get_parent_file_names_from_l3d_json
from tests.test_helpers import get_test_data_path


class TestL3dUtils(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3d.utils.os')
    @patch('imap_l3_processing.glows.l3d.utils.json')
    @patch('builtins.open', new_callable=mock_open, create=False)
    def test_create_glows_l3c_json_file_from_cdf(self, mock_open_file, mock_json, mock_os):
        l3c_path = get_test_data_path('glows/imap_glows_l3c_sw-profile_20101030_v008.cdf')

        expected: dict = {
            'solar_wind_profile': {
                'proton_density': [2.3197076, 2.2874057, 2.1938286, 2.5905547, 3.4460852, 4.4701824,
                                   5.5983787, 6.1791005, 7.7572236, 9.174307, 7.96888, 6.582614,
                                   5.348592, 4.2680264, 3.3404691, 2.4287271, 2.0275042, 1.9867367,
                                   1.9867367],
                'plasma_speed': [522., 526., 538., 491., 415., 351., 300., 279., 234., 204., 229.,
                                 266., 310., 362., 423., 509., 561., 567., 567.]
            },
            'solar_wind_ecliptic': {
                'proton_density': 6.015008449554443,
                'alpha_abundance': 0.02850634977221489,
            }
        }

        json_file = MagicMock()
        mock_open_file.return_value.__enter__.return_value = json_file

        create_glows_l3c_json_file_from_cdf(l3c_path)

        data_path = Path(imap_l3_processing.__file__).parent / 'glows' / 'l3d' / 'science' / 'data_l3c'

        mock_os.makedirs.assert_called_once_with(data_path, exist_ok=True)

        mock_open_file.assert_called_once_with(data_path / 'imap_glows_l3c_cr_2103_v008.json', 'w')

        mock_json.dump.assert_called_once()
        actual = mock_json.dump.call_args.args[0]
        self.assertEqual(json_file, mock_json.dump.call_args.args[1])
        self.assertEqual('imap_glows_l3c_sw-profile_20101030_v008.cdf', actual['header']['filename'])

        self.assertIsInstance(actual['solar_wind_profile']['proton_density'], list)
        np.testing.assert_allclose(actual['solar_wind_profile']['proton_density'],
                                   expected['solar_wind_profile']['proton_density'])

        self.assertIsInstance(actual['solar_wind_profile']['plasma_speed'], list)
        np.testing.assert_array_equal(actual['solar_wind_profile']['plasma_speed'],
                                      expected['solar_wind_profile']['plasma_speed'])

        self.assertIsInstance(actual['solar_wind_ecliptic']['proton_density'], float)
        self.assertEqual(expected['solar_wind_ecliptic']['proton_density'],
                         actual['solar_wind_ecliptic']['proton_density'])

        self.assertIsInstance(actual['solar_wind_ecliptic']['alpha_abundance'], float)
        self.assertEqual(expected['solar_wind_ecliptic']['alpha_abundance'],
                         actual['solar_wind_ecliptic']['alpha_abundance'])

    @patch('imap_l3_processing.glows.l3d.utils.os')
    @patch('imap_l3_processing.glows.l3d.utils.json')
    @patch('builtins.open', new_callable=mock_open, create=False)
    def test_create_glows_l3b_json_file_from_cdf(self, mock_open_file, mock_json, mock_os):
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
            'CR': 2097,
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

        json_file = MagicMock()
        mock_open_file.return_value.__enter__.return_value = json_file

        create_glows_l3b_json_file_from_cdf(l3b_path)

        data_path = Path(imap_l3_processing.__file__).parent / 'glows' / 'l3d' / 'science' / 'data_l3b'

        mock_os.makedirs.assert_called_once_with(data_path, exist_ok=True)

        mock_json.dump.assert_called_once()
        actual = mock_json.dump.call_args.args[0]
        self.assertEqual(json_file, mock_json.dump.call_args.args[1])

        mock_open_file.assert_called_once_with(data_path / 'imap_glows_l3b_cr_2097_v012.json', 'w')

        self.assertIsInstance(actual['CR'], int)
        self.assertEqual(expected['CR'], actual['CR'])

        np.testing.assert_array_equal(expected["header"]['filename'],
                                      actual["header"]['filename'])

        np.testing.assert_array_equal(expected["header"]['l3a_input_files_name'],
                                      actual["header"]['l3a_input_files_name'])

        self.assertIsInstance(actual['uv_anisotropy_factor'], list)
        np.testing.assert_array_equal(expected["uv_anisotropy_factor"],
                                      actual["uv_anisotropy_factor"])

        np.testing.assert_array_equal(expected["ion_rate_profile"]['lat_grid'],
                                      actual["ion_rate_profile"]['lat_grid'])
        self.assertIsInstance(actual["ion_rate_profile"]["lat_grid"], list)

        np.testing.assert_array_almost_equal(expected["ion_rate_profile"]['ph_rate'],
                                             actual["ion_rate_profile"]['ph_rate'])
        self.assertIsInstance(actual["ion_rate_profile"]["ph_rate"], list)

    def test_convert_json_l3d_to_data_product(self):
        l3d_data_product: GlowsL3DSolarParamsHistory = convert_json_to_l3d_data_product(
            get_test_data_path('glows/imap_glows_l3d_cr_2095_v00.json'),
            sentinel.input_metadata,
            sentinel.parent_file_names,
        )

        self.assertEqual(sentinel.input_metadata, l3d_data_product.input_metadata)

        self.assertEqual(sentinel.parent_file_names, l3d_data_product.parent_file_names)

        np.testing.assert_array_equal(
            [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            l3d_data_product.latitude)

        self.assertEqual(846, len(l3d_data_product.cr))
        self.assertEqual(1250.5, l3d_data_product.cr[0])
        self.assertEqual(2095.5, l3d_data_product.cr[-1])

        self.assertEqual(846, len(l3d_data_product.epoch))
        self.assertEqual(datetime.fromisoformat("1947-03-03 07:35:32.640"), l3d_data_product.epoch[0])
        self.assertEqual(datetime.fromisoformat("2010-04-08 22:40:35.040"), l3d_data_product.epoch[-1])

        np.testing.assert_array_equal(np.arange(-90, 100, step=10), l3d_data_product.latitude)

        self.assertEqual(846, len(l3d_data_product.cr))
        self.assertEqual(1250.5, l3d_data_product.cr[0])
        self.assertEqual(2095.5, l3d_data_product.cr[-1])

        self.assertEqual(846, len(l3d_data_product.speed))
        np.testing.assert_allclose(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0], l3d_data_product.speed[0], rtol=1e-6)
        np.testing.assert_allclose(
            [557.4973128404005, 559.2167280154983, 566.8138038909882, 536.7419914396169, 482.3895941633434,
             436.7778976653025, 395.8709957198084, 373.2087723735307, 358.453450972747, 352.50397782098383,
             353.5757902723552, 361.29520544745316, 399.6768439688289, 446.0225762645189, 490.864330739225,
             554.518598443535, 594.9787143968655, 599.4175447470614, 599.4175447470614], l3d_data_product.speed[-1],
            rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.proton_density))
        np.testing.assert_allclose(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0], l3d_data_product.proton_density[0], rtol=1e-6)
        np.testing.assert_allclose(
            [2.7539010174985328, 2.737096030276837, 2.6641209660214256, 2.9648193397292495, 3.6167913862691368,
             4.267522103200842, 4.954865982954802, 5.4089727098360765, 5.753718175867142, 5.912686119174109,
             5.875275512094413, 5.687043093803157, 4.889661092087035, 4.129209143417914, 3.496191265226951,
             2.764380801055958, 2.403839571256985, 2.3686555409547543, 2.3686555409547543
             ], l3d_data_product.proton_density[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.ultraviolet_anisotropy))
        np.testing.assert_allclose(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0], l3d_data_product.ultraviolet_anisotropy[0], rtol=1e-6)
        np.testing.assert_allclose(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
             ], l3d_data_product.ultraviolet_anisotropy[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.phion))
        np.testing.assert_allclose(1.830438692701064e-07, l3d_data_product.phion[0], rtol=1e-6)
        np.testing.assert_allclose(1.0870768678455447e-07, l3d_data_product.phion[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.lyman_alpha))
        np.testing.assert_allclose(619975763789.9209, l3d_data_product.lyman_alpha[0], rtol=1e-6)
        np.testing.assert_allclose(391561526005.0816, l3d_data_product.lyman_alpha[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.electron_density))
        self.assertEqual(-1.0, l3d_data_product.electron_density[0])
        np.testing.assert_allclose(5.7140570690060715, l3d_data_product.electron_density[-1], rtol=1e-6)

    def test_get_parent_file_names_from_l3d_json(self):
        path_to_l3d_output_folder = get_test_data_path("glows/science/data_l3d")

        l3bc_filenames = get_parent_file_names_from_l3d_json(path_to_l3d_output_folder)

        expected = [
            "imap_glows_plasma-speed-Legendre-2010a_v001.dat",
            "imap_glows_proton-density-Legendre-2010a_v001.dat",
            "imap_glows_uv-anisotropy-2010a_v001.dat",
            "imap_glows_photoion-2010a_v001.dat",
            "imap_glows_lya-2010a_v001.dat",
            "imap_glows_electron-density-2010a_v001.dat",
            "imap_glows_l3b_ion-rate-profile_20100326_v011.cdf",
            "imap_glows_l3b_ion-rate-profile_20100422_v011.cdf",
            "imap_glows_l3b_ion-rate-profile_20100519_v011.cdf",
            "imap_glows_l3c_sw-profile_20100326_v011.cdf",
            "imap_glows_l3c_sw-profile_20100422_v011.cdf",
            "imap_glows_l3c_sw-profile_20100519_v011.cdf",
            'lyman_alpha_composite.nc'
        ]

        self.assertCountEqual(expected, l3bc_filenames)
