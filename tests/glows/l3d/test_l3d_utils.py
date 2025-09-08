import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, sentinel, Mock, call

import numpy as np

import imap_l3_processing
from imap_l3_processing.glows.l3d.models import GlowsL3DSolarParamsHistory
from imap_l3_processing.glows.l3d.utils import create_glows_l3b_json_file_from_cdf, convert_json_to_l3d_data_product, \
    create_glows_l3c_json_file_from_cdf, get_parent_file_names_from_l3d_json, rename_l3d_text_outputs, \
    get_most_recently_uploaded_ancillary
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path, create_glows_mock_query_results


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
            },
            'CR': 2103
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

        self.assertIsInstance(actual['CR'], float)
        self.assertEqual(expected['CR'], actual['CR'])

    @patch('imap_l3_processing.glows.l3d.utils.os')
    @patch('imap_l3_processing.glows.l3d.utils.json')
    @patch('builtins.open', new_callable=mock_open, create=False)
    def test_create_glows_l3b_json_file_from_cdf(self, mock_open_file, mock_json, mock_os):
        l3b_path = get_test_data_path("glows/imap_glows_l3b_ion-rate-profile_20100422_v011.cdf")

        expected: dict = {
            'header': {
                'filename': 'imap_glows_l3b_ion-rate-profile_20100422_v011.cdf',
            },
            'CR': 2096,
            'date': '2010-05-14 15:43:35.562',
            'uv_anisotropy_factor': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            'ion_rate_profile': {
                'lat_grid': [-90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60.,
                             70., 80., 90.],
                'ph_rate': [1.0475524e-07, 1.0475524e-07, 1.0475524e-07, 1.0475524e-07,
                            1.0475524e-07, 1.0475524e-07, 1.0475524e-07, 1.0475524e-07,
                            1.0475524e-07, 1.0475524e-07, 1.0475524e-07, 1.0475524e-07,
                            1.0475524e-07, 1.0475524e-07, 1.0475524e-07, 1.0475524e-07,
                            1.0475524e-07, 1.0475524e-07, 1.0475524e-07]
            },
            "uv_anisotropy_flag": 1000
        }

        json_file = MagicMock()
        mock_open_file.return_value.__enter__.return_value = json_file

        create_glows_l3b_json_file_from_cdf(l3b_path)

        data_path = Path(imap_l3_processing.__file__).parent / 'glows' / 'l3d' / 'science' / 'data_l3b'

        mock_os.makedirs.assert_called_once_with(data_path, exist_ok=True)

        mock_json.dump.assert_called_once()
        actual = mock_json.dump.call_args.args[0]
        self.assertEqual(json_file, mock_json.dump.call_args.args[1])

        mock_open_file.assert_called_once_with(data_path / 'imap_glows_l3b_cr_2096_v011.json', 'w')

        self.assertIsInstance(actual['CR'], int)
        self.assertEqual(expected['CR'], actual['CR'])

        np.testing.assert_array_equal(expected["header"]['filename'],
                                      actual["header"]['filename'])

        self.assertIsInstance(actual['uv_anisotropy_factor'], list)
        np.testing.assert_array_equal(expected["uv_anisotropy_factor"],
                                      actual["uv_anisotropy_factor"])

        np.testing.assert_array_equal(expected["ion_rate_profile"]['lat_grid'],
                                      actual["ion_rate_profile"]['lat_grid'])
        self.assertIsInstance(actual["ion_rate_profile"]["lat_grid"], list)

        np.testing.assert_array_almost_equal(expected["ion_rate_profile"]['ph_rate'],
                                             actual["ion_rate_profile"]['ph_rate'])
        self.assertIsInstance(actual["ion_rate_profile"]["ph_rate"], list)

        self.assertIsInstance(actual["uv_anisotropy_flag"], int)
        self.assertEqual(expected["uv_anisotropy_flag"], actual["uv_anisotropy_flag"])

    def test_convert_json_l3d_to_data_product(self):
        input_metadata = Mock(spec=InputMetadata)
        input_metadata.start_date = datetime.fromisoformat("2025-05-12")
        l3d_data_product: GlowsL3DSolarParamsHistory = convert_json_to_l3d_data_product(
            get_test_data_path('glows/imap_glows_l3d_solar-params-history_19470303-cr02095_v00.json'),
            input_metadata,
            sentinel.parent_file_names,
        )

        self.assertEqual(input_metadata, l3d_data_product.input_metadata)
        self.assertEqual(input_metadata.start_date, datetime.fromisoformat("1947-03-03").date())

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

        self.assertEqual(846, len(l3d_data_product.plasma_speed))
        np.testing.assert_allclose(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0], l3d_data_product.plasma_speed[0], rtol=1e-6)
        np.testing.assert_allclose(
            [
                468.1725019310069,
                470.41196695859816,
                476.08272082078383,
                449.52578318361645,
                401.38306014556065,
                363.7120316891783,
                330.5607405468131,
                305.29544288745444,
                273.7598466912451,
                246.7446552487376,
                266.49159920773036,
                289.2521463358576,
                324.9993546613053,
                359.23215479267344,
                398.4985502246217,
                460.79164287941717,
                503.2472821431676,
                508.93503849297326,
                509.10575708733097
            ],
            l3d_data_product.plasma_speed[-1],
            rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.proton_density))
        np.testing.assert_allclose(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0], l3d_data_product.proton_density[0], rtol=1e-6)
        np.testing.assert_allclose(
            [
                3.134819994443553,
                3.1041926011053467,
                3.0267698024403056,
                3.3898787480759545,
                4.215698147420914,
                4.9924188349188245,
                5.72062210604848,
                6.206572506330694,
                7.007246183637803,
                7.666900424298345,
                7.238530407122755,
                6.691594046599381,
                5.7658475286137065,
                4.954055736090567,
                4.139454391383636,
                3.1651607050304924,
                2.68129704271522,
                2.6272758631787716,
                2.625943370301117
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
        np.testing.assert_allclose(1.0538921234648196e-07, l3d_data_product.phion[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.lyman_alpha))
        np.testing.assert_allclose(619975763789.9209, l3d_data_product.lyman_alpha[0], rtol=1e-6)
        np.testing.assert_allclose(396312787467.8571, l3d_data_product.lyman_alpha[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.electron_density))
        self.assertEqual(-1.0, l3d_data_product.electron_density[0])
        np.testing.assert_allclose(5.978193763000784, l3d_data_product.electron_density[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.plasma_speed_flag))
        self.assertEqual(-1.0, l3d_data_product.plasma_speed_flag[0])
        np.testing.assert_allclose(20001.0, l3d_data_product.plasma_speed_flag[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.uv_anisotropy_flag))
        self.assertEqual(-1.0, l3d_data_product.uv_anisotropy_flag[0])
        np.testing.assert_allclose(30001.0, l3d_data_product.uv_anisotropy_flag[-1], rtol=1e-6)

        self.assertEqual(846, len(l3d_data_product.proton_density_flag))
        self.assertEqual(-1.0, l3d_data_product.proton_density_flag[0])
        np.testing.assert_allclose(10001.0, l3d_data_product.proton_density_flag[-1], rtol=1e-6)

    def test_get_parent_file_names_from_l3d_json(self):
        path_to_l3d_output_folder = get_test_data_path("glows/science/data_l3d")

        l3bc_filenames = get_parent_file_names_from_l3d_json(path_to_l3d_output_folder)

        expected = [
            "imap_glows_plasma-speed-2010a_v003.dat",
            "imap_glows_proton-density-2010a_v003.dat",
            "imap_glows_uv-anisotropy-2010a_v003.dat",
            "imap_glows_photoion-2010a_v003.dat",
            "imap_glows_lya-2010a_v003.dat",
            "imap_glows_electron-density-2010a_v003.dat",
            "lyman_alpha_composite.nc",
            "imap_glows_l3b_ion-rate-profile_20100422_v011.cdf",
            "imap_glows_l3b_ion-rate-profile_20100519_v011.cdf",
            "imap_glows_l3c_sw-profile_20100422_v011.cdf",
            "imap_glows_l3c_sw-profile_20100519_v011.cdf"
        ]

        self.assertCountEqual(expected, l3bc_filenames)

    @patch("imap_l3_processing.glows.l3d.utils.get_date_range_of_cr")
    def test_rename_l3d_text_outputs(self, mock_get_date_range_of_cr):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            original_lya_path = tmpdir / "imap_glows_l3d_lya_19470303-cr02092_v00.dat"
            original_p_dens_path = tmpdir / "imap_glows_l3d_p-dens_19470303-cr02092_v00.dat"

            original_lya_path.write_text("lya")
            original_p_dens_path.write_text("p_dens")

            mock_get_date_range_of_cr.return_value = (datetime(2010, 1, 1), datetime(2010, 2, 1))

            version = "v012"
            actual_new_paths = rename_l3d_text_outputs([original_lya_path, original_p_dens_path], version)

            mock_get_date_range_of_cr.assert_has_calls([call(2092), call(2092)])

            expected_lya_output = tmpdir / "imap_glows_lya_19470303_20100201_v012.dat"
            expected_p_dens_output = tmpdir / "imap_glows_p-dens_19470303_20100201_v012.dat"

            self.assertEqual([expected_lya_output, expected_p_dens_output], actual_new_paths)

            self.assertEqual("lya", expected_lya_output.read_text())
            self.assertEqual("p_dens", expected_p_dens_output.read_text())

            self.assertFalse(original_lya_path.exists())
            self.assertFalse(original_p_dens_path.exists())

    def test_get_most_recently_uploaded_ancillary(self):
        query_result = create_glows_mock_query_results([
            "imap_glows_l3d_solar-hist_20100101-cr02091_v001.cdf",
            "imap_glows_l3d_solar-hist_20100201-cr02092_v002.cdf",
            "imap_glows_l3d_solar-hist_20100301-cr02093_v001.cdf"

        ], ingestion_dates=[datetime(2010, 1, 2), datetime(2010, 5, 2), datetime(2010, 3, 2)])

        [expected] = create_glows_mock_query_results(
            ["imap_glows_l3d_solar-hist_20100201-cr02092_v002.cdf"],
            ingestion_dates=[datetime(2010, 5, 2)])

        self.assertEqual(expected, get_most_recently_uploaded_ancillary(query_result))

    def test_get_most_recently_uploaded_ancillary_with_empty_query(self):
        self.assertIsNone(get_most_recently_uploaded_ancillary([]))
