import unittest
from unittest.mock import Mock, sentinel, patch, call, MagicMock

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from tests.test_helpers import get_test_data_path


class TestGlowsL3DDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.query')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.ZipFile')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.download')
    def test_fetch_dependencies(self, mock_download, mock_zip_file_class, mock_query):
        waw_helio_ion_mp_speed = get_test_data_path("glows/imap_glows_plasma-speed-Legendre-2010a_v001.dat")
        waw_helio_ion_mp_p_dens = get_test_data_path("glows/imap_glows_proton-density-Legendre-2010a_v001.dat")
        waw_helio_ion_mp_uv_anis = get_test_data_path("glows/imap_glows_uv-anisotropy-2010a_v001.dat")
        waw_helio_ion_mp_phion = get_test_data_path("glows/imap_glows_photoion-2010a_v001.dat")
        waw_helio_ion_mp_lya = get_test_data_path("glows/imap_glows_lya-2010a_v001.dat")
        waw_helio_ion_mp_e_dens = get_test_data_path("glows/imap_glows_electron-density-2010a_v001.dat")
        pipeline_settings = get_test_data_path('glows/imap_glows_pipeline-settings-L3bc_v001.json')
        external_dependency_zip_path = Mock()

        mock_processing_input_collection = Mock(spec=ProcessingInputCollection)

        l3b_file_1 = {'file_path': Mock()}
        l3b_file_2 = {'file_path': Mock()}
        l3c_file_1 = {'file_path': Mock()}
        l3c_file_2 = {'file_path': Mock()}

        mock_query.side_effect = [
            [l3b_file_1, l3b_file_2],
            [l3c_file_1, l3c_file_2]
        ]

        mock_processing_input_collection.get_file_paths.side_effect = [
            [sentinel.waw_helio_ion_mp_speed],
            [sentinel.waw_helio_ion_mp_p_dens],
            [sentinel.waw_helio_ion_mp_uv_anis],
            [sentinel.waw_helio_ion_mp_phion],
            [sentinel.waw_helio_ion_mp_lya],
            [sentinel.waw_helio_ion_mp_e_dens],
            [sentinel.pipeline_settings],
            [sentinel.external_dependency_zip],
        ]

        mock_download.side_effect = [
            waw_helio_ion_mp_speed,
            waw_helio_ion_mp_p_dens,
            waw_helio_ion_mp_uv_anis,
            waw_helio_ion_mp_phion,
            waw_helio_ion_mp_lya,
            waw_helio_ion_mp_e_dens,
            pipeline_settings,
            external_dependency_zip_path,
            sentinel.l3b_downloaded_path_1,
            sentinel.l3b_downloaded_path_2,
            sentinel.l3c_downloaded_path_1,
            sentinel.l3c_downloaded_path_2,
        ]

        mock_zip_file = MagicMock()
        mock_zip_file_class.return_value.__enter__.return_value = mock_zip_file

        actual_dependencies: GlowsL3DDependencies = GlowsL3DDependencies.fetch_dependencies(
            mock_processing_input_collection)

        mock_query.assert_has_calls([
            call(instrument='glows', descriptor='ion-rate-profile', version='latest'),
            call(instrument='glows', descriptor='sw-profile', version='latest'),
        ])

        mock_processing_input_collection.get_file_paths.assert_has_calls([
            call(source='glows', descriptor='plasma-speed-Legendre-2010a'),
            call(source='glows', descriptor='proton-density-Legendre-2010a'),
            call(source='glows', descriptor='uv-anisotropy-2010a'),
            call(source='glows', descriptor='photoion-2010a'),
            call(source='glows', descriptor='lya-2010a'),
            call(source='glows', descriptor='electron-density-2010a'),
            call(source='glows', descriptor='pipeline-settings-l3bc'),
            call(source='glows', descriptor='l3b-archive'),
        ])

        mock_download.assert_has_calls([
            call(str(sentinel.waw_helio_ion_mp_speed)),
            call(str(sentinel.waw_helio_ion_mp_p_dens)),
            call(str(sentinel.waw_helio_ion_mp_uv_anis)),
            call(str(sentinel.waw_helio_ion_mp_phion)),
            call(str(sentinel.waw_helio_ion_mp_lya)),
            call(str(sentinel.waw_helio_ion_mp_e_dens)),
            call(str(sentinel.pipeline_settings)),
            call(str(sentinel.external_dependency_zip)),
            call(l3b_file_1['file_path']),
            call(l3b_file_2['file_path']),
            call(l3c_file_1['file_path']),
            call(l3c_file_2['file_path']),
        ])

        mock_zip_file_class.assert_called_with(external_dependency_zip_path, 'r')
        mock_zip_file.extract.assert_has_calls([
            call('lyman_alpha_composite.nc', TEMP_CDF_FOLDER_PATH),
        ])

        self.assertEqual({
            'pipeline_settings': pipeline_settings,
            'WawHelioIon': {
                'speed': waw_helio_ion_mp_speed,
                'p-dens': waw_helio_ion_mp_p_dens,
                'uv-anis': waw_helio_ion_mp_uv_anis,
                'phion': waw_helio_ion_mp_phion,
                'lya': waw_helio_ion_mp_lya,
                'e-dens': waw_helio_ion_mp_e_dens
            }}, actual_dependencies.ancillary_files)

        self.assertEqual([sentinel.l3b_downloaded_path_1, sentinel.l3b_downloaded_path_2],
                         actual_dependencies.l3b_file_paths)

        self.assertEqual([sentinel.l3c_downloaded_path_1, sentinel.l3c_downloaded_path_2],
                         actual_dependencies.l3c_file_paths)

        self.assertEqual({
            'lya_raw_data': TEMP_CDF_FOLDER_PATH / 'lyman_alpha_composite.nc'
        }, actual_dependencies.external_files)
