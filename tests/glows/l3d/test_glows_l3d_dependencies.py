import unittest
from unittest.mock import Mock, sentinel, patch, call, MagicMock

from imap_data_access.processing_input import ProcessingInputCollection, AncillaryInput, ScienceInput

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from tests.test_helpers import get_test_data_path


class TestGlowsL3DDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.ZipFile')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.download_dependency_from_path')
    @patch('imap_l3_processing.glows.l3d.glows_l3d_dependencies.create_glows_l3c_dictionary_from_cdf')
    def test_fetch_dependencies(self, mock_create_glows_l3c_dictionary_from_cdf, mock_download_dependency_from_path,
                                mock_zip_file_class):
        waw_helio_ion_mp_speed = get_test_data_path("glows/imap_glows_plasma-speed-Legendre-2010a_v001.dat")
        waw_helio_ion_mp_p_dens = get_test_data_path("glows/imap_glows_proton-density-Legendre-2010a_v001.dat")
        waw_helio_ion_mp_uv_anis = get_test_data_path("glows/imap_glows_uv-anisotropy-2010a_v001.dat")
        waw_helio_ion_mp_phion = get_test_data_path("glows/imap_glows_photoion-2010a_v001.dat")
        waw_helio_ion_mp_lya = get_test_data_path("glows/imap_glows_lya-2010a_v001.dat")
        waw_helio_ion_mp_e_dens = get_test_data_path("glows/imap_glows_electron-density-2010a_v001.dat")
        pipeline_settings = get_test_data_path('glows/imap_glows_pipeline-settings-L3bc_v001.json')
        external_dependency_zip_path = Mock()
        l3c_data = get_test_data_path('glows/imap_glows_l3c_sw-profile_20101030_v008.cdf')

        mock_processing_input_collection = Mock(spec=ProcessingInputCollection)

        mock_processing_input_collection.get_file_paths.side_effect = [
            [sentinel.waw_helio_ion_mp_speed],
            [sentinel.waw_helio_ion_mp_p_dens],
            [sentinel.waw_helio_ion_mp_uv_anis],
            [sentinel.waw_helio_ion_mp_phion],
            [sentinel.waw_helio_ion_mp_lya],
            [sentinel.waw_helio_ion_mp_e_dens],
            [sentinel.pipeline_settings],
            [sentinel.external_dependency_zip],
            [sentinel.l3c_data]
        ]

        mock_download_dependency_from_path.side_effect = [
            waw_helio_ion_mp_speed,
            waw_helio_ion_mp_p_dens,
            waw_helio_ion_mp_uv_anis,
            waw_helio_ion_mp_phion,
            waw_helio_ion_mp_lya,
            waw_helio_ion_mp_e_dens,
            pipeline_settings,
            external_dependency_zip_path,
            l3c_data,
        ]

        mock_create_glows_l3c_dictionary_from_cdf.return_value = sentinel.glows_l3c_data_dictionary

        mock_zip_file = MagicMock()
        mock_zip_file_class.return_value.__enter__.return_value = mock_zip_file

        actual_dependencies: GlowsL3DDependencies = GlowsL3DDependencies.fetch_dependencies(
            mock_processing_input_collection)

        mock_processing_input_collection.get_file_paths.assert_has_calls([
            call(source='glows', descriptor='plasma-speed-legendre'),
            call(source='glows', descriptor='proton-density-legendre'),
            call(source='glows', descriptor='uv-anisotropy'),
            call(source='glows', descriptor='photoion'),
            call(source='glows', descriptor='lya'),
            call(source='glows', descriptor='electron-density'),
            call(source='glows', descriptor='pipeline-settings-l3bc'),
            call(source='glows', descriptor='l3b-archive'),
            call(source='glows', descriptor='solar-params'),
        ])

        mock_download_dependency_from_path.assert_has_calls([
            call(str(sentinel.waw_helio_ion_mp_speed)),
            call(str(sentinel.waw_helio_ion_mp_p_dens)),
            call(str(sentinel.waw_helio_ion_mp_uv_anis)),
            call(str(sentinel.waw_helio_ion_mp_phion)),
            call(str(sentinel.waw_helio_ion_mp_lya)),
            call(str(sentinel.waw_helio_ion_mp_e_dens)),
            call(str(sentinel.pipeline_settings)),
            call(str(sentinel.external_dependency_zip)),
            call(str(sentinel.l3c_data))
        ])

        mock_create_glows_l3c_dictionary_from_cdf.assert_called_once_with(l3c_data)

        mock_zip_file_class.assert_called_with(external_dependency_zip_path, 'r')
        mock_zip_file.extract.assert_has_calls([
            call('lyman_alpha_composite.nc', TEMP_CDF_FOLDER_PATH),
        ])

        self.assertEqual(sentinel.glows_l3c_data_dictionary, actual_dependencies.l3c_data)
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
        self.assertEqual({
            'lya_raw_data': TEMP_CDF_FOLDER_PATH / 'lyman_alpha_composite.nc'
        }, actual_dependencies.external_files)
