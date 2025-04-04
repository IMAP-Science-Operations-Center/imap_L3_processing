import json
from datetime import datetime
from unittest import TestCase

from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.science.generate_l3bc import generate_l3bc
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, assert_dict_close


class TestGenerateL3BC(TestCase):
    def test_generate_l3bc_integration(self):
        cr = 2091
        external_files = {
            'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            'omni_raw_data': get_test_instrument_team_data_path('glows/omni_2010.dat')
        }
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
            'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
            'pipeline_settings': get_test_instrument_team_data_path('glows/imap_glows_pipeline-settings-L3bc_v001.json')
        }
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = []
        l3a_file_names = [f"imap_glows_l3a_hist_2010010{x}_v001.cdf" for x in (1, 2, 3)]
        for name in l3a_file_names:
            l3a_data.append(create_glows_l3a_dictionary_from_cdf(l3a_data_folder_path / name))

        dependencies = GlowsL3BCDependencies(l3a_data=l3a_data, external_files=external_files,
                                             ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                             start_date=datetime(2025, 4, 3), end_date=datetime(2025, 4, 4))

        actual_l3b, actual_l3c = generate_l3bc(dependencies)

        with open(get_test_instrument_team_data_path("glows/imap_glows_l3b_cr_2091_v00.json")) as f:
            expected_l3b = json.load(f)

        with open(get_test_instrument_team_data_path("glows/imap_glows_l3c_cr_2091_v00.json")) as f:
            expected_l3c = json.load(f)
        del expected_l3b["header"]
        del expected_l3c["header"]
        del actual_l3b["header"]
        del actual_l3c["header"]
        assert_dict_close(expected_l3b, actual_l3b)
        assert_dict_close(expected_l3c, actual_l3c)

    def test_generate_l3bc_integration_all_data_in_bad_season(self):
        cr = 2093
        external_files = {
            'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            'omni_raw_data': get_test_instrument_team_data_path('glows/omni_2010.dat')
        }
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
            'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
            'pipeline_settings': get_test_instrument_team_data_path('glows/imap_glows_pipeline-settings-L3bc_v001.json')
        }
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [
            create_glows_l3a_dictionary_from_cdf(l3a_data_folder_path / 'imap_glows_l3a_hist_20100201_v001.cdf')]

        dependencies = GlowsL3BCDependencies(l3a_data=l3a_data, external_files=external_files,
                                             ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                             start_date=datetime(2025, 4, 3), end_date=datetime(2025, 4, 4))

        with self.assertRaises(CannotProcessCarringtonRotationError) as context:
            generate_l3bc(dependencies)
        self.assertTrue("All days for Carrington Rotation are in a bad season." in str(context.exception))
