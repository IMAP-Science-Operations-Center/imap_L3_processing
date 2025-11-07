import json
from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import sentinel

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
            'omni_raw_data': get_test_instrument_team_data_path('glows/omni2_all_years.dat')
        }
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20250514_v002.json'),
            'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_v002.json'),
            'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
            'pipeline_settings': get_test_data_path(
                "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json")
        }
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = []
        l3a_file_names = [f"imap_glows_l3a_hist_2010010{x}-repoint0000{x}_v001.cdf" for x in (1, 2, 3)]
        for name in l3a_file_names:
            l3a_data.append(create_glows_l3a_dictionary_from_cdf(l3a_data_folder_path / name))

        dependencies = GlowsL3BCDependencies(
            l3a_data=l3a_data,
            external_files=external_files,
            ancillary_files=ancillary_files,
            carrington_rotation_number=cr,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
            version=1,
            repointing_file_path=sentinel.repointing_file_path
        )

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
            'omni_raw_data': get_test_instrument_team_data_path('glows/omni2_all_years.dat')
        }
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
            'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
            'pipeline_settings': get_test_instrument_team_data_path(
                'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
        }
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [
            create_glows_l3a_dictionary_from_cdf(
                l3a_data_folder_path / 'imap_glows_l3a_hist_20100201-repoint00032_v001.cdf')]

        dependencies = GlowsL3BCDependencies(
            l3a_data=l3a_data,
            external_files=external_files,
            ancillary_files=ancillary_files,
            carrington_rotation_number=cr,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
            version=1,
            repointing_file_path=sentinel.repointing_file_path
        )

        with self.assertRaises(CannotProcessCarringtonRotationError) as context:
            generate_l3bc(dependencies)
        self.assertTrue("All days for Carrington Rotation are in a bad season." in str(context.exception))

    def test_generate_l3bc_appends_used_l3a_files_to_header(self):
        cr = 2096
        external_files = {
            'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            'omni_raw_data': get_test_instrument_team_data_path('glows/omni2_all_years.dat')
        }
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
            'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
            'pipeline_settings': get_test_data_path('glows/imap_glows_pipeline-settings-l3bcde_20250423_v001.json')
        }
        l3a_files = [
            "imap_glows_l3a_hist_20100422-repoint00112_v012.cdf",
            "imap_glows_l3a_hist_20100423-repoint00113_v012.cdf",
            "imap_glows_l3a_hist_20100424-repoint00114_v012.cdf",
            "imap_glows_l3a_hist_20100425-repoint00115_v012.cdf",
            "imap_glows_l3a_hist_20100426-repoint00116_v012.cdf",
            "imap_glows_l3a_hist_20100427-repoint00117_v012.cdf",
            "imap_glows_l3a_hist_20100428-repoint00118_v012.cdf",
            "imap_glows_l3a_hist_20100429-repoint00119_v012.cdf",
            "imap_glows_l3a_hist_20100430-repoint00120_v012.cdf",
            "imap_glows_l3a_hist_20100501-repoint00121_v012.cdf",
            "imap_glows_l3a_hist_20100502-repoint00122_v012.cdf",
            "imap_glows_l3a_hist_20100503-repoint00123_v012.cdf",
            "imap_glows_l3a_hist_20100504-repoint00124_v012.cdf",
            "imap_glows_l3a_hist_20100505-repoint00125_v012.cdf",
            "imap_glows_l3a_hist_20100506-repoint00126_v012.cdf",
            "imap_glows_l3a_hist_20100507-repoint00127_v012.cdf",
            "imap_glows_l3a_hist_20100508-repoint00128_v012.cdf",
            "imap_glows_l3a_hist_20100509-repoint00129_v012.cdf",
            "imap_glows_l3a_hist_20100510-repoint00130_v012.cdf",
            "imap_glows_l3a_hist_20100511-repoint00131_v012.cdf",
            "imap_glows_l3a_hist_20100512-repoint00132_v012.cdf",
            "imap_glows_l3a_hist_20100513-repoint00133_v012.cdf",
            "imap_glows_l3a_hist_20100514-repoint00134_v012.cdf",
            "imap_glows_l3a_hist_20100515-repoint00135_v012.cdf",
            "imap_glows_l3a_hist_20100516-repoint00136_v012.cdf",
            "imap_glows_l3a_hist_20100517-repoint00137_v012.cdf",
            "imap_glows_l3a_hist_20100518-repoint00138_v012.cdf",
        ]

        l3a_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [create_glows_l3a_dictionary_from_cdf(l3a_folder_path / file) for file in l3a_files]

        dependencies = GlowsL3BCDependencies(
            l3a_data=l3a_data,
            external_files=external_files,
            ancillary_files=ancillary_files,
            carrington_rotation_number=cr,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1),
            version=1,
            repointing_file_path=sentinel.repointing_file_path
        )
        l3b, l3c = generate_l3bc(dependencies)

        expected_l3a_parent_files = [
            "imap_glows_l3a_hist_20100511-repoint00131_v012.cdf",
            "imap_glows_l3a_hist_20100512-repoint00132_v012.cdf",
            "imap_glows_l3a_hist_20100513-repoint00133_v012.cdf",
            "imap_glows_l3a_hist_20100514-repoint00134_v012.cdf",
            "imap_glows_l3a_hist_20100515-repoint00135_v012.cdf",
            "imap_glows_l3a_hist_20100516-repoint00136_v012.cdf",
            "imap_glows_l3a_hist_20100517-repoint00137_v012.cdf",
            "imap_glows_l3a_hist_20100518-repoint00138_v012.cdf",
        ]

        self.assertEqual(expected_l3a_parent_files, l3b['header']['l3a_input_files_name'])
