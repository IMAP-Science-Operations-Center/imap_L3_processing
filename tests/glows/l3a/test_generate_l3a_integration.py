import unittest
import warnings
from pathlib import Path

from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.glows_toolkit.l3a_data import L3aData
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_dictionary_from_cdf
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, assert_dict_close


class TestGenerateL3aIntegration(unittest.TestCase):
    def test_generate_l3a_integration(self):
        cdf_data = CDF(str(get_test_data_path('glows/imap_glows_l2_hist_20130908-repoint00001_v004.cdf')))
        l2_glows_data = read_l2_glows_data(cdf_data)

        dependencies = GlowsL3ADependencies(l2_glows_data, 5, {
            "calibration_data": get_test_instrument_team_data_path(
                "glows/imap_glows_calibration-data_20250707_v000.dat"),
            "settings": get_test_instrument_team_data_path("glows/imap_glows_pipeline-settings_20250707_v002.json"),
            "time_dependent_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_time-dep-bckgrd_20250707_v000.dat"),
            "extra_heliospheric_bckgrd": get_test_instrument_team_data_path(
                'glows/imap_glows_map-of-extra-helio-bckgrd_20250707_v000.dat'),
        })
        l3a_path = get_test_data_path('glows/imap_glows_l3a_hist_20130908_v001.cdf')
        expected_l3a = create_glows_l3a_dictionary_from_cdf(Path(l3a_path))

        warnings.filterwarnings('ignore', category=UserWarning)
        l3a_data = L3aData(dependencies.ancillary_files)
        l3a_data.process_l2_data_file(dependencies.data)
        l3a_data.generate_l3a_data(dependencies.ancillary_files)
        warnings.resetwarnings()

        assert_dict_close(l3a_data.data['daily_lightcurve'], expected_l3a['daily_lightcurve'])


if __name__ == '__main__':
    unittest.main()
