import unittest
from unittest.mock import patch, call

from astropy.time import TimeDelta

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import \
    GlowsInitializerAncillaryDependencies, \
    F107_FLUX_TABLE_URL, \
    LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from tests.glows.l3bc.test_utils import create_imap_data_access_json
from tests.test_helpers import get_test_data_path


class TestGlowsInitializerAncillaryDependencies(unittest.TestCase):
    @patch("imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.download")
    @patch("imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.query")
    @patch("imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.download_external_dependency")
    def test_fetch_dependencies(self, mock_download_external_dependency, mock_query, mock_download):
        uv_anisotropy_factor = create_imap_data_access_json(file_path="path_to_uv_file", data_level=None,
                                                            start_date=None, descriptor="uv-anisotropy-1CR",
                                                            version="latest")
        waw_helioion_mp = create_imap_data_access_json(file_path="path_to_waw_file", data_level=None,
                                                       start_date=None, descriptor="WawHelioIonMP",
                                                       version="latest")
        bad_days_list = create_imap_data_access_json(file_path="path_to_bad_days_list", data_level=None,
                                                     start_date=None, descriptor="bad-day-list",
                                                     version="latest")
        pipeline_settings = create_imap_data_access_json(file_path="path_to_pipeline_settings", data_level=None,
                                                         start_date=None, descriptor="pipeline-settings-L3bc",
                                                         version="latest")

        mock_query.side_effect = [
            [uv_anisotropy_factor],
            [waw_helioion_mp],
            [bad_days_list],
            [pipeline_settings]
        ]

        mock_download.return_value = get_test_data_path("glows/imap_glows_pipeline-settings-L3bc.json")

        f107_index_path = TEMP_CDF_FOLDER_PATH / "f107_fluxtable.txt"
        lyman_alpha_path = TEMP_CDF_FOLDER_PATH / "lyman_alpha_composite.nc"
        omni_2_path = TEMP_CDF_FOLDER_PATH / "omni2_all_years.dat"

        mock_download_external_dependency.side_effect = [
            f107_index_path,
            lyman_alpha_path,
            omni_2_path
        ]

        actual_dependencies = GlowsInitializerAncillaryDependencies.fetch_dependencies()
        self.assertIsInstance(actual_dependencies, GlowsInitializerAncillaryDependencies)

        mock_download.assert_called_once_with(pipeline_settings['file_path'])

        self.assertEqual([
            call(instrument="glows", descriptor="uv-anisotropy-1CR", version="latest"),
            call(instrument="glows", descriptor="WawHelioIonMP", version="latest"),
            call(instrument="glows", descriptor="bad-days-list", version="latest"),
            call(instrument="glows", descriptor="pipeline-settings-L3bc", version="latest"),
        ], mock_query.call_args_list)

        self.assertEqual([
            call(F107_FLUX_TABLE_URL, f107_index_path),
            call(LYMAN_ALPHA_COMPOSITE_INDEX_URL, lyman_alpha_path),
            call(OMNI2_URL, omni_2_path)
        ], mock_download_external_dependency.call_args_list)

        self.assertEqual(uv_anisotropy_factor["file_path"], actual_dependencies.uv_anisotropy_path)
        self.assertEqual(waw_helioion_mp["file_path"], actual_dependencies.waw_helioion_mp_path)
        self.assertEqual(bad_days_list["file_path"], actual_dependencies.bad_days_list)
        self.assertEqual(pipeline_settings["file_path"], actual_dependencies.pipeline_settings)
        self.assertEqual(f107_index_path, actual_dependencies.f107_index_file_path)
        self.assertEqual(lyman_alpha_path, actual_dependencies.lyman_alpha_path)
        self.assertEqual(omni_2_path, actual_dependencies.omni2_data_path)
        self.assertEqual(TimeDelta(56, format="jd").value, actual_dependencies.initializer_time_buffer.value)
