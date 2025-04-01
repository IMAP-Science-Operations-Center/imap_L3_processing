import unittest
from pathlib import Path
from unittest.mock import patch, call

from imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies, \
    F107_FLUX_TABLE_URL, \
    LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from tests.glows.l3b.test_utils import create_imap_data_access_json


class TestGlowsInitializerAncillaryDependencies(unittest.TestCase):
    @patch("imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies.query")
    @patch("imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies.download_external_dependency")
    def test_fetch_dependencies(self, mock_download_external_dependency, mock_query):
        uv_anisotropy_factor = create_imap_data_access_json(file_path="path_to_uv_file", data_level=None,
                                                            start_date=None, descriptor="uv-anisotropy-1CR",
                                                            version="latest")
        waw_helioion_mp = create_imap_data_access_json(file_path="path_to_waw_file", data_level=None,
                                                       start_date=None, descriptor="WawHelioIonMP",
                                                       version="latest")

        mock_query.side_effect = [
            [uv_anisotropy_factor],
            [waw_helioion_mp]
        ]

        f107_index_path = Path("f107_fluxtable.txt")
        lyman_alpha_path = Path("lyman_alpha_composite.nc")
        omni_2_path = Path("omni2_all_years.dat")

        mock_download_external_dependency.side_effect = [
            f107_index_path,
            lyman_alpha_path,
            omni_2_path
        ]

        actual_dependencies = GlowsInitializerAncillaryDependencies.fetch_dependencies()
        self.assertIsInstance(actual_dependencies, GlowsInitializerAncillaryDependencies)

        self.assertEqual(uv_anisotropy_factor["file_path"], actual_dependencies.uv_anisotropy_path)
        self.assertEqual(waw_helioion_mp["file_path"], actual_dependencies.waw_helioion_mp_path)
        self.assertEqual(f107_index_path, actual_dependencies.f107_index_file_path)
        self.assertEqual(lyman_alpha_path, actual_dependencies.lyman_alpha_path)
        self.assertEqual(omni_2_path, actual_dependencies.omni2_data_path)

        self.assertEqual([
            call(instrument="glows", descriptor="uv-anisotropy-1CR", version="latest"),
            call(instrument="glows", descriptor="WawHelioIonMP", version="latest"),
        ], mock_query.call_args_list)

        self.assertEqual([
            call(F107_FLUX_TABLE_URL, f107_index_path.name),
            call(LYMAN_ALPHA_COMPOSITE_INDEX_URL, lyman_alpha_path.name),
            call(OMNI2_URL, omni_2_path.name)
        ], mock_download_external_dependency.call_args_list)
