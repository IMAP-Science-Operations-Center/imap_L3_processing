import unittest
from pathlib import Path
from unittest.mock import patch, call

from imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies, \
    F107_FLUX_TABLE_URL, \
    LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from imap_l3_processing.models import UpstreamDataDependency


class TestGlowsInitializerAncillaryDependencies(unittest.TestCase):

    @patch("imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies.download_external_dependency")
    @patch("imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies.download_dependency")
    def test_fetch_dependencies(self, mock_download_dependency, mock_download_external_dependency):
        uv_anisotropy_factor = Path("uv_anisotropy_factor")
        waw_helioion_mp_path = Path("waw_helioion_mp_path")

        f107_index_path = Path("f107_fluxtable.txt")
        lyman_alpha_path = Path("lyman_alpha_composite.nc")
        omni_2_path = Path("omni2_all_years.dat")

        mock_download_dependency.side_effect = [
            uv_anisotropy_factor,
            waw_helioion_mp_path,
        ]

        mock_download_external_dependency.side_effect = [
            f107_index_path,
            lyman_alpha_path,
            omni_2_path
        ]
        uv_anisotropy_factor_dependency = UpstreamDataDependency("glows", "l3", None, None,
                                                                 "latest",
                                                                 descriptor="uv-anisotropy-factor")
        waw_helioion_mp_dependency = UpstreamDataDependency("glows", "l3", None, None,
                                                            "latest",
                                                            descriptor="waw-helioion-mp")

        actual_dependencies = GlowsInitializerAncillaryDependencies.fetch_dependencies()
        self.assertIsInstance(actual_dependencies, GlowsInitializerAncillaryDependencies)

        self.assertEqual(uv_anisotropy_factor, actual_dependencies.uv_anisotropy_path)
        self.assertEqual(waw_helioion_mp_path, actual_dependencies.waw_helioion_mp_path)
        self.assertEqual(f107_index_path, actual_dependencies.f107_index_file_path)
        self.assertEqual(lyman_alpha_path, actual_dependencies.lyman_alpha_path)
        self.assertEqual(omni_2_path, actual_dependencies.omni2_data_path)

        self.assertEqual([
            call(uv_anisotropy_factor_dependency),
            call(waw_helioion_mp_dependency)
        ], mock_download_dependency.call_args_list)

        self.assertEqual([
            call(F107_FLUX_TABLE_URL, f107_index_path.name),
            call(LYMAN_ALPHA_COMPOSITE_INDEX_URL, lyman_alpha_path.name),
            call(OMNI2_URL, omni_2_path.name)
        ], mock_download_external_dependency.call_args_list)
