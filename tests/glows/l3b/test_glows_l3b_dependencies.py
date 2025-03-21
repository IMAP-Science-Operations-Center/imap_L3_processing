import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call

from imap_l3_processing.glows.l3b.glows_l3b_dependencies import GlowsL3BDependencies
from imap_l3_processing.models import UpstreamDataDependency


class TestGlowsL3BDependencies(unittest.TestCase):
    @patch("imap_l3_processing.glows.l3b.glows_l3b_dependencies.read_glows_l3a_data")
    @patch("imap_l3_processing.glows.l3b.glows_l3b_dependencies.read_l3a_alpha_sw_swapi_data")
    @patch("imap_l3_processing.glows.l3b.glows_l3b_dependencies.CDF")
    @patch("imap_l3_processing.glows.l3b.glows_l3b_dependencies.download_dependency")
    def test_fetch_dependencies(self, mock_download_dependency, mock_cdf_class, mock_read_swapi_l3a_data,
                                mock_read_glows_l3a_data):
        l3a_glows_dependency = UpstreamDataDependency("glows", "l3a", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                      "v001",
                                                      descriptor="hist")
        l3a_swapi_dependency = UpstreamDataDependency("swapi", "l3a", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                      "v001",
                                                      descriptor="alpha-sw")
        bad_day_dependency = UpstreamDataDependency("glows", "l3", None, None,
                                                    "latest",
                                                    descriptor="bad-day-list")
        uv_anisotropy_factor_dependency = UpstreamDataDependency("glows", "l3", None, None,
                                                                 "latest",
                                                                 descriptor="uv-anisotropy-factor")
        waw_helioion_mp_dependency = UpstreamDataDependency("glows", "l3", None, None,
                                                            "latest",
                                                            descriptor="waw-helioion-mp")

        glows_l3a_path = Path("l3a_lightcurve_path")
        swapi_l3a = Path("swapi_l3a")

        bad_day_path = Path("bad_day_path")
        uv_anisotropy_factor = Path("uv_anisotropy_factor")
        waw_helioion_mp_path = Path("waw_helioion_mp_path")

        mock_download_dependency.side_effect = [
            glows_l3a_path,
            swapi_l3a,
            bad_day_path,
            uv_anisotropy_factor,
            waw_helioion_mp_path,
        ]

        actual_dependencies = GlowsL3BDependencies.fetch_dependencies([l3a_glows_dependency, l3a_swapi_dependency])
        self.assertIsInstance(actual_dependencies, GlowsL3BDependencies)

        mock_cdf_class.assert_has_calls([
            call(str(glows_l3a_path)),
            call(str(swapi_l3a))])
        self.assertEqual(actual_dependencies.glows_l3a_data, mock_read_glows_l3a_data.return_value)
        self.assertEqual(actual_dependencies.swapi_l3a_alpha_sw_data, mock_read_swapi_l3a_data.return_value)
        self.assertEqual(bad_day_path, actual_dependencies.ancillary_files["bad_day_list"])
        self.assertEqual(uv_anisotropy_factor, actual_dependencies.ancillary_files["uv_anisotropy_factor"])
        self.assertEqual(waw_helioion_mp_path, actual_dependencies.ancillary_files["waw_helioion_mp"])

        self.assertEqual([
            call(l3a_glows_dependency),
            call(l3a_swapi_dependency),
            call(bad_day_dependency),
            call(uv_anisotropy_factor_dependency),
            call(waw_helioion_mp_dependency)
        ], mock_download_dependency.call_args_list)
