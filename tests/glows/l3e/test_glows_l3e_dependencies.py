import unittest
from datetime import datetime
from unittest.mock import patch, Mock, call

from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.models import UpstreamDataDependency


class TestGlowsL3EDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3e.glows_l3e_dependencies.download_dependency')
    def test_fetch_dependencies(self, mock_download_dependency):
        mock_l3d = Mock()

        mock_energy_grid = Mock()
        mock_tess_xyz_8 = Mock()
        mock_tess_ang16 = Mock()
        mock_download_dependency.side_effect = [mock_l3d, mock_energy_grid, mock_tess_xyz_8, mock_tess_ang16]

        l3d_dependency = UpstreamDataDependency("glows", "l3d", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                "v001",
                                                descriptor="solar-hist")
        ignored_dependency = UpstreamDataDependency("glows", "l3d", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                    "v001",
                                                    descriptor="not-solar-hist")

        energy_grid_dependency = UpstreamDataDependency("glows", None, None, None,
                                                        "v001",
                                                        descriptor="energy-grid")
        tess_xyz_8_dependency = UpstreamDataDependency("glows", None, None, None,
                                                       "v001",
                                                       descriptor="tess-xyz-8")
        tess_ang16_dependency = UpstreamDataDependency("glows", None, None, None,
                                                       "v001",
                                                       descriptor="tess-ang-16")

        actual = GlowsL3EDependencies.fetch_dependencies([ignored_dependency, l3d_dependency])

        mock_download_dependency.assert_has_calls([call(l3d_dependency),
                                                   call(energy_grid_dependency),
                                                   call(tess_xyz_8_dependency),
                                                   call(tess_ang16_dependency)])

        self.assertEqual(mock_l3d, actual.l3d_data)
        self.assertEqual(mock_energy_grid, actual.energy_grid)
        self.assertEqual(mock_tess_xyz_8, actual.tess_xyz_8)
        self.assertEqual(mock_tess_ang16, actual.tess_ang16)
