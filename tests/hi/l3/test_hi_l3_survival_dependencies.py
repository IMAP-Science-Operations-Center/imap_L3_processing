import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock, sentinel, call

from spacepy.pycdf import CDF

from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies, find_glows_l3e_dependencies
from imap_l3_processing.models import UpstreamDataDependency


class TestHiL3SurvivalDependencies(unittest.TestCase):

    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.find_glows_l3e_dependencies")
    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.download_dependency_from_path")
    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.download_dependency")
    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.read_glows_l3e_data")
    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.read_hi_l1c_data")
    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.read_hi_l2_data")
    def test_fetch_dependencies(self, mock_read_hi_l2: Mock, mock_read_hi_l1c, mock_read_glows_l3e,
                                mock_download_dependency,
                                mock_download_dependency_from_path, mock_find_glows_l3e_dependencies):
        l1c_file_paths = ["imap_hi_l1c_45sensor-pset_20201001_v001.cdf", "imap_hi_l1c_45sensor-pset_20201002_v002.cdf",
                          "imap_hi_l1c_45sensor-pset_20201003_v001.cdf"]
        glows_file_paths = [
            "imap_glows_l3e_survival-probability-hi-45_20201001_v001.cdf",
            "imap_glows_l3e_survival-probability-hi-45_20201002_v002.cdf",
            "imap_glows_l3e_survival-probability-hi-45_20201003_v003.cdf"]

        with tempfile.TemporaryDirectory() as tmpdir:
            l2_map_path = Path(tmpdir) / "l2_map.cdf"
            with CDF(str(l2_map_path), masterpath='') as l2_map:
                l2_map.attrs["PARENTS"] = ",".join(l1c_file_paths)

            hi_l2_dependency = UpstreamDataDependency("hi", "l2", None, None, "latest", "45sensor-spacecraft-map")
            glows_dependency = UpstreamDataDependency("glows", "l3e", None, None, "latest", "not-used")
            dependencies = [glows_dependency, hi_l2_dependency]

            mock_download_dependency.return_value = l2_map_path

            mock_download_dependency_from_path.side_effect = [sentinel.l1c_file_path_1, sentinel.l1c_file_path_2,
                                                              sentinel.l1c_file_path_3]
            mock_read_hi_l1c.side_effect = [sentinel.l1c_data_1, sentinel.l1c_data_2, sentinel.l1c_data_3]

            mock_find_glows_l3e_dependencies.return_value = glows_file_paths
            mock_read_glows_l3e.side_effect = [sentinel.glows_data_1, sentinel.glows_data_2, sentinel.glows_data_3]

            actual = HiL3SurvivalDependencies.fetch_dependencies(dependencies)

            mock_download_dependency.assert_called_once_with(hi_l2_dependency)
            mock_read_hi_l2.assert_called_once_with(mock_download_dependency.return_value)

            mock_download_dependency_from_path.assert_has_calls([call(path) for path in l1c_file_paths])

            mock_read_hi_l1c.assert_has_calls([
                call(sentinel.l1c_file_path_1),
                call(sentinel.l1c_file_path_2),
                call(sentinel.l1c_file_path_3)
            ])

            mock_read_glows_l3e.assert_has_calls([call(path) for path in glows_file_paths])

            self.assertEqual(actual.l2_data, mock_read_hi_l2.return_value)
            self.assertEqual(actual.pset_data, [(sentinel.l1c_data_1, sentinel.glows_data_1),
                                                (sentinel.l1c_data_2, sentinel.glows_data_2),
                                                (sentinel.l1c_data_3, sentinel.glows_data_3)])

    @patch("imap_l3_processing.hi.l3.hi_l3_survival_dependencies.imap_data_access.query")
    def test_find_glows_l3e_dependencies(self, mock_data_access_query):
        l1c_90sensor_file_paths = [Path("imap_hi_l1c_90sensor-pset_20201001_v001.cdf"),
                                   Path("imap_hi_l1c_90sensor-pset_20201002_v002.cdf"),
                                   Path("imap_hi_l1c_90sensor-pset_20201003_v001.cdf")]
        l1c_45sensor_file_paths = [Path("imap_hi_l1c_45sensor-pset_20210509_v001.cdf"),
                                   Path("imap_hi_l1c_45sensor-pset_20210508_v002.cdf"),
                                   Path("imap_hi_l1c_45sensor-pset_20210507_v001.cdf")]

        test_cases = [
            (l1c_90sensor_file_paths, "90", "20201001", "20201003"),
            (l1c_45sensor_file_paths, "45", "20210507", "20210509"),
        ]

        mock_data_access_query.return_value = [{"file_path": "glows_1"},
                                               {"file_path": "glows_2"},
                                               {"file_path": "glows_3"}]

        for l1c_file_paths, sensor, expected_start_date, expected_end_date in test_cases:
            with self.subTest(f"sensor: {sensor}"):
                glows_file_paths = find_glows_l3e_dependencies(l1c_file_paths)

                mock_data_access_query.assert_called_with(instrument="glows",
                                                          data_level="l3e",
                                                          descriptor=f"survival-probabilities-hi-{sensor}",
                                                          start_date=expected_start_date,
                                                          end_date=expected_end_date,
                                                          version="latest")

                self.assertEqual([Path("glows_1"), Path("glows_2"), Path("glows_3")],
                                 glows_file_paths)
