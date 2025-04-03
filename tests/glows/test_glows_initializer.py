import unittest
from unittest.mock import Mock, patch, call, sentinel

from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies


class TestGlowsInitializer(unittest.TestCase):

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializerAncillaryDependencies")
    def test_validate_and_initialize_returns_empty_list_when_missing_ancillary_dependencies(self,
                                                                                            mock_glows_initializer_ancillary_dependencies: Mock):
        test_cases = [
            ("Missing f107_path", None, Mock(), Mock(), Mock(), Mock(), Mock(), Mock()),
            ("Missing lyman_alpha_path", Mock(), None, Mock(), Mock(), Mock(), Mock(), Mock()),
            ("Missing omni_data_path", Mock(), Mock(), None, Mock(), Mock(), Mock(), Mock()),
            ("Missing uv_anisotropy", Mock(), Mock(), Mock(), None, Mock(), Mock(), Mock()),
            ("Missing waw_helioion_mp", Mock(), Mock(), Mock(), Mock(), None, Mock(), Mock()),
            ("Missing pipeline_settings", Mock(), Mock(), Mock(), Mock(), Mock(), None, Mock()),
            ("Missing bad_days", Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), None),
            ("Missing all dependencies", None, None, None, None, None, None, None),
        ]

        for name, f107_path, lyman_alpha_path, omni_data_path, uv_anisotropy, waw_helioion_mp, pipeline_settings, bad_days in test_cases:
            with self.subTest(name):
                mock_glows_initializer_ancillary_dependencies.fetch_dependencies.return_value = GlowsInitializerAncillaryDependencies(
                    uv_anisotropy, waw_helioion_mp, pipeline_settings, bad_days, f107_path, lyman_alpha_path,
                    omni_data_path)

                self.assertEqual([], GlowsInitializer.validate_and_initialize(""))

    @patch("imap_l3_processing.glows.glows_initializer.archive_dependencies")
    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializerAncillaryDependencies")
    @patch("imap_l3_processing.glows.glows_initializer.query")
    @patch("imap_l3_processing.glows.glows_initializer.find_unprocessed_carrington_rotations")
    def test_validate_and_initialize_l3b(self, mock_find_unprocessed_carrington_rotations: Mock, mock_query: Mock,
                                         mock_glows_initializer_ancillary_dependencies: Mock,
                                         mock_archive_dependencies: Mock):
        version = "v003"
        mock_l3a = Mock()
        mock_l3b = Mock()

        mock_query.side_effect = [
            mock_l3a,
            mock_l3b
        ]
        mock_archive_dependencies.side_effect = [
            sentinel.zip_path_1,
            sentinel.zip_path_2
        ]

        ancillary_dependencies = GlowsInitializerAncillaryDependencies(Mock(), Mock(), Mock(), Mock(), Mock(), Mock(),
                                                                       Mock())

        mock_glows_initializer_ancillary_dependencies.fetch_dependencies.return_value = ancillary_dependencies
        mock_find_unprocessed_carrington_rotations.return_value = [sentinel.cr_to_process1, sentinel.cr_to_process2]

        actual_zip_paths = GlowsInitializer.validate_and_initialize(version)

        self.assertEqual(2, mock_query.call_count)
        mock_query.assert_has_calls([call(instrument="glows", version=version, data_level="l3a"),
                                     call(instrument="glows", version=version, data_level="l3b")])

        mock_find_unprocessed_carrington_rotations.assert_called_once_with(mock_l3a, mock_l3b, ancillary_dependencies)

        mock_glows_initializer_ancillary_dependencies.fetch_dependencies.assert_called_once()

        mock_archive_dependencies.assert_has_calls([
            call(sentinel.cr_to_process1, version, ancillary_dependencies),
            call(sentinel.cr_to_process2, version, ancillary_dependencies),
        ])

        self.assertEqual(2, len(actual_zip_paths))
        self.assertEqual(sentinel.zip_path_1, actual_zip_paths[0])
        self.assertEqual(sentinel.zip_path_2, actual_zip_paths[1])
