import unittest
from unittest.mock import Mock, patch, call

from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies


class TestGlowsInitializer(unittest.TestCase):

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializerAncillaryDependencies")
    def test_validate_and_initialize_returns_empty_list_when_missing_ancillary_dependencies(self,
                                                                                            mock_glows_initializer_ancillary_dependencies: Mock):
        test_cases = [
            (None, Mock(), Mock(), Mock(), Mock()),
            (Mock(), None, Mock(), Mock(), Mock()),
            (Mock(), Mock(), None, Mock(), Mock()),
            (Mock(), Mock(), Mock(), None, Mock()),
            (Mock(), Mock(), Mock(), Mock(), None),
            (None, None, None, None, None),
        ]

        for f107_path, lyman_alpha_path, omni_data_path, uv_anisotropy, waw_helioion_mp in test_cases:
            with self.subTest(
                    f"f107 {f107_path}, lyman_alpha_path {lyman_alpha_path}, omni_data_path {omni_data_path}"):
                mock_glows_initializer_ancillary_dependencies.fetch_dependencies.return_value = GlowsInitializerAncillaryDependencies(
                    uv_anisotropy, waw_helioion_mp, f107_path, lyman_alpha_path, omni_data_path)

                self.assertEqual([], GlowsInitializer.validate_and_initialize(""))

    @patch("imap_l3_processing.glows.glows_initializer.GlowsInitializerAncillaryDependencies")
    @patch("imap_l3_processing.glows.glows_initializer.query")
    @patch("imap_l3_processing.glows.glows_initializer.find_unprocessed_carrington_rotations")
    def test_validate_and_initialize_l3b(self, mock_find_unprocessed_carrington_rotations: Mock, mock_query: Mock,
                                         mock_glows_initializer_ancillary_dependencies: Mock):
        version = "v003"
        mock_l3a = Mock()
        mock_l3b = Mock()

        mock_query.side_effect = [
            mock_l3a,
            mock_l3b
        ]

        ancillary_dependencies = GlowsInitializerAncillaryDependencies(Mock(), Mock(), Mock(), Mock(), Mock())

        mock_glows_initializer_ancillary_dependencies.fetch_dependencies.return_value = ancillary_dependencies

        GlowsInitializer.validate_and_initialize(version)

        self.assertEqual(2, mock_query.call_count)
        mock_query.assert_has_calls([call(instrument="glows", version=version, data_level="l3a"),
                                     call(instrument="glows", version=version, data_level="l3b")])

        mock_find_unprocessed_carrington_rotations.assert_called_once_with(mock_l3a, mock_l3b)

        mock_glows_initializer_ancillary_dependencies.fetch_dependencies.assert_called_once()
