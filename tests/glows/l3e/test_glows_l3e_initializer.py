import unittest
from unittest.mock import patch, sentinel, call

from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializer


class TestGlowsL3EInitializer(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        return

    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.get_most_recently_uploaded_ancillary')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_initializer.imap_data_access.query')
    def test_determine_l3e_files_to_produce(self, mock_query, mock_get_most_recently_uploaded_ancillary):
        mock_query.side_effect = [
            sentinel.ionization_files,
            sentinel.pipeline_settings_l3bcde,
            sentinel.energy_grid_lo,
            sentinel.tess_xyz_8,
            sentinel.elongation_data,
            sentinel.energy_grid_hi,
            sentinel.energy_grid_ultra,
            sentinel.tess_ang_16
        ]

        GlowsL3EInitializer.determine_l3e_files_to_produce()

        mock_query.assert_has_calls([
            call(instrument='glows', descriptor='ionization-files'),
            call(instrument='glows', descriptor='pipeline-settings-l3bcde'),
            call(instrument='glows', descriptor='energy-grid-lo'),
            call(instrument='glows', descriptor='tess-xyz-8'),
            call(instrument='lo', descriptor='elongation-data'),
            call(instrument='glows', descriptor='energy-grid-hi'),
            call(instrument='glows', descriptor='energy-grid-ultra'),
            call(instrument='glows', descriptor='tess-ang-16'),
        ])

        mock_get_most_recently_uploaded_ancillary.assert_has_calls([
            call(sentinel.ionization_files),
            call(sentinel.pipeline_settings_l3bcde),
            call(sentinel.energy_grid_lo),
            call(sentinel.tess_xyz_8),
            call(sentinel.elongation_data),
            call(sentinel.energy_grid_hi),
            call(sentinel.energy_grid_ultra),
            call(sentinel.tess_ang_16)
        ])


