import unittest
from pathlib import Path
from unittest.mock import patch, call

import imap_data_access
from imap_data_access.processing_input import ScienceInput, AncillaryInput, ProcessingInputCollection

from imap_l3_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR, GLOWS_CALIBRATION_DATA_DESCRIPTOR, \
    GLOWS_PIPELINE_SETTINGS_DESCRIPTOR, GLOWS_EXTRA_HELIOSPHERIC_BACKGROUND_DESCRIPTOR, \
    GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies


class TestGlowsL3aDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.CDF')
    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.download')
    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.read_l2_glows_data')
    def test_fetch_dependencies(self, mock_read_l2_glows_data, mock_download, mock_cdf_constructor):
        mission = "imap"
        instrument = "glows"
        level = "l2a"
        start_date = "20230101"
        version = "v001"
        repointing = "repoint00002"

        ignored_dependency_filename = f'{mission}_{instrument}_{level}_ignored-data_{start_date}_{version}.cdf'
        cdf_dependency_filename = f'{mission}_{instrument}_{level}_{GLOWS_L2_DESCRIPTOR}_{start_date}-{repointing}_{version}.cdf'
        calibration_data_dependency_filename = f'{mission}_{instrument}_{GLOWS_CALIBRATION_DATA_DESCRIPTOR}_{start_date}_{version}.dat'
        settings_dependency_filename = f'{mission}_{instrument}_{GLOWS_PIPELINE_SETTINGS_DESCRIPTOR}_{start_date}_{version}.json'
        extra_heliospheric_background_dependency_filename = f'{mission}_{instrument}_{GLOWS_EXTRA_HELIOSPHERIC_BACKGROUND_DESCRIPTOR}_{start_date}_{version}.dat'
        time_dependent_background_dependency_filename = f'{mission}_{instrument}_{GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR}_{start_date}_{version}.dat'

        processing_input_collection = ProcessingInputCollection(
            ScienceInput(ignored_dependency_filename),
            ScienceInput(cdf_dependency_filename),
            AncillaryInput(calibration_data_dependency_filename),
            AncillaryInput(settings_dependency_filename),
            AncillaryInput(extra_heliospheric_background_dependency_filename),
            AncillaryInput(time_dependent_background_dependency_filename),
        )
        science_data_dir = imap_data_access.config['DATA_DIR'] / 'imap' / 'glows' / 'l2a' / '2023' / '01'
        ancillary_data_dir = imap_data_access.config['DATA_DIR'] / 'imap' / 'ancillary' / 'glows'

        cdf_path_str = "some_cdf.cdf"
        cdf_path_name = Path(cdf_path_str)
        calibration_data = Path("calibration.dat")
        settings = Path("settings.json")
        time_dependent_background_path = Path("time_dependent_background.dat")
        extra_heliospheric_background = Path("extra_heliospheric_background.dat")

        mock_download.side_effect = [
            cdf_path_name,
            calibration_data,
            extra_heliospheric_background,
            time_dependent_background_path,
            settings,
        ]

        result = GlowsL3ADependencies.fetch_dependencies(processing_input_collection)

        mock_cdf_constructor.assert_called_with(cdf_path_str)
        mock_read_l2_glows_data.assert_called_with(mock_cdf_constructor.return_value)
        self.assertIsInstance(result, GlowsL3ADependencies)
        self.assertEqual(mock_read_l2_glows_data.return_value, result.data)

        self.assertEqual(calibration_data, result.ancillary_files["calibration_data"])
        self.assertEqual(settings, result.ancillary_files["settings"])
        self.assertEqual(time_dependent_background_path,
                         result.ancillary_files["time_dependent_bckgrd"])
        self.assertEqual(extra_heliospheric_background,
                         result.ancillary_files["extra_heliospheric_bckgrd"])
        self.assertEqual(2, result.repointing)

        self.assertEqual([
            call(science_data_dir / cdf_dependency_filename),
            call(ancillary_data_dir / calibration_data_dependency_filename),
            call(ancillary_data_dir / extra_heliospheric_background_dependency_filename),
            call(ancillary_data_dir / time_dependent_background_dependency_filename),
            call(ancillary_data_dir / settings_dependency_filename),
        ], mock_download.call_args_list)
