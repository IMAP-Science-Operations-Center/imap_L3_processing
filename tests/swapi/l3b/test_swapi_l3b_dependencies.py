import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection

from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies


class TestSwapiL3BDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_imap_patcher = patch('imap_l3_processing.utils.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()

    @patch(
        "imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.SwapiL3BDependencies.from_file_paths")
    @patch(
        "imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_from_file_paths):
        incoming_data_level = 'l2'
        version = 'v002'
        start_date = datetime(2025, 1, 1).strftime("%Y%m%d")

        science_file_path = f'imap_swapi_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf'
        geometric_calibration_path = f'imap_swapi_{incoming_data_level}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf'
        efficiency_table_path = f'imap_swapi_{incoming_data_level}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf'

        science_input = ScienceInput(science_file_path)
        geometric_calibration_input = ScienceInput(geometric_calibration_path)
        efficiency_table_input = ScienceInput(efficiency_table_path)
        dependencies = ProcessingInputCollection(science_input, geometric_calibration_input, efficiency_table_input)

        actual_swapi_l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(dependencies)

        data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'swapi' / 'l2' / '2025' / '01'

        expected_download_science_path = data_dir / science_file_path
        expected_geometric_calibration_path = data_dir / geometric_calibration_path
        expected_efficiency_table_path = data_dir / efficiency_table_path

        mock_download.assert_has_calls([
            call(expected_download_science_path),
            call(expected_geometric_calibration_path),
            call(expected_efficiency_table_path),
        ])

        mock_from_file_paths.assert_called_with(
            expected_download_science_path,
            expected_geometric_calibration_path,
            expected_efficiency_table_path,
        )

        self.assertEqual(mock_from_file_paths.return_value, actual_swapi_l3b_dependencies)

    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.CDF')
    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.EfficiencyCalibrationTable')
    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.GeometricFactorCalibrationTable.from_file')
    @patch('imap_l3_processing.swapi.l3b.swapi_l3b_dependencies.read_l2_swapi_data')
    def test_from_file_paths(self, mock_read_l2_swapi, mock_read_geometric_factor, mock_efficiency_calibration,
                             mock_cdf):
        start_date = '20100105'
        mission = 'imap'
        instrument = 'swapi'
        data_level = 'l2'
        version = 'v010'

        swapi_science_file_download_path = f"{mission}_{instrument}_{data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_geometric_factor_calibration_file_name = f"{mission}_{instrument}_{data_level}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_efficiency_calibration_file_name = f"{mission}_{instrument}_{data_level}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"

        mock_read_l2_swapi.return_value = sentinel.swapi_l2_data
        mock_read_geometric_factor.return_value = sentinel.geometric_factor_data
        mock_efficiency_calibration.return_value = sentinel.efficiency_calibration_data

        expected_dependencies = SwapiL3BDependencies(sentinel.swapi_l2_data,
                                                     sentinel.geometric_factor_data,
                                                     sentinel.efficiency_calibration_data,
                                                     )

        actual_dependencies = SwapiL3BDependencies.from_file_paths(
            Path(swapi_science_file_download_path),
            Path(swapi_geometric_factor_calibration_file_name),
            Path(swapi_efficiency_calibration_file_name)
        )

        self.assertEqual(expected_dependencies, actual_dependencies)
