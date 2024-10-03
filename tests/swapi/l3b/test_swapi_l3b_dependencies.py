import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, sentinel, call

import imap_processing
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR
from imap_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies


class TestSwapiL3BDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_imap_patcher = patch('imap_processing.utils.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()
        self.mock_imap_api.query.side_effect = [
            [{'file_path': sentinel.data_file_path}],
            [{'file_path': sentinel.energy_gf_file_path}],
            [{'file_path': sentinel.efficiency_calibration_table_file_path}],
        ]

    @patch('imap_processing.swapi.l3b.swapi_l3b_dependencies.CDF')
    @patch('imap_processing.swapi.l3b.swapi_l3b_dependencies.GeometricFactorCalibrationTable')
    @patch('imap_processing.swapi.l3b.swapi_l3b_dependencies.EfficiencyCalibrationTable')
    def test_fetch_dependencies(self, mock_efficiency_calibration_table_class, mock_geometric_factor_calibration_table,
                                mock_cdf_constructor):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = SWAPI_L2_DESCRIPTOR
        end_date = datetime.now()
        version = 'f'
        start_date = datetime.now() - timedelta(days=1)

        dependencies = UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                              version, descriptor)

        data_file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.download.side_effect = [
            data_file_path,
            sentinel.energy_gf_local_file_path,
            sentinel.efficiency_calibration_table_local_file_path
        ]

        fetched_dependencies = SwapiL3BDependencies.fetch_dependencies([dependencies])

        start_date_as_str = start_date.strftime("%Y%m%d")
        end_date_as_str = end_date.strftime("%Y%m%d")
        self.mock_imap_api.query.assert_has_calls([call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor=descriptor, start_date=start_date_as_str,
                                                        end_date=end_date_as_str, version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor="energy-gf-lut-not-cdf",
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   call(instrument=instrument, data_level=incoming_data_level,
                                                        descriptor="efficiency-lut-text-not-cdf",
                                                        start_date=None,
                                                        end_date=None,
                                                        version='latest'),
                                                   ])
        self.mock_imap_api.download.assert_has_calls(
            [call(sentinel.data_file_path), call(sentinel.energy_gf_file_path),
             call(sentinel.efficiency_calibration_table_file_path)])

        mock_cdf_constructor.assert_called_with(str(data_file_path))
        mock_geometric_factor_calibration_table.from_file.assert_called_with(
            sentinel.energy_gf_local_file_path)
        mock_efficiency_calibration_table_class.assert_called_with(
            sentinel.efficiency_calibration_table_local_file_path)

        self.assertIs(mock_cdf_constructor.return_value,
                      fetched_dependencies.data)
        self.assertIs(mock_geometric_factor_calibration_table.from_file.return_value,
                      fetched_dependencies.geometric_factor_calibration_table)
        self.assertIs(mock_efficiency_calibration_table_class.return_value,
                      fetched_dependencies.efficiency_calibration_table)

    def test_throws_exception_when_more_than_one_file_is_downloaded(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.download.return_value = file_path

        self.mock_imap_api.query.side_effect = [
            [{'file_path': '1 thing'}],
            [{'file_path': '2 thing'}, {'file_path': '3 thing'}]
        ]

        dependencies = [UpstreamDataDependency('swapi', 'l2', datetime.now() - timedelta(days=1),
                                               datetime.now(), 'f', SWAPI_L2_DESCRIPTOR)]

        with self.assertRaises(ValueError) as cm:
            SwapiL3BDependencies.fetch_dependencies(dependencies)
        exception = cm.exception
        self.assertEqual(f"Unexpected files found for SWAPI L3:"
                         f"{['2 thing', '3 thing']}. Expected one file to download, found 2.",
                         str(exception))

    def test_throws_exception_when_missing_swapi_data(self):
        dependencies = [
            UpstreamDataDependency('swapi', 'l2', datetime.now() - timedelta(days=1), datetime.now(), 'f', 'data')]

        with self.assertRaises(ValueError) as cm:
            SwapiL3BDependencies.fetch_dependencies(dependencies)
        exception = cm.exception
        self.assertEqual(f"Missing {SWAPI_L2_DESCRIPTOR} dependency.",
                         str(exception))
