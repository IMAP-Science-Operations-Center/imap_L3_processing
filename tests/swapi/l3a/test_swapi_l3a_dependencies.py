import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, AncillaryInput

from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, \
    PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR, \
    INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR, DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, \
    CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR, EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies


class TestSwapiL3ADependencies(unittest.TestCase):

    @patch(
        "imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.SwapiL3ADependencies.from_file_paths")
    @patch(
        "imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_from_file_paths):
        input_collection = ProcessingInputCollection()

        start_date = '20100105'
        mission = 'imap'
        instrument = 'swapi'
        data_level = 'l2'
        version = 'v010'

        mock_download.side_effect = [
            sentinel.swapi_l2_data,
            sentinel.proton_density_and_temperature_calibration_file,
            sentinel.alpha_density_and_temperature_calibration_file,
            sentinel.clock_and_deflection_file,
            sentinel.efficiency_file,
            sentinel.geometric_factor_calibration_table,
            sentinel.instrument_response_table,
            sentinel.neutral_helium_table,
        ]

        swapi_science_file_download_path = f"{mission}_{instrument}_{data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_clock_angle_calibration_table_file_name = f"{mission}_{instrument}_{CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_alpha_temp_density_calibration_file_name = f"{mission}_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_proton_temp_and_density_calibration_file_name = f"{mission}_{instrument}_{PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_efficiency_file_name = f"{mission}_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_geometric_factor_calibration_file_name = f"{mission}_{instrument}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_instrument_response_lookup_table_collection = f"{mission}_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf"
        swapi_density_of_neutral_helium_lookup = f"{mission}_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{start_date}_{version}.cdf"

        science_input = ScienceInput(swapi_science_file_download_path)
        clock_angle_calibration_ancillary = AncillaryInput(swapi_clock_angle_calibration_table_file_name)
        alpha_temp_density_calibration_ancillary = AncillaryInput(swapi_alpha_temp_density_calibration_file_name)
        proton_temp_and_density_calibration_ancillary = AncillaryInput(swapi_proton_temp_and_density_calibration_file_name)
        efficiency_ancillary = AncillaryInput(swapi_efficiency_file_name)
        geometric_factor_calibration_ancillary = AncillaryInput(swapi_geometric_factor_calibration_file_name)
        instrument_response_lookup_ancillary = AncillaryInput(swapi_instrument_response_lookup_table_collection)
        density_of_neutral_helium_ancillary = AncillaryInput(swapi_density_of_neutral_helium_lookup)

        input_collection.add(
            [science_input, clock_angle_calibration_ancillary, alpha_temp_density_calibration_ancillary,
             proton_temp_and_density_calibration_ancillary, efficiency_ancillary, geometric_factor_calibration_ancillary,
             instrument_response_lookup_ancillary, density_of_neutral_helium_ancillary])

        actual_swapi_l3_dependencies = SwapiL3ADependencies.fetch_dependencies(input_collection)

        science_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'swapi' / 'l2' / '2010' / '01'
        ancillary_data_dir = imap_data_access.config["DATA_DIR"] / 'imap' / 'ancillary' / 'swapi'

        expected_download_science_path = science_data_dir / swapi_science_file_download_path
        expected_download_ancillary_path1 = ancillary_data_dir / swapi_proton_temp_and_density_calibration_file_name
        expected_download_ancillary_path2 = ancillary_data_dir / swapi_alpha_temp_density_calibration_file_name
        expected_download_ancillary_path3 = ancillary_data_dir / swapi_clock_angle_calibration_table_file_name
        expected_download_ancillary_path4 = ancillary_data_dir / swapi_efficiency_file_name
        expected_download_ancillary_path5 = ancillary_data_dir / swapi_geometric_factor_calibration_file_name
        expected_download_ancillary_path6 = ancillary_data_dir / swapi_instrument_response_lookup_table_collection
        expected_download_ancillary_path7 = ancillary_data_dir / swapi_density_of_neutral_helium_lookup

        mock_download.assert_has_calls([
            call(expected_download_science_path),
            call(expected_download_ancillary_path1),
            call(expected_download_ancillary_path2),
            call(expected_download_ancillary_path3),
            call(expected_download_ancillary_path4),
            call(expected_download_ancillary_path5),
            call(expected_download_ancillary_path6),
            call(expected_download_ancillary_path7),
        ])

        mock_from_file_paths.assert_called_with(
            sentinel.swapi_l2_data,
            sentinel.proton_density_and_temperature_calibration_file,
            sentinel.alpha_density_and_temperature_calibration_file,
            sentinel.clock_and_deflection_file,
            sentinel.efficiency_file,
            sentinel.geometric_factor_calibration_table,
            sentinel.instrument_response_table,
            sentinel.neutral_helium_table,
        )

        self.assertEqual(mock_from_file_paths.return_value, actual_swapi_l3_dependencies)

    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.CDF')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.ProtonTemperatureAndDensityCalibrationTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.AlphaTemperatureDensityCalibrationTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.ClockAngleCalibrationTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.EfficiencyCalibrationTable')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.GeometricFactorCalibrationTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.InstrumentResponseLookupTableCollection.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.DensityOfNeutralHeliumLookupTable.from_file')
    @patch('imap_l3_processing.swapi.l3a.swapi_l3a_dependencies.read_l2_swapi_data')
    def test_from_file_paths(self, mock_read_l2_swapi, mock_neutral_helium_from_file,
                             mock_instrument_from_file, mock_geometric_from_file, mock_efficiency_lookup_class,
                             mock_clock_angle_from_file, mock_alpha_temp_from_file,
                             mock_proton_temp_from_file, mock_CDF):
        start_date = '20100105'
        mission = 'imap'
        instrument = 'swapi'
        data_level = 'l2'
        version = 'v010'

        swapi_science_file_download_path = Path(f"{mission}_{instrument}_{data_level}_{SWAPI_L2_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_clock_angle_calibration_table_file_name = Path(f"{mission}_{instrument}_{CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_alpha_temp_density_calibration_file_name = Path(f"{mission}_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_proton_temp_and_density_calibration_file_name = Path(f"{mission}_{instrument}_{PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_efficiency_calibration_file_name = Path(f"{mission}_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_geometric_factor_calibration_file_name = Path(f"{mission}_{instrument}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_instrument_response_lookup_table_collection = Path(f"{mission}_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{start_date}_{version}.cdf")
        swapi_density_of_neutral_helium_lookup = Path(f"{mission}_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{start_date}_{version}.cdf")

        mock_read_l2_swapi.return_value = sentinel.swapi_l2_data
        mock_proton_temp_from_file.return_value = sentinel.proton_temp_data
        mock_alpha_temp_from_file.return_value = sentinel.alpha_temp_data
        mock_clock_angle_from_file.return_value = sentinel.clock_angle_data
        mock_efficiency_lookup_class.return_value = sentinel.efficiency_lookup
        mock_geometric_from_file.return_value = sentinel.geometric_data
        mock_instrument_from_file.return_value = sentinel.instrument_data
        mock_neutral_helium_from_file.return_value = sentinel.neutral_helium_data

        expected_dependencies = SwapiL3ADependencies(sentinel.swapi_l2_data,
                                                     sentinel.proton_temp_data,
                                                     sentinel.alpha_temp_data,
                                                     sentinel.clock_angle_data,
                                                     sentinel.efficiency_lookup,
                                                     sentinel.geometric_data,
                                                     sentinel.instrument_data,
                                                     sentinel.neutral_helium_data)

        actual_dependencies = SwapiL3ADependencies.from_file_paths(
            swapi_science_file_download_path,
            swapi_proton_temp_and_density_calibration_file_name,
            swapi_alpha_temp_density_calibration_file_name,
            swapi_clock_angle_calibration_table_file_name,
            swapi_efficiency_calibration_file_name,
            swapi_geometric_factor_calibration_file_name,
            swapi_instrument_response_lookup_table_collection,
            swapi_density_of_neutral_helium_lookup,
        )

        mock_CDF.assert_called_once_with(str(swapi_science_file_download_path))

        mock_read_l2_swapi.assert_called_once_with(mock_CDF.return_value)
        mock_proton_temp_from_file.assert_called_once_with(swapi_proton_temp_and_density_calibration_file_name)
        mock_alpha_temp_from_file.assert_called_once_with(swapi_alpha_temp_density_calibration_file_name)
        mock_clock_angle_from_file.assert_called_once_with(swapi_clock_angle_calibration_table_file_name)
        mock_efficiency_lookup_class.assert_called_once_with(swapi_efficiency_calibration_file_name)
        mock_geometric_from_file.assert_called_once_with(swapi_geometric_factor_calibration_file_name)
        mock_instrument_from_file.assert_called_once_with(swapi_instrument_response_lookup_table_collection)
        mock_neutral_helium_from_file.assert_called_once_with(swapi_density_of_neutral_helium_lookup)

        self.assertEqual(expected_dependencies, actual_dependencies)
