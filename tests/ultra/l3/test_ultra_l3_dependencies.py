import unittest
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
from imap_data_access import ScienceInput, AncillaryInput, ProcessingInputCollection

from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies, UltraL3SpectralIndexDependencies, \
    UltraL3CombinedDependencies
from tests.test_helpers import get_test_data_path


class TestUltraL3Dependencies(unittest.TestCase):
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.imap_data_access.download')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.ultra_l2')
    def test_fetch_dependencies(self, mock_ultra_l2, mock_download, mock_read_xarray, mock_read_ultra_l1c,
                                mock_read_glows):
        l1c_input_paths = ["imap_ultra_l1c_pset_20251010_v001.cdf", "imap_ultra_l1c_pset_20251011_v001.cdf",
                           "imap_ultra_l1c_pset_20251012_v001.cdf"]

        glows_file_paths = [
            "imap_glows_l3e_survival-probability-ultra-45_20201001_v001.cdf",
            "imap_glows_l3e_survival-probability-ultra-45_20201002_v002.cdf",
            "imap_glows_l3e_survival-probability-ultra-45_20201003_v003.cdf"]

        input_collection = Mock()
        input_collection.get_file_paths.side_effect = [
            [sentinel.l2_server_path],
            l1c_input_paths,
            glows_file_paths
        ]

        returned_download_paths = [sentinel.l2_map_path, sentinel.l1c_path_1, sentinel.l1c_path_2, sentinel.l1c_path_3,
                                   sentinel.glows_path_1, sentinel.glows_path_2, sentinel.glows_path_3]

        mock_download.side_effect = returned_download_paths
        mock_read_xarray.return_value = sentinel.ultra_l2_data

        l1c_data = [sentinel.ultra_l1c_data_1, sentinel.ultra_l1c_data_2,
                    sentinel.ultra_l1c_data_3]
        mock_read_ultra_l1c.side_effect = l1c_data

        glows_l3e_data = [sentinel.glows_data_1, sentinel.glows_data_2, sentinel.glows_data_3]
        mock_read_glows.side_effect = glows_l3e_data
        mock_ultra_l2.return_value = [sentinel.ultra_l2_healpix_map]

        dependencies = UltraL3Dependencies.fetch_dependencies(input_collection)

        input_collection.get_file_paths.assert_has_calls([
            call("ultra", data_type="l2"),
            call("ultra", data_type="l1c"),
            call("glows")
        ])
        expected_data_dictionary = {"l1c_path_1": sentinel.l1c_path_1,
                                    "l1c_path_2": sentinel.l1c_path_2,
                                    "l1c_path_3": sentinel.l1c_path_3}
        mock_ultra_l2.assert_has_calls([call(expected_data_dictionary)])

        expected_parent_file_paths = [sentinel.l2_server_path, *l1c_input_paths, *glows_file_paths]
        mock_download.assert_has_calls([call(file_path) for file_path in
                                        expected_parent_file_paths])
        mock_read_xarray.assert_called_once_with(sentinel.ultra_l2_healpix_map)

        self.assertEqual(l1c_data, dependencies.ultra_l1c_pset)
        self.assertEqual(glows_l3e_data, dependencies.glows_l3e_sp)
        self.assertEqual(sentinel.ultra_l2_data, dependencies.ultra_l2_map)
        self.assertEqual(returned_download_paths, dependencies.dependency_file_paths)

    def test_raise_error_for_more_than_one_input_files_paths(self):
        input_collection = Mock()
        input_collection.get_file_paths.return_value = [sentinel.imap_l2_map_path_1, sentinel.imap_l2_map_path_2]

        with self.assertRaises(AssertionError) as e:
            UltraL3Dependencies.fetch_dependencies(input_collection)

        self.assertEqual("Incorrect number of map dependencies: 2", str(e.exception))

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.ultra_l2')
    def test_from_file_paths(self, mock_ultra_l2, mock_read_l1c: Mock, mock_read_glows: Mock, mock_read_l2: Mock):
        ultra_l2_input_path = Path("um_path")
        ultra_l1c_input_paths = [Path("u_path_1"), Path("u_path_2")]
        glows_input_paths = [Path("g_path_1"), Path("g_path_2")]
        mock_ultra_l2.return_value = [sentinel.ultra_l2_healpix_map]
        mock_read_l1c.side_effect = [sentinel.ultra_data1, sentinel.ultra_data2]
        mock_read_glows.side_effect = [sentinel.glows_data1, sentinel.glows_data2]
        mock_read_l2.return_value = sentinel.ultra_l2_data

        result = UltraL3Dependencies.from_file_paths(ultra_l2_input_path, ultra_l1c_input_paths, glows_input_paths)

        mock_read_l2.assert_called_with(sentinel.ultra_l2_healpix_map)
        mock_read_l1c.assert_has_calls([call(file_path) for file_path in ultra_l1c_input_paths])
        mock_read_glows.assert_has_calls([call(file_path) for file_path in glows_input_paths])

        self.assertEqual(result.ultra_l1c_pset, [sentinel.ultra_data1, sentinel.ultra_data2])
        self.assertEqual(result.glows_l3e_sp, [sentinel.glows_data1, sentinel.glows_data2])
        self.assertEqual(result.ultra_l2_map, sentinel.ultra_l2_data)

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.imap_data_access.download')
    def test_spectral_index_fetch_dependencies(self, mock_download, mock_rectangular_read_from_path):
        map_file_name = 'imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf'
        ancillary_file_name = 'imap_ultra_spx-energy-ranges_20250601_v000.cdf'
        expected_energy_ranges = [[5, 15], [15, 50]]
        map_input = ScienceInput(map_file_name)
        ancillary_input = AncillaryInput(ancillary_file_name)
        processing_input_collection = ProcessingInputCollection(map_input, ancillary_input)

        mock_download.side_effect = [
            "map_file",
            get_test_data_path('ultra/imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat')
        ]

        ultra_l3_dependencies = UltraL3SpectralIndexDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            call(map_file_name),
            call(ancillary_file_name)
        ])
        mock_rectangular_read_from_path.assert_called_once_with("map_file")
        self.assertEqual(ultra_l3_dependencies.map_data, mock_rectangular_read_from_path.return_value)
        np.testing.assert_array_equal(ultra_l3_dependencies.fit_energy_ranges, expected_energy_ranges)

    def test_spectral_index_fetch_dependencies_raises_exception_on_missing_science_file(self):
        ancillary_input = AncillaryInput('imap_ultra_spx-energy-ranges_20250601_v000.dat')
        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(ancillary_input))
        self.assertEqual("Missing Ultra L3 file", str(context.exception))

    def test_spectral_index_fetch_dependencies_raises_exception_on_missing_ancillary_file(self):
        science_input = ScienceInput('imap_ultra_l3_ultra-cool-descriptor_20250601_v000.cdf')

        with self.assertRaises(ValueError) as context:
            UltraL3SpectralIndexDependencies.fetch_dependencies(ProcessingInputCollection(science_input))
        self.assertEqual("Missing fit energy ranges ancillary file", str(context.exception))

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.RectangularIntensityMapData.read_from_path')
    def test_spectral_index_from_file_paths(self, mock_read_from_path):
        map_file_path = Mock()
        ancillary_file_path = get_test_data_path('ultra') / 'imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat'
        expected_energy_range_values = np.loadtxt(ancillary_file_path)

        actual_dependencies = UltraL3SpectralIndexDependencies.from_file_paths(map_file_path, ancillary_file_path)
        self.assertEqual(mock_read_from_path.return_value, actual_dependencies.map_data)
        np.testing.assert_array_equal(actual_dependencies.fit_energy_ranges, expected_energy_range_values)

    def test_spectral_index_get_fit_energy_ranges(self):
        expected_energy_range_values = np.array([[5, 10], [15, 20]])

        dependencies = UltraL3SpectralIndexDependencies(map_data=Mock(),
                                                        fit_energy_ranges=expected_energy_range_values)
        actual_fit_energy_ranges = dependencies.get_fit_energy_ranges()

        np.testing.assert_array_equal(actual_fit_energy_ranges, expected_energy_range_values)

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.imap_data_access.download')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.ultra_l2')
    def test_combined_fetch_dependencies(self, mock_ultra_l2, mock_download, mock_read_from_xarray,
                                         mock_l1c_read_from_path):
        u45_map_file_name = 'imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf'
        u90_map_file_name = 'imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf'
        u45_pset_file_name = ["imap_ultra_l1c_45sensor-spacecraftpset_20251010_v001.cdf", "imap_ultra_l1c_45sensor-spacecraftpset_20251011_v001.cdf",
                           "imap_ultra_l1c_45sensor-spacecraftpset_20251012_v001.cdf"]
        u90_pset_file_name = ["imap_ultra_l1c_90sensor-spacecraftpset_20251010_v001.cdf", "imap_ultra_l1c_90sensor-spacecraftpset_20251011_v001.cdf",
                           "imap_ultra_l1c_90sensor-spacecraftpset_20251012_v001.cdf"]

        u45_map_input = ScienceInput(u45_map_file_name)
        u90_map_input = ScienceInput(u90_map_file_name)
        u45_pset_inputs = [ScienceInput(pset) for pset in u45_pset_file_name]
        u90_pset_inputs = [ScienceInput(pset) for pset in u90_pset_file_name]

        processing_input_collection = ProcessingInputCollection(u45_map_input, u90_map_input, *u45_pset_inputs, *u90_pset_inputs)

        expected_file_paths = [
            "imap_ultra_l1c_u45-pset_20251010_v001.cdf",
            "imap_ultra_l1c_u45-pset_20251011_v001.cdf",
            "imap_ultra_l1c_u45-pset_20251012_v001.cdf",
            "imap_ultra_l1c_u90-pset_20251010_v001.cdf",
            "imap_ultra_l1c_u90-pset_20251011_v001.cdf",
            "imap_ultra_l1c_u90-pset_20251012_v001.cdf",
            "imap_ultra_l2_u45-cool-descriptor_20250601_v000.cdf",
            "imap_ultra_l2_u90-cool-descriptor_20250601_v000.cdf",
        ]

        mock_download.side_effect = expected_file_paths

        mock_l1c_read_from_path.side_effect = [
            sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3,
            sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3,
        ]

        mock_ultra_l2.side_effect = [
            [sentinel.u45_l2_xarray],
            [sentinel.u90_l2_xarray]
        ]

        mock_read_from_xarray.side_effect = [
            sentinel.u45_healpix_dataset,
            sentinel.u90_healpix_dataset
        ]

        combined_dependencies = UltraL3CombinedDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            *[call(filename) for filename in u45_pset_file_name],
            *[call(filename) for filename in u90_pset_file_name],
            call(u45_map_file_name),
            call(u90_map_file_name)
        ])

        mock_l1c_read_from_path.assert_has_calls([
            call("imap_ultra_l1c_u45-pset_20251010_v001.cdf"),
            call("imap_ultra_l1c_u45-pset_20251011_v001.cdf"),
            call("imap_ultra_l1c_u45-pset_20251012_v001.cdf"),
            call("imap_ultra_l1c_u90-pset_20251010_v001.cdf"),
            call("imap_ultra_l1c_u90-pset_20251011_v001.cdf"),
            call("imap_ultra_l1c_u90-pset_20251012_v001.cdf"),
        ])


        expected_u45_l1c_dictionary = {"l1c_path_1": 'imap_ultra_l1c_u45-pset_20251010_v001.cdf',
                                    "l1c_path_2": "imap_ultra_l1c_u45-pset_20251011_v001.cdf",
                                    "l1c_path_3": "imap_ultra_l1c_u45-pset_20251012_v001.cdf",}

        expected_u90_l1c_dictionary = {"l1c_path_1": "imap_ultra_l1c_u90-pset_20251010_v001.cdf",
                                       "l1c_path_2": "imap_ultra_l1c_u90-pset_20251011_v001.cdf",
                                       "l1c_path_3": "imap_ultra_l1c_u90-pset_20251012_v001.cdf"}

        mock_ultra_l2.assert_has_calls([
            call(expected_u45_l1c_dictionary),
            call(expected_u90_l1c_dictionary),
        ])

        mock_read_from_xarray.assert_has_calls([
            call(sentinel.u45_l2_xarray),
            call(sentinel.u90_l2_xarray)
        ])

        self.assertEqual(combined_dependencies.u45_l2_map, sentinel.u45_healpix_dataset)
        self.assertEqual(combined_dependencies.u90_l2_map, sentinel.u90_healpix_dataset)
        self.assertEqual(combined_dependencies.u45_l1c_pset, [sentinel.u45_l1c_1, sentinel.u45_l1c_2, sentinel.u45_l1c_3])
        self.assertEqual(combined_dependencies.u90_l1c_pset, [sentinel.u90_l1c_1, sentinel.u90_l1c_2, sentinel.u90_l1c_3])
        self.assertEqual(combined_dependencies.dependency_file_paths, expected_file_paths)
