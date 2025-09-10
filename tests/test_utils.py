import os
import tempfile
from datetime import datetime, date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call, Mock, sentinel
from urllib.error import URLError

import imap_data_access
import numpy as np
from imap_data_access import config
from requests import RequestException
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.l3a.models import SwapiL3AlphaSolarWindData
from imap_l3_processing.utils import format_time, download_dependency, read_l1d_mag_data, save_data, \
    find_glows_l3e_dependencies, download_external_dependency, download_dependency_with_repointing, \
    combine_glows_l3e_with_l1c_pointing, furnish_local_spice, get_spice_parent_file_names, furnish_spice_metakernel, \
    SpiceKernelTypes, FurnishMetakernelOutput
from imap_l3_processing.version import VERSION
from tests.cdf.test_cdf_utils import TestDataProduct
from tests.test_helpers import get_spice_data_path


class TestUtils(TestCase):
    def setUp(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    @patch("imap_l3_processing.utils.ScienceFilePath")
    def test_save_data(self, mock_science_file_path_class, mock_write_cdf, mock_today, mock_attribute_manager):
        mock_today.today.return_value = date(2024, 9, 16)

        input_metadata = InputMetadata("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                       "descriptor", repointing=None)
        epoch = np.array([1, 2, 3])
        alpha_sw_speed = np.array([4, 5, 6])
        alpha_sw_density = np.array([5, 5, 5])
        alpha_sw_temperature = np.array([4, 3, 5])

        data_product = SwapiL3AlphaSolarWindData(input_metadata=input_metadata, epoch=epoch,
                                                 alpha_sw_speed=alpha_sw_speed,
                                                 alpha_sw_temperature=alpha_sw_temperature,
                                                 alpha_sw_density=alpha_sw_density,
                                                 parent_file_names=sentinel.parent_files)

        mock_science_file_path = Mock()
        mock_science_file_path_class.generate_from_inputs.return_value = mock_science_file_path

        expected_file_path = mock_science_file_path.construct_path.return_value

        returned_file_path = save_data(data_product, cr_number=sentinel.cr)

        mock_science_file_path_class.generate_from_inputs.assert_called_once_with(
            instrument="swapi",
            data_level="l2",
            descriptor="descriptor",
            start_time="20240917",
            version="v002",
            repointing=None,
            cr=sentinel.cr
        )

        mock_science_file_path.construct_path.assert_called_once()

        mock_write_cdf.assert_called_once_with(str(expected_file_path), data_product,
                                               mock_attribute_manager.return_value)

        mock_attribute_manager.return_value.add_global_attribute.assert_has_calls([
            call("Data_version", "002"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_swapi_l2_descriptor"),
            call("Logical_file_id", expected_file_path.stem),
            call("ground_software_version", VERSION),
            call("Parents", sentinel.parent_files),
        ])

        mock_attribute_manager.return_value.add_instrument_attrs.assert_called_with(
            "swapi", "l2", input_metadata.descriptor
        )

        self.assertEqual(expected_file_path, returned_file_path)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_adds_repointing_if_present(self, mock_write_cdf, mock_today, mock_attribute_manager):
        mock_today.today.return_value = date(2024, 9, 16)

        expected_repointing = 2
        data_product = TestDataProduct()
        data_product.input_metadata.repointing = expected_repointing
        returned_file_path = save_data(data_product)

        mock_write_cdf.assert_called_once_with(str(returned_file_path), data_product,
                                               mock_attribute_manager.return_value)

        expected_file_path = (config["DATA_DIR"] / "imap" / "glows" / "l3" / "2025" / "05" /
                              f"imap_glows_l3_descriptor_20250510-repoint0000{expected_repointing}_v003.cdf")
        self.assertEqual(expected_file_path, returned_file_path)

        mock_attribute_manager.return_value.add_global_attribute.assert_has_calls([
            call("Data_version", "003"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_glows_l3_descriptor"),
            call("Logical_file_id",
                 f"imap_glows_l3_descriptor_20250510-repoint0000{expected_repointing}_v003")
        ])

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_adds_CR_if_present(self, mock_write_cdf, mock_today, mock_attribute_manager):
        mock_today.today.return_value = date(2024, 9, 16)

        expected_cr = 2
        data_product = TestDataProduct()
        data_product.input_metadata.repointing = None
        returned_file_path = save_data(data_product, cr_number=expected_cr)

        expected_file_path = (config["DATA_DIR"] / "imap" / "glows" / "l3" / "2025" / "05" /
                              f"imap_glows_l3_descriptor_20250510-cr0000{expected_cr}_v003.cdf")
        self.assertEqual(expected_file_path, returned_file_path)

        mock_attribute_manager.return_value.add_global_attribute.assert_has_calls([
            call("Data_version", "003"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_glows_l3_descriptor"),
            call("Logical_file_id",
                 f"imap_glows_l3_descriptor_20250510-cr0000{expected_cr}_v003")
        ])

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_throws_exception_if_given_cr_and_repointing(self, _, mock_today, __):
        mock_today.today.return_value = date(2024, 9, 16)

        cr = 2
        data_product = TestDataProduct()
        data_product.input_metadata.repointing = 3
        with self.assertRaises(AssertionError) as exception_manager:
            save_data(data_product, cr_number=cr)

        self.assertEqual(str(exception_manager.exception),
                         "You cannot call save_data with both a repointing in the metadata while passing in a CR number")

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_does_not_add_parent_attribute_if_empty(self, mock_write_cdf, mock_today, _):
        mock_today.today.return_value = date(2024, 9, 16)

        input_metadata = InputMetadata("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                       "descriptor")
        epoch = np.array([1, 2, 3])
        alpha_sw_speed = np.array([4, 5, 6])
        alpha_sw_density = np.array([5, 5, 5])
        alpha_sw_temperature = np.array([4, 3, 5])

        data_product = SwapiL3AlphaSolarWindData(input_metadata=input_metadata, epoch=epoch,
                                                 alpha_sw_speed=alpha_sw_speed,
                                                 alpha_sw_temperature=alpha_sw_temperature,
                                                 alpha_sw_density=alpha_sw_density)
        save_data(data_product)

        mock_write_cdf.assert_called_once()
        actual_attribute_manager = mock_write_cdf.call_args.args[2]

        self.assertEqual([
            call("Data_version", "002"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_swapi_l2_descriptor"),
            call("Logical_file_id", "imap_swapi_l2_descriptor_20240917_v002"),
            call("ground_software_version", VERSION)
        ], actual_attribute_manager.add_global_attribute.call_args_list)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_custom_path(self, mock_write_cdf, mock_today, _):
        mock_today.today.return_value = date(2024, 9, 16)

        input_metadata = InputMetadata("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                       "descriptor")
        epoch = np.array([1, 2, 3])
        alpha_sw_speed = np.array([4, 5, 6])
        alpha_sw_density = np.array([5, 5, 5])
        alpha_sw_temperature = np.array([4, 3, 5])

        data_product = SwapiL3AlphaSolarWindData(input_metadata=input_metadata, epoch=epoch,
                                                 alpha_sw_speed=alpha_sw_speed,
                                                 alpha_sw_temperature=alpha_sw_temperature,
                                                 alpha_sw_density=alpha_sw_density)

        custom_path = TEMP_CDF_FOLDER_PATH / "fancy_path"
        returned_file_path = save_data(data_product, folder_path=custom_path)

        mock_write_cdf.assert_called_once()
        actual_file_path = mock_write_cdf.call_args.args[0]

        expected_file_path = custom_path / "imap_swapi_l2_descriptor_20240917_v002.cdf"
        self.assertEqual(str(expected_file_path), actual_file_path)

        self.assertEqual(expected_file_path, returned_file_path)

    @patch("imap_l3_processing.utils.ImapAttributeManager.add_instrument_attrs", autospec=True)
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_procedurally_generates_map_global_metadata_if_absent(self, mock_write_cdf,
                                                                            mock_add_instrument_attrs):
        non_map_input_metadata = InputMetadata("swapi", "l3", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                               "descriptor")

        def add_swapi_attrs_from_file(attr_manager, instrument, level, descriptor):
            for logical_source in ["imap_swapi_l3_descriptor", "imap_ultra_l3_ena-map-descriptor"]:
                attr_manager.add_global_attribute(logical_source, {
                    "Data_level": "Data_level from file",
                    "Data_type": "Data_type from file",
                    "Logical_source_description": "Logical_source_description from file"
                })

        mock_add_instrument_attrs.side_effect = add_swapi_attrs_from_file

        map_input_metadata = InputMetadata("hi", "l3b", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                           "spx-map-descriptor")

        map_input_metadata_with_existing_global_attrs = InputMetadata("ultra", "l3", datetime(2024, 9, 17),
                                                                      datetime(2024, 9, 18), "v002",
                                                                      "ena-map-descriptor")

        cases = [
            (non_map_input_metadata, [("Data_level", "Data_level from file"), ("Data_type", "Data_type from file"),
                                      ("Logical_source_description", "Logical_source_description from file")]),
            (map_input_metadata,
             [("Data_level", "3b"), ("Data_type", "L3b_spx-map-descriptor>Level-3b Spectral Fit Index Map"),
              ("Logical_source_description", "IMAP-hi Level-3b Spectral Fit Index Map")]),
            (map_input_metadata_with_existing_global_attrs,
             [("Data_level", "Data_level from file"), ("Data_type", "Data_type from file"),
              ("Logical_source_description", "Logical_source_description from file")]
             )
        ]
        for input_metadata, expected_global_metadata in cases:
            with self.subTest(name=input_metadata.logical_source):
                mock_write_cdf.reset_mock()

                save_data(Mock(input_metadata=input_metadata), )

                mock_write_cdf.assert_called_once()

                [_file_path, _data, attribute_manager] = mock_write_cdf.call_args_list[0].args

                global_metadata = attribute_manager.try_load_global_metadata(input_metadata.logical_source)
                self.assertIsNotNone(global_metadata)
                for metadata_field, metadata_value in expected_global_metadata:
                    self.assertEqual(metadata_value, global_metadata[metadata_field])

    def test_format_time(self):
        time = datetime(2024, 7, 9)
        actual_time = format_time(time)
        self.assertEqual("20240709", actual_time)

        actual_time = format_time(None)
        self.assertEqual(None, actual_time)

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency(self, mock_data_access):
        dependency = InputMetadata("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
                                   "descriptor")
        query_dictionary = [{'file_path': "imap_swapi_l2_descriptor-fake-menlo-444_20240917_v2.cdf",
                             'second_entry': '12345'}]
        mock_data_access.query.return_value = query_dictionary

        path = download_dependency(dependency)

        mock_data_access.query.assert_called_once_with(instrument=dependency.instrument,
                                                       data_level=dependency.data_level,
                                                       descriptor=dependency.descriptor,
                                                       start_date="20240917",
                                                       end_date="20240918",
                                                       version='v2')
        mock_data_access.download.assert_called_once_with("imap_swapi_l2_descriptor-fake-menlo-444_20240917_v2.cdf")

        self.assertIs(path, mock_data_access.download.return_value)

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency_with_repointing(self, mock_data_access):
        dependency = InputMetadata("glows", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                   "hist")
        query_dictionary = [{'file_path': "imap_glows_l2_hist_20240917-repoint00001_v002.cdf",
                             'repointing': 1,
                             'third_entry': '12345'}]
        mock_data_access.query.return_value = query_dictionary

        path, repointing = download_dependency_with_repointing(dependency)

        mock_data_access.query.assert_called_once_with(instrument=dependency.instrument,
                                                       data_level=dependency.data_level,
                                                       descriptor=dependency.descriptor,
                                                       start_date="20240917",
                                                       end_date="20240918",
                                                       version='v002')
        mock_data_access.download.assert_called_once_with("imap_glows_l2_hist_20240917-repoint00001_v002.cdf")
        self.assertEqual(path, mock_data_access.download.return_value)
        self.assertEqual(1, repointing)

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency_with_repointing_throws_if_no_files_or_more_than_one_found(self, mock_data_access):
        dependency = InputMetadata("glows", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
                                   "hist")

        for return_values in ([], [{'file_path': "a", 'repointing': ''}, {'file_path': "b", 'repointing': ''}]):
            with self.subTest(return_values):
                mock_data_access.query.return_value = return_values
                with self.assertRaises(Exception) as cm:
                    download_dependency_with_repointing(dependency)
                expected_files_to_download = [dict_entry['file_path'] for dict_entry in return_values]
                mock_data_access.download.assert_not_called()

                self.assertEqual(
                    f"{expected_files_to_download}. Expected one file to download, found {len(return_values)}.",
                    str(cm.exception))

    @patch("imap_l3_processing.utils.requests")
    @patch('builtins.open')
    def test_download_external_dependency(self, mock_open_file, mock_requests):
        expected_url = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
        expected_filename = "f107_fluxtable.txt"

        mock_response = Mock(status_code=200)
        mock_requests.get.return_value = mock_response

        mock_opened_file = Mock()
        mock_open_file.return_value.__enter__.return_value = mock_opened_file

        saved_path = download_external_dependency(expected_url, expected_filename)

        mock_open_file.assert_called_once_with(expected_filename, "wb")
        mock_requests.get.assert_called_once_with(expected_url)

        mock_opened_file.write.assert_called_once_with(mock_response.content)
        self.assertEqual(Path(expected_filename), saved_path)

    @patch("imap_l3_processing.utils.requests")
    def test_download_external_dependency_error_case(self, mock_requests):
        expected_url = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/no_such_file.txt"

        expected_filename = "f107_fluxtable.txt"
        mock_requests.side_effect = RequestException

        returned = download_external_dependency(expected_url, expected_filename)
        self.assertIsNone(returned)

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency_throws_value_error_if_not_one_file_returned(self, mock_data_access):
        dependency = InputMetadata("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
                                   "descriptor")
        query_dictionary_more_than_one_file = [{'file_path': "imap_swapi_l2_descriptor-fake-menlo-444_20240917_v2.cdf",
                                                'second_entry': '12345'}, {"file_path": "extra_value"}]
        query_dictionary_less_than_one_file = []

        cases = [("2", query_dictionary_more_than_one_file),
                 ("0", query_dictionary_less_than_one_file)]

        for case, query_dictionary in cases:
            with self.subTest(case):
                mock_data_access.query.return_value = query_dictionary
                expected_files_to_download = [dict_entry['file_path'] for dict_entry in query_dictionary]
                with self.assertRaises(Exception) as cm:
                    download_dependency(dependency)

                self.assertEqual(
                    f"{expected_files_to_download}. Expected one file to download, found {case}.",
                    str(cm.exception))

    def test_read_l1d_mag_data(self):
        file_name_as_str = "test_cdf.cdf"
        file_name_as_path = Path(file_name_as_str)

        epoch = np.array([datetime(2010, 1, 1, 0, 0, 46)])
        vectors_with_magnitudes = np.array([[0, 1, 2, 0], [255, 255, 255, 255], [6, 7, 8, 0]], dtype=np.float64)
        trimmed_vectors = np.array([[0, 1, 2], [np.nan, np.nan, np.nan], [6, 7, 8]])
        with CDF(file_name_as_str, "") as mag_cdf:
            mag_cdf["epoch"] = epoch
            mag_cdf["vectors"] = vectors_with_magnitudes
            mag_cdf["vectors"].attrs['FILLVAL'] = 255.0

        cases = [
            ("file name as str", file_name_as_str),
            ("file name as Path", file_name_as_path)
        ]
        for name, path in cases:
            with self.subTest(name):
                results = read_l1d_mag_data(path)

                np.testing.assert_array_equal(epoch, results.epoch)
                np.testing.assert_array_equal(trimmed_vectors, results.mag_data)

    @patch('imap_l3_processing.utils.imap_data_access.query')
    def test_find_glows_l3e_dependencies(self, mock_data_access_query):
        l1c_hi_90sensor_file_paths = ["imap_hi_l1c_90sensor-pset_20201001_v001.cdf",
                                      "imap_hi_l1c_90sensor-pset_20201002_v002.cdf",
                                      "imap_hi_l1c_90sensor-pset_20201003_v001.cdf"]
        l1c_hi_45sensor_file_paths = ["imap_hi_l1c_45sensor-pset_20210509_v001.cdf",
                                      "imap_hi_l1c_45sensor-pset_20210508_v002.cdf",
                                      "imap_hi_l1c_45sensor-pset_20210507_v001.cdf"]
        l1c_lo_file_paths = ["imap_lo_l1c_pset_20210509_v001.cdf",
                             "imap_lo_l1c_pset_20210508_v002.cdf",
                             "imap_lo_l1c_pset_20210507_v001.cdf"]
        l1c_ultra_file_paths = ["imap_ultra_l1c_pset_20201001_v001.cdf",
                                "imap_ultra_l1c_pset_20201002_v002.cdf",
                                "imap_ultra_l1c_pset_20201003_v001.cdf"]

        test_cases = [
            (l1c_hi_90sensor_file_paths, "survival-probability-hi-90", "20201001", "20201003", "hi"),
            (l1c_hi_45sensor_file_paths, "survival-probability-hi-45", "20210507", "20210509", "hi"),
            (l1c_ultra_file_paths, "survival-probability-ul", "20201001", "20201003", "ultra"),
            (l1c_lo_file_paths, "survival-probability-lo", "20210507", "20210509", "lo"),
        ]

        mock_data_access_query.return_value = [{"file_path": "glows_1"},
                                               {"file_path": "glows_2"},
                                               {"file_path": "glows_3"}]

        for l1c_file_paths, expected_descriptor, expected_start_date, expected_end_date, instrument in test_cases:
            with self.subTest(instrument=instrument, descriptor=expected_descriptor):
                glows_file_paths = find_glows_l3e_dependencies(l1c_file_paths, instrument)

                mock_data_access_query.assert_called_with(instrument="glows",
                                                          data_level="l3e",
                                                          descriptor=expected_descriptor,
                                                          start_date=expected_start_date,
                                                          end_date=expected_end_date,
                                                          version="latest")

                self.assertEqual(["glows_1", "glows_2", "glows_3"],
                                 glows_file_paths)

    def test_combine_glows_l3e_with_l1c_pointing(self):
        glows_l3e_data = [
            GlowsL3eRectangularMapInputData(epoch=datetime.fromisoformat("2023-01-01T00:00:00+00:00"), spin_angle=None,
                                            energy=None, probability_of_survival=None),
            GlowsL3eRectangularMapInputData(epoch=datetime.fromisoformat("2023-01-02T00:00:00+00:00"), spin_angle=None,
                                            energy=None, probability_of_survival=None),
            GlowsL3eRectangularMapInputData(epoch=datetime.fromisoformat("2023-01-03T00:00:00+00:00"), spin_angle=None,
                                            energy=None, probability_of_survival=None),
            GlowsL3eRectangularMapInputData(epoch=datetime.fromisoformat("2023-01-05T00:00:00+00:00"), spin_angle=None,
                                            energy=None, probability_of_survival=None),
        ]

        hi_l1c_data = [
            InputRectangularPointingSet(epoch=datetime.fromisoformat("2023-01-02T00:00:00+00:00"), epoch_j2000=None,
                                        exposure_times=None,
                                        esa_energy_step=None),
            InputRectangularPointingSet(epoch=datetime.fromisoformat("2023-01-04T00:00:00+00:00"), epoch_j2000=None,
                                        exposure_times=None,
                                        esa_energy_step=None),
            InputRectangularPointingSet(epoch=datetime.fromisoformat("2023-01-05T00:00:00+00:00"), epoch_j2000=None,
                                        exposure_times=None,
                                        esa_energy_step=None),
            InputRectangularPointingSet(epoch=datetime.fromisoformat("2023-01-06T00:00:00+00:00"), epoch_j2000=None,
                                        exposure_times=None,
                                        esa_energy_step=None),
        ]

        expected = [
            (hi_l1c_data[0], glows_l3e_data[1],),
            (hi_l1c_data[1], None,),
            (hi_l1c_data[2], glows_l3e_data[3],),
            (hi_l1c_data[3], None,),
        ]

        actual = combine_glows_l3e_with_l1c_pointing(glows_l3e_data, hi_l1c_data)

        self.assertEqual(expected, actual)

    @patch("imap_l3_processing.utils.spiceypy")
    def test_furnish_local_spice(self, mock_spiceypy):
        mock_spiceypy.kdata.side_effect = [
            ("/Users/harrison/Development/imap_L3_processing/spice_kernels/naif0012.tls", "TEXT", '', 0)
        ]
        mock_spiceypy.ktotal.return_value = 1

        furnish_local_spice()

        mock_spiceypy.ktotal.assert_called_once_with('ALL')
        mock_spiceypy.kdata.assert_called_once_with(0, 'ALL')

        self.assertEqual(10, mock_spiceypy.furnsh.call_count)
        mock_spiceypy.furnsh.assert_has_calls([
            call(str(get_spice_data_path("de440s.bsp"))),
            call(str(get_spice_data_path("imap_science_0001.tf"))),
            call(str(get_spice_data_path("imap_science_draft.tf"))),
            call(str(get_spice_data_path("imap_sclk_0000.tsc"))),
            call(str(get_spice_data_path("imap_sim_ck_2hr_2secsampling_with_nutation.bc"))),
            call(str(get_spice_data_path("imap_spk_demo.bsp"))),
            call(str(get_spice_data_path("imap_wkcp.tf"))),
            call(str(get_spice_data_path("pck00011.tpc"))),
            call(str(get_spice_data_path("sim_1yr_imap_attitude.bc"))),
            call(str(get_spice_data_path("sim_1yr_imap_pointing_frame.bc"))),
        ], any_order=True)

    @patch("imap_l3_processing.utils.spiceypy")
    def test_get_spice_parent_file_names(self, mock_spicepy):
        mock_spicepy.ktotal.return_value = 2
        mock_spicepy.kdata.side_effect = [
            [Path("some_spice_1.tf"), 'type', 'source', 'handle'],
            [Path("some_spice_2.kf"), 'type', 'source', 'handle'],
        ]

        actual_spice_parents = get_spice_parent_file_names()

        mock_spicepy.ktotal.assert_called_once_with('ALL')
        mock_spicepy.kdata.assert_has_calls([
            call(0, 'ALL'),
            call(1, 'ALL')
        ])

        self.assertEqual(["some_spice_1.tf", "some_spice_2.kf"], actual_spice_parents)

    @patch("imap_l3_processing.utils.imap_data_access.download")
    @patch("imap_l3_processing.utils.spiceypy")
    @patch("imap_l3_processing.utils.requests")
    def test_furnish_spice_metakernel(self, mock_requests, mock_spiceypy, mock_download):

        metakernel_bytes = b"\n\\begintext\n\nThis is the most up to date Metakernel as of ..."

        mock_requests.get.side_effect = [
            Mock(content=metakernel_bytes),
            Mock(text='["naif0012.tls", "imap_001.tf"]')
        ]

        mock_download.side_effect = [sentinel.naif_downloaded_path, sentinel.imap_downloaded_path]

        start_date = datetime(2010, 1, 1)
        end_date = datetime(2010, 3, 1)

        kernel_types = [
            SpiceKernelTypes.Leapseconds,
            SpiceKernelTypes.IMAPFrames,
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            mock_data_dir = tmp_dir / "some_data_directory"
            mock_data_access_url = "https://imap-mission.com"
            with patch.dict(imap_data_access.config, { "DATA_DIR": mock_data_dir, "DATA_ACCESS_URL": mock_data_access_url}):

                actual_output = furnish_spice_metakernel(start_date, end_date, kernel_types)

                expected_request_params = {
                    "file_types": ["leapseconds", "imap_frames"],
                    "start_time": "1262322000",
                    "end_time": "1267419600",
                    "spice_path": mock_data_dir / "spice",
                }

                mock_requests.get.assert_has_calls([
                    call("https://imap-mission.com/metakernel", params=expected_request_params),
                    call("https://imap-mission.com/metakernel", params={**expected_request_params, "list_files": "true"})
                ])

                mock_download.assert_has_calls([
                    call('naif0012.tls'),
                    call('imap_001.tf')
                ])

                expected_metakernel_path = mock_data_dir / "metakernel" / "metakernel.txt"
                self.assertEqual(metakernel_bytes, expected_metakernel_path.read_bytes())

                mock_spiceypy.furnsh.assert_called_once_with(str(expected_metakernel_path))

                expected_output = FurnishMetakernelOutput(
                    metakernel_path=expected_metakernel_path,
                    spice_kernel_paths=[sentinel.naif_downloaded_path, sentinel.imap_downloaded_path]
                )

                self.assertEqual(expected_output, actual_output)

