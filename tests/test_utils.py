import os
import tempfile
from datetime import datetime, date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call, Mock, sentinel
from urllib.error import URLError

import numpy as np
from imap_processing.ena_maps.utils.spatial_utils import build_solid_angle_map
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH, FIVE_MINUTES_IN_NANOSECONDS, ONE_SECOND_IN_NANOSECONDS, \
    SECONDS_PER_DAY
from imap_l3_processing.hi.l3.models import HiL1cData, HiGlowsL3eData
from imap_l3_processing.models import UpstreamDataDependency, RectangularIntensityMapData
from imap_l3_processing.swapi.l3a.models import SwapiL3AlphaSolarWindData
from imap_l3_processing.utils import format_time, download_dependency, read_l1d_mag_data, save_data, \
    find_glows_l3e_dependencies, \
    download_external_dependency, download_dependency_from_path, download_dependency_with_repointing, \
    combine_glows_l3e_with_l1c_pointing, read_rectangular_intensity_map_data_from_cdf
from imap_l3_processing.version import VERSION
from tests.cdf.test_cdf_utils import TestDataProduct
from tests.test_helpers import get_test_data_folder


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
    def test_save_data(self, mock_write_cdf, mock_today, _):
        mock_today.today.return_value = date(2024, 9, 16)

        input_metadata = UpstreamDataDependency("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
                                                "descriptor")
        epoch = np.array([1, 2, 3])
        alpha_sw_speed = np.array([4, 5, 6])
        alpha_sw_density = np.array([5, 5, 5])
        alpha_sw_temperature = np.array([4, 3, 5])

        data_product = SwapiL3AlphaSolarWindData(input_metadata=input_metadata, epoch=epoch,
                                                 alpha_sw_speed=alpha_sw_speed,
                                                 alpha_sw_temperature=alpha_sw_temperature,
                                                 alpha_sw_density=alpha_sw_density,
                                                 parent_file_names=sentinel.parent_files)
        returned_file_path = save_data(data_product)

        mock_write_cdf.assert_called_once()
        actual_file_path = mock_write_cdf.call_args.args[0]
        actual_data = mock_write_cdf.call_args.args[1]
        actual_attribute_manager = mock_write_cdf.call_args.args[2]

        expected_file_path = str(TEMP_CDF_FOLDER_PATH / "imap_swapi_l2_descriptor_20240917_v2.cdf")
        self.assertEqual(expected_file_path, actual_file_path)
        self.assertIs(data_product, actual_data)

        actual_attribute_manager.add_global_attribute.assert_has_calls([
            call("Data_version", "2"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_swapi_l2_descriptor"),
            call("Logical_file_id", "imap_swapi_l2_descriptor_20240917_v2"),
            call("ground_software_version", VERSION),
            call("Parents", sentinel.parent_files),
        ])

        actual_attribute_manager.add_instrument_attrs.assert_called_with(
            "swapi", "l2", input_metadata.descriptor
        )

        self.assertEqual(expected_file_path, returned_file_path)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_adds_repointing_if_present(self, mock_write_cdf, mock_today, _):
        mock_today.today.return_value = date(2024, 9, 16)

        expected_repointing = 2
        data_product = TestDataProduct()
        data_product.input_metadata.repointing = expected_repointing
        returned_file_path = save_data(data_product)

        actual_file_path = mock_write_cdf.call_args.args[0]

        actual_attribute_manager = mock_write_cdf.call_args.args[2]

        expected_file_path = str(
            TEMP_CDF_FOLDER_PATH / f"imap_instrument_data-level_descriptor_20250510-repoint0000{expected_repointing}_v003.cdf")
        self.assertEqual(expected_file_path, actual_file_path)

        actual_attribute_manager.add_global_attribute.assert_has_calls([
            call("Data_version", "003"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_instrument_data-level_descriptor"),
            call("Logical_file_id",
                 f"imap_instrument_data-level_descriptor_20250510-repoint0000{expected_repointing}_v003")
        ])

        self.assertEqual(expected_file_path, returned_file_path)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_does_not_add_parent_attribute_if_empty(self, mock_write_cdf, mock_today, _):
        mock_today.today.return_value = date(2024, 9, 16)

        input_metadata = UpstreamDataDependency("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
                                                "descriptor")
        epoch = np.array([1, 2, 3])
        alpha_sw_speed = np.array([4, 5, 6])
        alpha_sw_density = np.array([5, 5, 5])
        alpha_sw_temperature = np.array([4, 3, 5])

        data_product = SwapiL3AlphaSolarWindData(input_metadata=input_metadata, epoch=epoch,
                                                 alpha_sw_speed=alpha_sw_speed,
                                                 alpha_sw_temperature=alpha_sw_temperature,
                                                 alpha_sw_density=alpha_sw_density)
        returned_file_path = save_data(data_product)

        mock_write_cdf.assert_called_once()
        actual_attribute_manager = mock_write_cdf.call_args.args[2]

        self.assertEqual([
            call("Data_version", "2"),
            call("Generation_date", "20240916"),
            call("Logical_source", "imap_swapi_l2_descriptor"),
            call("Logical_file_id", "imap_swapi_l2_descriptor_20240917_v2"),
            call("ground_software_version", VERSION)
        ], actual_attribute_manager.add_global_attribute.call_args_list)

    @patch("imap_l3_processing.utils.ImapAttributeManager")
    @patch("imap_l3_processing.utils.date")
    @patch("imap_l3_processing.utils.write_cdf")
    def test_save_data_custom_path(self, mock_write_cdf, mock_today, _):
        mock_today.today.return_value = date(2024, 9, 16)

        input_metadata = UpstreamDataDependency("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
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

        expected_file_path = str(custom_path / "imap_swapi_l2_descriptor_20240917_v2.cdf")
        self.assertEqual(expected_file_path, actual_file_path)

        self.assertEqual(expected_file_path, returned_file_path)

    def test_format_time(self):
        time = datetime(2024, 7, 9)
        actual_time = format_time(time)
        self.assertEqual("20240709", actual_time)

        actual_time = format_time(None)
        self.assertEqual(None, actual_time)

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency(self, mock_data_access):
        dependency = UpstreamDataDependency("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
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
        dependency = UpstreamDataDependency("glows", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
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
        dependency = UpstreamDataDependency("glows", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v002",
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

    @patch("imap_l3_processing.utils.urlretrieve")
    def test_download_external_dependency(self, mock_urlretrieve):
        expected_url = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
        expected_filename = "f107_fluxtable.txt"
        mock_urlretrieve.return_value = (expected_filename, Mock())
        saved_path = download_external_dependency(expected_url, expected_filename)

        mock_urlretrieve.assert_called_once_with(expected_url, expected_filename)
        self.assertEqual(Path(expected_filename), saved_path)

    @patch("imap_l3_processing.utils.urlretrieve")
    def test_download_external_dependency_error_case(self, mock_urlretrieve):
        expected_url = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/no_such_file.txt"
        expected_filename = "f107_fluxtable.txt"
        mock_urlretrieve.side_effect = URLError("server is down")
        returned = download_external_dependency(expected_url, expected_filename)
        self.assertIsNone(returned)

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency_throws_value_error_if_not_one_file_returned(self, mock_data_access):
        dependency = UpstreamDataDependency("swapi", "l2", datetime(2024, 9, 17), datetime(2024, 9, 18), "v2",
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

    @patch('imap_l3_processing.utils.imap_data_access')
    def test_download_dependency_from_path(self, mock_data_access):
        local_path = download_dependency_from_path(sentinel.sdc_path)

        self.assertEqual(mock_data_access.download.return_value, local_path)
        mock_data_access.download.assert_called_once_with(sentinel.sdc_path)

    @patch('imap_l3_processing.utils.imap_data_access.query')
    def test_find_glows_l3e_dependencies(self, mock_data_access_query):
        l1c_90sensor_file_paths = ["imap_hi_l1c_90sensor-pset_20201001_v001.cdf",
                                   "imap_hi_l1c_90sensor-pset_20201002_v002.cdf",
                                   "imap_hi_l1c_90sensor-pset_20201003_v001.cdf"]
        l1c_45sensor_file_paths = ["imap_hi_l1c_45sensor-pset_20210509_v001.cdf",
                                   "imap_hi_l1c_45sensor-pset_20210508_v002.cdf",
                                   "imap_hi_l1c_45sensor-pset_20210507_v001.cdf"]

        test_cases = [
            (l1c_90sensor_file_paths, "90", "20201001", "20201003", "hi"),
            (l1c_45sensor_file_paths, "45", "20210507", "20210509", "hi"),
            (l1c_90sensor_file_paths, "90", "20201001", "20201003", "ultra"),
            (l1c_45sensor_file_paths, "45", "20210507", "20210509", "ultra"),
        ]

        mock_data_access_query.return_value = [{"file_path": "glows_1"},
                                               {"file_path": "glows_2"},
                                               {"file_path": "glows_3"}]

        for l1c_file_paths, sensor, expected_start_date, expected_end_date, instrument in test_cases:
            with self.subTest(f"sensor: {sensor}"):
                glows_file_paths = find_glows_l3e_dependencies(l1c_file_paths, instrument)

                mock_data_access_query.assert_called_with(instrument="glows",
                                                          data_level="l3e",
                                                          descriptor=f"survival-probabilities-{instrument}-{sensor}",
                                                          start_date=expected_start_date,
                                                          end_date=expected_end_date,
                                                          version="latest")

                self.assertEqual(["glows_1", "glows_2", "glows_3"],
                                 glows_file_paths)

    def test_combine_glows_l3e_with_l1c_pointing(self):
        glows_l3e_data = [
            HiGlowsL3eData(epoch=datetime.fromisoformat("2023-01-01T00:00:00Z"), spin_angle=None,
                           energy=None, probability_of_survival=None),
            HiGlowsL3eData(epoch=datetime.fromisoformat("2023-01-02T00:00:00Z"), spin_angle=None,
                           energy=None, probability_of_survival=None),
            HiGlowsL3eData(epoch=datetime.fromisoformat("2023-01-03T00:00:00Z"), spin_angle=None,
                           energy=None, probability_of_survival=None),
            HiGlowsL3eData(epoch=datetime.fromisoformat("2023-01-05T00:00:00Z"), spin_angle=None,
                           energy=None, probability_of_survival=None),
        ]

        hi_l1c_data = [
            HiL1cData(epoch=datetime.fromisoformat("2023-01-02T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-04T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-05T00:00:00Z"), epoch_j2000=None, exposure_times=None,
                      esa_energy_step=None),
            HiL1cData(epoch=datetime.fromisoformat("2023-01-06T00:00:00Z"), epoch_j2000=None, exposure_times=None,
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

    def test_read_intensity_map_with_rectangular_cords_data_from_cdf(self):

        rng = np.random.default_rng()
        with tempfile.TemporaryDirectory() as temp_dir:
            pathname = os.path.join(temp_dir, "test_cdf")
            with CDF(pathname, '') as cdf:
                cdf.col_major(True)

                ena_intensity = rng.random((1, 9, 90, 45))
                energy = rng.random(9)
                energy_delta_plus = rng.random(9)
                energy_delta_minus = rng.random(9)
                energy_label = energy.astype(str)
                ena_intensity_stat_unc = rng.random(ena_intensity.shape)
                ena_intensity_sys_err = rng.random(ena_intensity.shape)

                epoch = np.array([datetime.now()])
                epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
                exposure = np.full(ena_intensity.shape[:-1], 1.0)
                lat = np.arange(-88.0, 92.0, 4.0)
                lat_delta = np.full(lat.shape, 2.0)
                lat_label = [f"{x} deg" for x in lat]
                lon = np.arange(0.0, 360.0, 4.0)
                lon_delta = np.full(lon.shape, 2.0)
                lon_label = [f"{x} deg" for x in lon]

                obs_date = np.full(ena_intensity.shape, datetime.now())
                obs_date_range = np.full(ena_intensity.shape, ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY * 2)
                solid_angle = build_solid_angle_map(4)
                solid_angle = solid_angle[np.newaxis, ...]

                cdf.new("epoch", epoch)
                cdf.new("energy", energy, recVary=False)
                cdf.new("latitude", lat, recVary=False)
                cdf.new("latitude_delta", lat_delta, recVary=False)
                cdf.new("latitude_label", lat_label, recVary=False)
                cdf.new("longitude", lon, recVary=False)
                cdf.new("longitude_delta", lon_delta, recVary=False)
                cdf.new("longitude_label", lon_label, recVary=False)
                cdf.new("ena_intensity", ena_intensity, recVary=True)
                cdf.new("ena_intensity_stat_unc", ena_intensity_stat_unc, recVary=True)
                cdf.new("ena_intensity_sys_err", ena_intensity_sys_err, recVary=True)
                cdf.new("exposure_factor", exposure, recVary=True)
                cdf.new("obs_date", obs_date, recVary=True)
                cdf.new("obs_date_range", obs_date_range, recVary=True)
                cdf.new("solid_angle", solid_angle, recVary=True)
                cdf.new("epoch_delta", epoch_delta, recVary=True)
                cdf.new("energy_delta_plus", energy_delta_plus, recVary=False)
                cdf.new("energy_delta_minus", energy_delta_minus, recVary=False)
                cdf.new("energy_label", energy_label, recVary=False)

                for var in cdf:
                    cdf[var].attrs['FILLVAL'] = 1000000

            for path in [pathname, Path(pathname)]:
                with self.subTest(path=path):
                    result = read_rectangular_intensity_map_data_from_cdf(path)
                    self.assertIsInstance(result, RectangularIntensityMapData)

                    rectangular_coords = result.coords
                    map_data = result.intensity_map_data

                    np.testing.assert_array_equal(epoch, map_data.epoch)
                    np.testing.assert_array_equal(epoch_delta, map_data.epoch_delta)
                    np.testing.assert_array_equal(energy, map_data.energy)
                    np.testing.assert_array_equal(energy_delta_plus, map_data.energy_delta_plus)
                    np.testing.assert_array_equal(energy_delta_minus, map_data.energy_delta_minus)
                    np.testing.assert_array_equal(energy_label, map_data.energy_label)
                    np.testing.assert_array_equal(lat, map_data.latitude)
                    np.testing.assert_array_equal(lat_delta, rectangular_coords.latitude_delta)
                    np.testing.assert_array_equal(lat_label, rectangular_coords.latitude_label)
                    np.testing.assert_array_equal(lon, map_data.longitude)
                    np.testing.assert_array_equal(lon_delta, rectangular_coords.longitude_delta)
                    np.testing.assert_array_equal(lon_label, rectangular_coords.longitude_label)
                    np.testing.assert_array_equal(ena_intensity, map_data.ena_intensity)
                    np.testing.assert_array_equal(ena_intensity_stat_unc, map_data.ena_intensity_stat_unc)
                    np.testing.assert_array_equal(ena_intensity_sys_err, map_data.ena_intensity_sys_err)
                    np.testing.assert_array_equal(exposure, map_data.exposure_factor)
                    np.testing.assert_array_equal(obs_date, map_data.obs_date)
                    np.testing.assert_array_equal(obs_date_range, map_data.obs_date_range)
                    np.testing.assert_array_equal(solid_angle, map_data.solid_angle)

    def test_fill_values_in_read_rectangular_intensity_map_data_from_cdf(self):
        path = get_test_data_folder() / 'hi' / 'fake_l2_maps' / 'l2_map_with_fill_values.cdf'
        result = read_rectangular_intensity_map_data_from_cdf(path)
        map_data = result.intensity_map_data
        coords = result.coords

        with CDF(str(path)) as cdf:
            np.testing.assert_array_equal(map_data.epoch, cdf["epoch"], )

            self.assertTrue(np.all(map_data.epoch_delta.mask))
            self.assertTrue(np.all(map_data.obs_date.mask))
            self.assertTrue(np.all(map_data.obs_date_range.mask))

            np.testing.assert_array_equal(map_data.energy, np.full_like(cdf["energy"], np.nan))
            np.testing.assert_array_equal(map_data.energy_delta_plus, np.full_like(cdf["energy_delta_plus"], np.nan))
            np.testing.assert_array_equal(map_data.energy_delta_minus, np.full_like(cdf["energy_delta_minus"], np.nan))
            np.testing.assert_array_equal(map_data.latitude, np.full_like(cdf["latitude"], np.nan))
            np.testing.assert_array_equal(coords.latitude_delta, np.full_like(cdf["latitude_delta"], np.nan))
            np.testing.assert_array_equal(map_data.longitude, np.full_like(cdf["longitude"], np.nan))
            np.testing.assert_array_equal(coords.longitude_delta, np.full_like(cdf["longitude_delta"], np.nan))
            np.testing.assert_array_equal(map_data.ena_intensity, np.full_like(cdf["ena_intensity"], np.nan))
            np.testing.assert_array_equal(map_data.ena_intensity_stat_unc,
                                          np.full_like(cdf["ena_intensity_stat_unc"], np.nan))
            np.testing.assert_array_equal(map_data.ena_intensity_sys_err,
                                          np.full_like(cdf["ena_intensity_sys_err"], np.nan))
            np.testing.assert_array_equal(map_data.exposure_factor, np.full_like(cdf["exposure_factor"], np.nan))
            np.testing.assert_array_equal(map_data.solid_angle, np.full_like(cdf["solid_angle"], np.nan))
