import json
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import patch, Mock, sentinel, call

import numpy as np

from imap_l3_processing.constants import CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.glows import l3d
from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf, create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import GlowsL3BIonizationRate
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path, get_test_data_folder, \
    assert_dataclass_fields


class TestGlowsProcessor(unittest.TestCase):

    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.imap_data_access.upload')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_processor_handles_l3a(self, mock_spiceypy, mock_upload, mock_save_data, mock_L3aData,
                                   mock_glows_dependencies_class, mock_glows_initializer):
        mock_spiceypy.ktotal.return_value = 0
        instrument = 'glows'
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        outgoing_data_level = "l3a"
        outgoing_version = 'v002'

        mock_fetched_dependencies = mock_glows_dependencies_class.fetch_dependencies.return_value
        mock_fetched_dependencies.ancillary_files = {"settings": get_test_instrument_team_data_path(
            "glows/imap_glows_pipeline-settings_20250707_v002.json")}
        mock_fetched_dependencies.repointing = 5
        l3a_json_path = get_test_data_folder() / "glows" / "imap_glows_l3a_20130908085214_orbX_modX_p_v00.json"
        with open(l3a_json_path) as f:
            mock_L3aData.return_value.data = json.load(f)
        mock_cdf_path = mock_save_data.return_value

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)

        mock_processing_input_collection = Mock()
        parent_file_path = Path("test/parent_path")
        expected_parent_file_names = ['parent_path']
        mock_processing_input_collection.get_file_paths.return_value = [parent_file_path]

        processor = GlowsProcessor(dependencies=mock_processing_input_collection, input_metadata=input_metadata)
        processor.process()

        mock_glows_initializer.assert_not_called()
        mock_glows_dependencies_class.fetch_dependencies.assert_called_with(mock_processing_input_collection)
        expected_data_to_save = create_glows_l3a_from_dictionary(
            mock_L3aData.return_value.data, replace(input_metadata, descriptor=GLOWS_L3A_DESCRIPTOR))
        expected_data_to_save.input_metadata.repointing = mock_fetched_dependencies.repointing
        expected_data_to_save.parent_file_names = expected_parent_file_names
        actual_data = mock_save_data.call_args.args[0]
        self.assertEqual(expected_parent_file_names, actual_data.parent_file_names)
        assert_dataclass_fields(expected_data_to_save, actual_data)
        mock_upload.assert_called_with(mock_cdf_path)

    @patch('imap_l3_processing.glows.glows_processor.create_glows_l3a_from_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    def test_process_l3a(self, l3a_data_constructor, create_glows_l3a_from_dictionary):
        input_metadata = InputMetadata('glows', "l3a", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)

        processor.add_spin_angle_delta = Mock()
        fetched_dependencies = Mock()
        result = processor.process_l3a(fetched_dependencies)

        self.assertIs(create_glows_l3a_from_dictionary.return_value, result)
        l3a_data_constructor.assert_called_once_with(fetched_dependencies.ancillary_files)
        l3a_data_constructor.return_value.process_l2_data_file.assert_called_once_with(fetched_dependencies.data)
        l3a_data_constructor.return_value.generate_l3a_data.assert_called_once_with(
            fetched_dependencies.ancillary_files)
        processor.add_spin_angle_delta.assert_called_with(l3a_data_constructor.return_value.data,
                                                          fetched_dependencies.ancillary_files)
        create_glows_l3a_from_dictionary.assert_called_once_with(processor.add_spin_angle_delta.return_value,
                                                                 replace(input_metadata,
                                                                         descriptor=GLOWS_L3A_DESCRIPTOR))

    @patch("imap_l3_processing.glows.glows_processor.imap_data_access")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    def test_does_not_process_l3b_if_no_zip_files(self, mock_glows_initializer_class,
                                                  mock_save_data,
                                                  mock_imap_data_access):

        mock_glows_initializer_class.validate_and_initialize.return_value = []

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        mock_processing_input_collection = Mock()

        processor = GlowsProcessor(dependencies=mock_processing_input_collection, input_metadata=input_metadata)
        processor.process()

        mock_glows_initializer_class.validate_and_initialize.assert_called_with(input_metadata.version,
                                                                                mock_processing_input_collection)
        mock_save_data.assert_not_called()
        mock_imap_data_access.upload.assert_not_called()

    def test_add_spin_angle_delta(self):
        cases = [
            (60, 3),
            (90, 2)
        ]

        for bins, expected_delta in cases:
            with self.subTest(bin=bins, expected_delta=expected_delta):
                ancillary_files = {}
                with open(get_test_data_path("glows/imap_glows_l3a_20130908085214_orbX_modX_p_v00.json")) as f:
                    example_data = json.load(f)

                with tempfile.TemporaryDirectory() as tempdir:
                    temp_file_path = Path(tempdir) / "settings.json"
                    example_settings = get_test_instrument_team_data_path(
                        "glows/imap_glows_pipeline-settings_20250707_v002.json")

                    with open(example_settings) as file:
                        loaded_file = json.load(file)

                    loaded_file['l3a_nominal_number_of_bins'] = bins

                    with open(temp_file_path, 'w') as file:
                        json.dump(loaded_file, file)
                    ancillary_files['settings'] = temp_file_path

                    result = GlowsProcessor.add_spin_angle_delta(deepcopy(example_data), ancillary_files)
                for k, v in example_data.items():
                    if k != "daily_lightcurve":
                        self.assertEqual(v, result[k])

                for k2, v2 in example_data["daily_lightcurve"].items():
                    self.assertEqual(v2, result["daily_lightcurve"][k2])
                spin_angle_delta = result["daily_lightcurve"]["spin_angle_delta"]
                self.assertEqual(len(example_data["daily_lightcurve"]["spin_angle"]), len(spin_angle_delta))
                self.assertTrue(np.all(spin_angle_delta == expected_delta))

    @patch("imap_l3_processing.processor.spiceypy")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCDependencies")
    @patch("imap_l3_processing.glows.glows_processor.imap_data_access")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind')
    @patch('imap_l3_processing.glows.glows_processor.filter_out_bad_days')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    def test_process_l3bc(self, mock_generate_l3bc, mock_filter_bad_days, mock_l3c_model_class, mock_l3b_model_class,
                          mock_glows_initializer_class, mock_save_data, mock_imap_data_access, mock_l3bc_dependencies,
                          mock_spiceypy):
        mock_spiceypy.ktotal.return_value = 1
        mock_spiceypy.kdata.return_value = ['kernel_1', 'type', 'source', 'handle']

        mock_glows_initializer_class.validate_and_initialize.return_value = [
            sentinel.zip_file_path_1,
            sentinel.zip_file_path_2,
        ]

        first_dependency = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data_1,
                                                 external_files=sentinel.external_files_1,
                                                 ancillary_files={
                                                     'bad_days_list': sentinel.bad_days_list_1,
                                                 },
                                                 carrington_rotation_number=sentinel.cr_1,
                                                 start_date=datetime(2024, 1, 1),
                                                 end_date=datetime(2024, 1, 30),
                                                 zip_file_path=Path('some/path1.zip'))
        second_dependency = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data_2,
                                                  external_files=sentinel.external_files_2,
                                                  ancillary_files={
                                                      'bad_days_list': sentinel.bad_days_list_2,
                                                  },
                                                  carrington_rotation_number=sentinel.cr_2,
                                                  start_date=datetime(2024, 2, 1),
                                                  end_date=datetime(2024, 2, 28),
                                                  zip_file_path=Path('some/path2.zip'))

        mock_l3bc_dependencies.fetch_dependencies.side_effect = [first_dependency, second_dependency]

        mock_generate_l3bc.side_effect = [(sentinel.l3b_data_1, sentinel.l3c_data_1),
                                          (sentinel.l3b_data_2, sentinel.l3c_data_2)]
        mock_filter_bad_days.side_effect = [sentinel.filtered_days_1, sentinel.filtered_days_2]

        l3b_model_1 = Mock()
        l3b_model_1.parent_file_names = ["file1"]
        l3b_model_2 = Mock()
        l3b_model_2.parent_file_names = ["file2"]
        l3c_model_1 = Mock()
        l3c_model_1.parent_file_names = ["file3"]
        l3c_model_2 = Mock()
        l3c_model_2.parent_file_names = ["file4"]
        mock_l3b_model_class.from_instrument_team_dictionary.side_effect = [l3b_model_1,
                                                                            l3b_model_2]
        mock_l3c_model_class.from_instrument_team_dictionary.side_effect = [l3c_model_1,
                                                                            l3c_model_2]
        mock_save_data.side_effect = ["path/to/l3b_file_1.cdf",
                                      sentinel.l3c_cdf_path_1,
                                      "path/to/l3b_file_2.cdf",
                                      sentinel.l3c_cdf_path_2]

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)
        processor.process()

        mock_l3bc_dependencies.fetch_dependencies.assert_has_calls(
            [call(sentinel.zip_file_path_1), call(sentinel.zip_file_path_2)])

        dependencies_with_filtered_list_1 = replace(first_dependency, l3a_data=sentinel.filtered_days_1)
        dependencies_with_filtered_list_2 = replace(second_dependency, l3a_data=sentinel.filtered_days_2)

        mock_filter_bad_days.assert_has_calls(
            [call(sentinel.l3a_data_1, sentinel.bad_days_list_1),
             call(sentinel.l3a_data_2, sentinel.bad_days_list_2)])

        mock_generate_l3bc.assert_has_calls(
            [call(dependencies_with_filtered_list_1), call(dependencies_with_filtered_list_2)])

        expected_l3b_metadata_1 = InputMetadata("glows", "l3b", first_dependency.start_date,
                                                first_dependency.end_date, 'v02', "ion-rate-profile")
        expected_l3b_metadata_2 = InputMetadata("glows", "l3b", second_dependency.start_date,
                                                second_dependency.end_date, 'v02', "ion-rate-profile")
        expected_l3c_metadata_1 = InputMetadata("glows", "l3c", first_dependency.start_date,
                                                first_dependency.end_date, 'v02', "sw-profile")
        expected_l3c_metadata_2 = InputMetadata("glows", "l3c", second_dependency.start_date,
                                                second_dependency.end_date, 'v02', "sw-profile")
        mock_l3b_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3b_data_1, expected_l3b_metadata_1),
             call(sentinel.l3b_data_2, expected_l3b_metadata_2)])

        mock_l3c_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3c_data_1, expected_l3c_metadata_1),
             call(sentinel.l3c_data_2, expected_l3c_metadata_2)])

        mock_save_data.assert_has_calls(
            [call(l3b_model_1), call(l3c_model_1), call(l3b_model_2), call(l3c_model_2)])

        self.assertEqual(["file1", "path1.zip", "kernel_1"], l3b_model_1.parent_file_names)
        self.assertEqual(["file2", "path2.zip", "kernel_1"], l3b_model_2.parent_file_names)
        self.assertEqual(["file3", "path1.zip", "kernel_1", "l3b_file_1.cdf"], l3c_model_1.parent_file_names)
        self.assertEqual(["file4", "path2.zip", "kernel_1", "l3b_file_2.cdf"], l3c_model_2.parent_file_names)
        mock_imap_data_access.upload.assert_has_calls([
            call("path/to/l3b_file_1.cdf"),
            call(sentinel.l3c_cdf_path_1),
            call(sentinel.zip_file_path_1),
            call("path/to/l3b_file_2.cdf"),
            call(sentinel.l3c_cdf_path_2),
            call(sentinel.zip_file_path_2),
        ])

    @patch('imap_l3_processing.glows.glows_processor.make_l3c_data_with_fill')
    @patch('imap_l3_processing.glows.glows_processor.make_l3b_data_with_fill')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind')
    @patch('imap_l3_processing.glows.glows_processor.filter_out_bad_days')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc',
           side_effect=CannotProcessCarringtonRotationError(""))
    def test_create_fill_val_when_generate_l3bc_throws_exception(self, _, mock_filter_bad_days,
                                                                 mock_l3c_model_class, mock_l3b_model_class,
                                                                 mock_make_l3b_data_with_fill,
                                                                 mock_make_l3c_data_with_fill):

        input_metadata = InputMetadata('glows', "l3b", sentinel.start_time,
                                       sentinel.end_time,
                                       'v02')

        dependencies = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data,
                                             external_files=sentinel.external_files,
                                             ancillary_files={
                                                 'bad_days_list': sentinel.bad_days_list,
                                             },
                                             carrington_rotation_number=sentinel.cr,
                                             start_date=sentinel.start_time, end_date=sentinel.end_time,
                                             zip_file_path=Path('some/path.zip'))
        l3b_metadata = InputMetadata("glows", "l3b", dependencies.start_date,
                                     dependencies.end_date, 'v02', "ion-rate-profile")

        l3c_metadata = InputMetadata("glows", "l3c", dependencies.start_date,
                                     dependencies.end_date, 'v02', "sw-profile")

        mock_filter_bad_days.return_value = sentinel.filtered_days

        mock_make_l3b_data_with_fill.return_value = sentinel.l3b_fill
        mock_make_l3c_data_with_fill.return_value = sentinel.l3c_fill

        l3b_model = Mock()
        l3b_model.parent_file_names = []
        l3c_model = Mock()
        l3c_model.parent_file_names = []

        mock_l3b_model_class.from_instrument_team_dictionary.return_value = l3b_model
        mock_l3c_model_class.from_instrument_team_dictionary.return_value = l3c_model

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)
        result_l3b, result_l3c = processor.process_l3bc(dependencies)

        self.assertEqual([call(dependencies)], mock_make_l3b_data_with_fill.call_args_list)
        self.assertEqual([call(dependencies)], mock_make_l3c_data_with_fill.call_args_list)
        mock_l3b_model_class.from_instrument_team_dictionary.assert_called_once_with(sentinel.l3b_fill,
                                                                                     l3b_metadata)
        mock_l3c_model_class.from_instrument_team_dictionary.assert_called_once_with(sentinel.l3c_fill,
                                                                                     l3c_metadata)
        self.assertEqual(l3b_model, result_l3b)
        self.assertEqual(l3c_model, result_l3c)

    def test_process_l3bc_all_data_in_bad_season_returns_data_products_with_fill_values(self):
        cr = 2093
        start_date = datetime(2025, 4, 3)
        end_date = datetime(2025, 4, 5)
        input_metadata = InputMetadata('glows', "l3b", start_date, end_date, 'v02', 'ion-rate-profile')
        external_files = {
            'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            'omni_raw_data': get_test_instrument_team_data_path('glows/omni_2010.dat')
        }
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
            'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
            'pipeline_settings': get_test_instrument_team_data_path(
                'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
        }
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [
            create_glows_l3a_dictionary_from_cdf(
                l3a_data_folder_path / 'imap_glows_l3a_hist_20100201-repoint00032_v001.cdf')]

        dependencies = GlowsL3BCDependencies(l3a_data=l3a_data, external_files=external_files,
                                             ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                             start_date=start_date, end_date=end_date,
                                             zip_file_path=Path('some/path.zip'))

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)
        actual_l3b, actual_l3c = processor.process_l3bc(dependencies)

        expected_lat_grid = [-90, -80, -70, -60, -50, -40, -30, -20, -10,
                             0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        expected_l3b = GlowsL3BIonizationRate(
            input_metadata=input_metadata,
            epoch=np.array([datetime(2010, 2, 13, 9, 27, 43, 199991)]),
            epoch_delta=np.array([CARRINGTON_ROTATION_IN_NANOSECONDS / 2]),
            cr=np.array([cr]),
            uv_anisotropy_factor=np.full((1, 19), 1),
            lat_grid=np.array(expected_lat_grid),
            lat_grid_delta=np.zeros(19),
            sum_rate=np.full((1, 19), np.nan),
            ph_rate=np.full((1, 19), np.nan),
            cx_rate=np.full((1, 19), np.nan),
            sum_uncert=np.full((1, 19), np.nan),
            ph_uncert=np.full((1, 19), np.nan),
            cx_uncert=np.full((1, 19), np.nan),
            lat_grid_label=[f"{x}°" for x in expected_lat_grid],
        )

        self.assertEqual(expected_l3b.input_metadata, actual_l3b.input_metadata)
        np.testing.assert_array_equal(expected_l3b.epoch, actual_l3b.epoch)
        np.testing.assert_array_equal(expected_l3b.epoch_delta, actual_l3b.epoch_delta)
        np.testing.assert_array_equal(expected_l3b.cr, actual_l3b.cr)
        np.testing.assert_array_equal(expected_l3b.uv_anisotropy_factor, actual_l3b.uv_anisotropy_factor)
        np.testing.assert_array_equal(expected_l3b.lat_grid, actual_l3b.lat_grid)
        np.testing.assert_array_equal(expected_l3b.lat_grid_delta, actual_l3b.lat_grid_delta)
        np.testing.assert_array_equal(expected_l3b.sum_rate, actual_l3b.sum_rate)
        np.testing.assert_array_equal(expected_l3b.ph_rate, actual_l3b.ph_rate)
        np.testing.assert_array_equal(expected_l3b.cx_rate, actual_l3b.cx_rate)
        np.testing.assert_array_equal(expected_l3b.sum_uncert, actual_l3b.sum_uncert)
        np.testing.assert_array_equal(expected_l3b.ph_uncert, actual_l3b.ph_uncert)
        np.testing.assert_array_equal(expected_l3b.cx_uncert, actual_l3b.cx_uncert)
        self.assertEqual(expected_l3b.lat_grid_label, actual_l3b.lat_grid_label)

    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.imap_data_access.upload')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.get_repoint_date_range")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    def test_process_l3e_ultra(self, mock_l3e_dependencies, mock_get_repoint_date_range, mock_determine_call_args,
                               mock_run, mock_convert_dat_to_glows_l3e_ul_product, mock_upload, mock_save_data,
                               mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='survival-probability-ul')
        dependencies = Mock()

        l3e_dependencies = Mock()
        repointing = 10

        mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, repointing)
        epoch = datetime(year=2024, month=10, day=7)
        end_date = datetime(year=2024, month=10, day=7, hour=23)

        epoch_delta = (end_date - epoch) / 2

        start_date = np.datetime64(epoch.isoformat())

        end_date = np.datetime64(end_date.isoformat())
        mock_get_repoint_date_range.return_value = (start_date, end_date)

        ultra_args = ["20241007_000000", "2024.765", "vx", "vy", "vz", "30.000"]

        mock_determine_call_args.return_value = ultra_args

        ultra_data = Mock()
        mock_convert_dat_to_glows_l3e_ul_product.return_value = ultra_data

        mock_save_data.return_value = sentinel.ultra_path

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        processor.process()

        mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies, input_metadata.descriptor)

        mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

        mock_get_repoint_date_range.assert_called_once_with(repointing)

        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

        mock_determine_call_args.assert_called_once_with(epoch,
                                                         datetime(year=2024, month=10, day=7, hour=11, minute=30), 30)

        mock_run.assert_called_once_with(["./survProbUltra", "20241007_000000", "2024.765", "vx", "vy", "vz", "30.000"])

        mock_convert_dat_to_glows_l3e_ul_product.assert_called_once_with(input_metadata, Path(
            "probSur.Imap.Ul_20241007_000000_2024.765.dat"), np.array(epoch), np.array(epoch_delta))

        mock_save_data.assert_called_once_with(ultra_data)
        survival_data_product: GlowsL3EUltraData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        mock_upload.assert_called_once_with(sentinel.ultra_path)

    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.imap_data_access.upload')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.get_repoint_date_range")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    def test_process_l3e_hi(self, mock_l3e_dependencies, mock_get_repoint_date_range, mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_hi_product, mock_save_data, mock_upload,
                            mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        test_cases = [("hi45", "45", "135.000"), ("hi90", "90", "90.000")]

        for test_name, descriptor, elongation in test_cases:
            with self.subTest(test_name):
                mock_l3e_dependencies.reset_mock()
                mock_get_repoint_date_range.reset_mock()
                mock_determine_call_args.reset_mock()
                mock_run.reset_mock()
                mock_convert_dat_to_glows_l3e_hi_product.reset_mock()
                mock_save_data.reset_mock()
                mock_upload.reset_mock()

                input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                               datetime(2024, 10, 8, 10, 00, 00),
                                               'v001', descriptor=f'survival-probability-hi-{descriptor}')

                dependencies = Mock()

                l3e_dependencies = Mock()
                repointing = 10

                mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, repointing)
                epoch = datetime(year=2024, month=10, day=7)
                end_date = datetime(year=2024, month=10, day=7, hour=23)

                epoch_delta = (end_date - epoch) / 2

                start_date = np.datetime64(epoch.isoformat())

                end_date = np.datetime64(end_date.isoformat())
                mock_get_repoint_date_range.return_value = (start_date, end_date)

                call_args_1 = ["20241007_000000", "2024.765", "vx", "vy", "vz", elongation]

                mock_determine_call_args.return_value = call_args_1

                mock_convert_dat_to_glows_l3e_hi_product.return_value = sentinel.hi_data

                hi_data = Mock()
                mock_save_data.return_value = hi_data

                processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
                processor.process()

                mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies,
                                                                                 input_metadata.descriptor)

                mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

                mock_get_repoint_date_range.assert_called_once_with(repointing)

                self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
                self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

                mock_determine_call_args.assert_called_once_with(epoch,
                                                                 datetime(year=2024, month=10, day=7, hour=11,
                                                                          minute=30), int(float(elongation)))

                mock_run.assert_called_once_with(["./survProbHi"] + call_args_1)

                mock_convert_dat_to_glows_l3e_hi_product.assert_called_once_with(
                    input_metadata,
                    Path(f"probSur.Imap.Hi_20241007_000000_2024.765_{elongation[:5]}.dat"),
                    np.array(epoch), np.array(epoch_delta)
                )

                mock_save_data.assert_called_once_with(sentinel.hi_data)
                survival_data_product: GlowsL3EHiData = mock_save_data.call_args_list[0].args[0]
                self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                                 survival_data_product.parent_file_names)

                mock_upload.assert_called_once_with(hi_data)

    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.imap_data_access.upload')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.get_repoint_date_range")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    def test_process_l3e_lo(self, mock_l3e_dependencies, mock_get_repoint_date_range, mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_lo_product, mock_save_data, mock_upload,
                            mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='survival-probability-lo')
        dependencies = Mock()

        l3e_dependencies = Mock()
        repointing = 10

        mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, repointing)
        epoch = datetime(year=2024, month=10, day=7)
        end_date = datetime(year=2024, month=10, day=7, hour=23)

        epoch_delta = (end_date - epoch) / 2

        start_date = np.datetime64("2024-10-07T00:00:00.000000000")

        end_date = np.datetime64(end_date.isoformat())
        mock_get_repoint_date_range.return_value = (start_date, end_date)

        call_args_1 = ["20241007_000000", "2024.765", "vx", "vy", "vz", "90.000"]

        mock_determine_call_args.return_value = call_args_1

        lo_data = Mock()
        mock_convert_dat_to_glows_l3e_lo_product.return_value = lo_data

        mock_save_data.return_value = sentinel.lo_path

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        processor.process()

        mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies, input_metadata.descriptor)

        mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

        mock_get_repoint_date_range.assert_called_once_with(repointing)

        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

        mock_determine_call_args.assert_called_once_with(epoch,
                                                         datetime(year=2024, month=10, day=7, hour=11, minute=30),
                                                         90)

        mock_run.assert_called_once_with(["./survProbLo"] + call_args_1)

        mock_convert_dat_to_glows_l3e_lo_product.assert_called_once_with(
            input_metadata,
            Path("probSur.Imap.Lo_20241007_000000_2024.765_90.00.dat"),
            np.array(epoch), np.array(epoch_delta)
        )

        mock_save_data.assert_called_once_with(lo_data)
        survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        mock_upload.assert_called_once_with(sentinel.lo_path)

    @patch('imap_l3_processing.glows.glows_processor.convert_json_l3d_to_cdf')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.shutil')
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3c_json_file_from_cdf")
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3b_json_file_from_cdf")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3DDependencies")
    def test_process_l3d(self, mock_l3d_dependencies_constructor, mock_create_glows_l3b_json_file_from_cdf,
                         mock_create_glows_l3c_json_file_from_cdf, mock_shutil, mock_os, mock_run,
                         mock_convert_json_l3d_to_cdf):

        input_metadata = InputMetadata('glows', "l3d", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='solar-hist')

        input_data_collection = Mock()

        mock_l3d_dependencies = Mock(spec=GlowsL3DDependencies)
        mock_l3d_dependencies.ancillary_files = {
            'pipeline_settings': Path('path/to/pipeline_settings'),
            'WawHelioIon': {
                'speed': Path('path/to/speed'),
                'p-dens': Path('path/to/p-dens'),
                'uv-anis': Path('path/to/uv-anis'),
                'phion': Path('path/to/phion'),
                'lya': Path('path/to/lya'),
                'e-dens': Path('path/to/e-dens')
            }
        }
        mock_l3d_dependencies.external_files = {
            'lya_raw_data': Path('path/to/lya'),
        }
        mock_l3d_dependencies.l3b_file_paths = [sentinel.l3b_file_1, sentinel.l3b_file_2]
        mock_l3d_dependencies.l3c_file_paths = [sentinel.l3c_file_1, sentinel.l3c_file_2]
        mock_l3d_dependencies_constructor.fetch_dependencies.return_value = mock_l3d_dependencies

        ran_out_of_l3b_exception = r"""Traceback (most recent call last):
          File "...\glows_L3bcd_pipeline\generate_l3d.py", line 46, in <module>
            solar_param_hist.update_solar_params_hist(EXT_DEPENDENCIES,data_l3b,data_l3c)
          File "...\glows_L3bcd_pipeline\toolkit\l3d_SolarParamHistory.py", line 554, in update_solar_params_hist
            self._update_l3bc_data(data_l3b,data_l3c,CR)
          File "...\glows_L3bcd_pipeline\toolkit\l3d_SolarParamHistory.py", line 516, in _update_l3bc_data
            anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read = self._generate_cr_solar_params(CR, data_l3b, data_l3c)
          File "...\glows_L3bcd_pipeline\toolkit\l3d_SolarParamHistory.py", line 181, in _generate_cr_solar_params
            if idx_read[-1]>=len(CR_list): raise Exception('L3d not generated: there is not enough L3bc data to interpolate')
        Exception: L3d not generated: there is not enough L3bc data to interpolate
        """
        cr_number = 2096

        mock_run.side_effect = [CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number}'),
                                CalledProcessError(cmd="", returncode=1, stderr=ran_out_of_l3b_exception)]

        mock_convert_json_l3d_to_cdf.return_value = sentinel.l3d_cdf_path

        processor = GlowsProcessor(input_data_collection, input_metadata)
        processor.process()

        mock_l3d_dependencies_constructor.fetch_dependencies.assert_called_once_with(input_data_collection)

        mock_os.makedirs.assert_has_calls([
            call(PATH_TO_L3D_TOOLKIT / 'data_ancillary', exist_ok=True),
            call(PATH_TO_L3D_TOOLKIT / 'external_dependencies', exist_ok=True)
        ])

        mock_shutil.move.assert_has_calls([
            call(Path('path/to/pipeline_settings'),
                 PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_pipeline-settings-L3bc_v001.json'),
            call(Path('path/to/speed'),
                 PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_plasma-speed-Legendre-2010a_v001.dat'),
            call(Path('path/to/p-dens'),
                 PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_proton-density-Legendre-2010a_v001.dat'),
            call(Path('path/to/uv-anis'),
                 PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_uv-anisotropy-2010a_v001.dat'),
            call(Path('path/to/phion'), PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_photoion-2010a_v001.dat'),
            call(Path('path/to/lya'), PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_lya-2010a_v001.dat'),
            call(Path('path/to/e-dens'),
                 PATH_TO_L3D_TOOLKIT / 'data_ancillary' / 'imap_glows_electron-density-2010a_v001.dat'),
            call(Path('path/to/lya'), PATH_TO_L3D_TOOLKIT / 'external_dependencies' / 'lyman_alpha_composite.nc'),
        ])

        mock_create_glows_l3c_json_file_from_cdf.assert_has_calls([
            call(sentinel.l3c_file_1), call(sentinel.l3c_file_2)
        ])

        mock_create_glows_l3b_json_file_from_cdf.assert_has_calls([
            call(sentinel.l3b_file_1), call(sentinel.l3b_file_2)
        ])

        expected_working_directory = Path(l3d.__file__).parent / 'science'

        self.assertEqual(2, mock_run.call_count)
        mock_run.assert_has_calls([
            call([sys.executable, './generate_l3d.py'], cwd=str(expected_working_directory), check=True,
                 capture_output=True, text=True),
            call([sys.executable, './generate_l3d.py'], cwd=str(expected_working_directory), check=True,
                 capture_output=True, text=True),
        ])

        mock_convert_json_l3d_to_cdf.assert_called_once_with(
            expected_working_directory / 'data_l3d' / f'imap_glows_l3d_cr_{cr_number}_v00.json', 'l3d.cdf')

    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.shutil')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_l3d_to_cdf')
    @patch('imap_l3_processing.glows.glows_processor.run')
    def test_process_l3d_handles_unexpcted_exception_from_science(self, mock_run,
                                                                  mock_convert_json_l3d_to_cdf, _, __):
        ancillary_files = {
            'pipeline_settings': Path('path/to/pipeline_settings'),
            'WawHelioIon': {
                'speed': Path('path/to/speed'),
                'p-dens': Path('path/to/p-dens'),
                'uv-anis': Path('path/to/uv-anis'),
                'phion': Path('path/to/phion'),
                'lya': Path('path/to/lya'),
                'e-dens': Path('path/to/e-dens')
            }
        }
        external_files = {
            'lya_raw_data': Path('path/to/lya'),
        }
        l3b_file_paths = []
        l3c_file_paths = []
        l3d_dependencies = GlowsL3DDependencies(ancillary_files=ancillary_files,
                                                external_files=external_files,
                                                l3b_file_paths=l3b_file_paths,
                                                l3c_file_paths=l3c_file_paths)

        unexpected_exception = r"""Traceback (most recent call last):
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\generate_l3d.py", line 46, in <module>
            solar_param_hist.update_solar_params_hist(EXT_DEPENDENCIES,data_l3b,data_l3c)
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\l3d_SolarParamHistory.py", line 554, in update_solar_params_hist
            self._update_l3bc_data(data_l3b,data_l3c,CR)
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\l3d_SolarParamHistory.py", line 516, in _update_l3bc_data
            anisotropy_CR, ph_ion_CR, sw_speed_CR, p_dens_CR, e_dens_CR, idx_read = self._generate_cr_solar_params(CR, data_l3b, data_l3c)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\funcs.py", line 71, in calculate_mean_date
            l3a=read_json(f)
                ^^^^^^^^^^^^
          File "...\imap-L3-processing\imap_l3_processing\glows\l3d\science\toolkit\funcs.py", line 451, in read_json
            fp = open(fn, 'r')
                 ^^^^^^^^^^^^^
        FileNotFoundError: [Errno 2] No such file or directory: 'imap_glows_l3a_hist_20100511-repoint00131_v011.cdf'"""

        mock_run.side_effect = [CalledProcessError(cmd="", returncode=1, stderr=unexpected_exception)]

        processor = GlowsProcessor(Mock(), Mock())

        with self.assertRaises(Exception) as context:
            processor.process_l3d(l3d_dependencies)
        self.assertEqual(unexpected_exception, str(context.exception))

        mock_convert_json_l3d_to_cdf.assert_not_called()


if __name__ == '__main__':
    unittest.main()
