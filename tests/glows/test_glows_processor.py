import json
import os
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import patch, Mock, sentinel, call, mock_open, MagicMock

import numpy as np
from imap_data_access import config
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput
from spacepy.pycdf import CDF

from imap_l3_processing.glows import l3d
from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf, create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import ExternalDependencies
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path, get_test_data_folder, \
    assert_dataclass_fields


class TestGlowsProcessor(unittest.TestCase):
    ran_out_of_l3b_exception = r"""Exception: Traceback (most recent call last):
  File ".../imap_L3_processing/imap_l3_processing/glows/l3d/science/./generate_l3d.py", line 36, in <module>
    solar_param_hist.update_solar_params_hist(EXT_DEPENDENCIES, data_l3b, data_l3c, CR_current)
  File ".../imap_L3_processing/imap_l3_processing/glows/l3d/science/toolkit/l3d_SolarParamHistory.py", line 487, in update_solar_params_hist
    self._update_l3bc_data(data_l3b,data_l3c,CR)
  File ".../imap_L3_processing/imap_l3_processing/glows/l3d/science/toolkit/l3d_SolarParamHistory.py", line 424, in _update_l3bc_data
    cr_params, idx_read_b, idx_read_c = self._generate_cr_solar_params(CR, data_l3b, data_l3c)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../imap_L3_processing/imap_l3_processing/glows/l3d/science/toolkit/l3d_SolarParamHistory.py", line 185, in _generate_cr_solar_params
    if idx_read_b[-1]>=len(CR_list_b): raise Exception('L3d not generated: there is not enough L3b data to interpolate')
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Exception: L3d not generated: there is not enough L3b data to interpolate
    """

    def tearDown(self):
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3b'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3b')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3c'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3c')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3d'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3d')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt')

    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_processor_handles_l3a(self, mock_spiceypy, mock_save_data, mock_L3aData,
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
        products = processor.process()

        mock_glows_initializer.assert_not_called()
        mock_glows_dependencies_class.fetch_dependencies.assert_called_with(mock_processing_input_collection)
        expected_data_to_save = create_glows_l3a_from_dictionary(
            mock_L3aData.return_value.data, replace(input_metadata, descriptor=GLOWS_L3A_DESCRIPTOR))
        expected_data_to_save.input_metadata.repointing = mock_fetched_dependencies.repointing
        expected_data_to_save.parent_file_names = expected_parent_file_names
        actual_data = mock_save_data.call_args.args[0]
        self.assertEqual(expected_parent_file_names, actual_data.parent_file_names)
        assert_dataclass_fields(expected_data_to_save, actual_data)
        self.assertEqual([mock_cdf_path], products)

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
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind')
    @patch('imap_l3_processing.glows.glows_processor.filter_l3a_files')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    @patch('imap_l3_processing.glows.glows_processor.archive_dependencies')
    def test_process_l3bc(self, mock_archive_dependencies, mock_generate_l3bc, mock_filter_bad_days,
                          mock_l3c_model_class, mock_l3b_model_class, mock_save_data, mock_l3bc_dependencies_class,
                          mock_spiceypy):
        mock_spiceypy.ktotal.return_value = 1
        mock_spiceypy.kdata.return_value = ['kernel_1', 'type', 'source', 'handle']

        first_cr_number_to_process = 2091
        second_cr_number_to_process = 2092

        cr_to_process_1 = Mock(cr_rotation_number=first_cr_number_to_process)
        cr_to_process_2 = Mock(cr_rotation_number=second_cr_number_to_process)

        external_deps = ExternalDependencies(f107_index_file_path=sentinel.f107_index_file_path,
                                             lyman_alpha_path=sentinel.lyman_alpha_path,
                                             omni2_data_path=sentinel.omni2_data_path)

        mock_archive_dependencies.side_effect = [Path("path1.zip"), Path("path2.zip")]

        first_dependency = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data_1,
                                                 external_files=sentinel.external_files_1,
                                                 ancillary_files={
                                                     'bad_days_list': sentinel.bad_days_list_1,
                                                 },
                                                 carrington_rotation_number=first_cr_number_to_process)
        second_dependency = GlowsL3BCDependencies(l3a_data=sentinel.l3a_data_2,
                                                  external_files=sentinel.external_files_2,
                                                  ancillary_files={
                                                      'bad_days_list': sentinel.bad_days_list_2,
                                                  },
                                                  carrington_rotation_number=second_cr_number_to_process)

        mock_l3bc_dependencies_class.from_cr_to_process.side_effect = [first_dependency, second_dependency]

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
        mock_save_data.side_effect = [Path("path/to/l3b_file_1.cdf"),
                                      Path("path/to/l3c_file_1.cdf"),
                                      Path("path/to/l3b_file_2.cdf"),
                                      Path("path/to/l3c_file_2.cdf")]

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)
        processor_output = processor.process_l3bc(external_deps, [(1, cr_to_process_1), (2, cr_to_process_2)])

        mock_l3bc_dependencies_class.from_cr_to_process.assert_has_calls([
            call(cr_to_process_1, external_deps), call(cr_to_process_2, external_deps)
        ])

        dependencies_with_filtered_list_1 = replace(first_dependency, l3a_data=sentinel.filtered_days_1)
        dependencies_with_filtered_list_2 = replace(second_dependency, l3a_data=sentinel.filtered_days_2)

        mock_filter_bad_days.assert_has_calls(
            [call(sentinel.l3a_data_1, sentinel.bad_days_list_1, first_cr_number_to_process),
             call(sentinel.l3a_data_2, sentinel.bad_days_list_2, second_cr_number_to_process)])

        mock_generate_l3bc.assert_has_calls(
            [call(dependencies_with_filtered_list_1), call(dependencies_with_filtered_list_2)])

        expected_l3b_metadata_1 = InputMetadata("glows", "l3b", cr_to_process_1.cr_start_date,
                                                cr_to_process_1.cr_end_date, 'v001', "ion-rate-profile")
        expected_l3b_metadata_2 = InputMetadata("glows", "l3b", cr_to_process_2.cr_start_date,
                                                cr_to_process_2.cr_end_date, 'v002', "ion-rate-profile")
        expected_l3c_metadata_1 = InputMetadata("glows", "l3c", cr_to_process_1.cr_start_date,
                                                cr_to_process_1.cr_end_date, 'v001', "sw-profile")
        expected_l3c_metadata_2 = InputMetadata("glows", "l3c", cr_to_process_2.cr_start_date,
                                                cr_to_process_2.cr_end_date, 'v002', "sw-profile")
        mock_l3b_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3b_data_1, expected_l3b_metadata_1),
             call(sentinel.l3b_data_2, expected_l3b_metadata_2)])

        mock_l3c_model_class.from_instrument_team_dictionary.assert_has_calls(
            [call(sentinel.l3c_data_1, expected_l3c_metadata_1),
             call(sentinel.l3c_data_2, expected_l3c_metadata_2)])

        mock_save_data.assert_has_calls([
            call(l3b_model_1, cr_number=first_cr_number_to_process),
            call(l3c_model_1, cr_number=first_cr_number_to_process),
            call(l3b_model_2, cr_number=second_cr_number_to_process),
            call(l3c_model_2, cr_number=second_cr_number_to_process)
        ])

        mock_archive_dependencies.assert_has_calls([
            call(cr_to_process=cr_to_process_1, external_dependencies=external_deps, version=1),
            call(cr_to_process=cr_to_process_2, external_dependencies=external_deps, version=2),
        ])

        self.assertEqual(["file1", "path1.zip", "kernel_1"], l3b_model_1.parent_file_names)
        self.assertEqual(["file2", "path2.zip", "kernel_1"], l3b_model_2.parent_file_names)
        self.assertEqual(["file3", "path1.zip", "l3b_file_1.cdf", "kernel_1", ], l3c_model_1.parent_file_names)
        self.assertEqual(["file4", "path2.zip", "l3b_file_2.cdf", "kernel_1", ], l3c_model_2.parent_file_names)
        self.assertEqual([
            Path("path/to/l3b_file_1.cdf"),
            Path("path/to/l3c_file_1.cdf"),
            Path("path1.zip"),
            Path("path/to/l3b_file_2.cdf"),
            Path("path/to/l3c_file_2.cdf"),
            Path("path2.zip"),
        ], processor_output.data_products)

        expected_l3bs_by_cr = {
            first_cr_number_to_process: "l3b_file_1.cdf",
            second_cr_number_to_process: "l3b_file_2.cdf"
        }
        self.assertEqual(expected_l3bs_by_cr, processor_output.l3bs_by_cr)
        expected_l3cs_by_cr = {
            first_cr_number_to_process: "l3c_file_1.cdf",
            second_cr_number_to_process: "l3c_file_2.cdf"
        }
        self.assertEqual(expected_l3cs_by_cr, processor_output.l3cs_by_cr)

    @patch('imap_l3_processing.glows.glows_processor.archive_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsInitializer")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCDependencies")
    def test_processor_catches_no_data_error_and_continues(self, mock_glows_l3bc_dependencies_class,
                                                           mock_glows_initializer_class,
                                                           mock_save_data,
                                                           mock_archive_dependencies):
        cr_to_process_1 = Mock(cr_rotation_number=1)
        cr_to_process_2 = Mock(cr_rotation_number=2)

        mock_glows_initializer_class.get_crs_to_process.return_value = [(1, cr_to_process_1), (2, cr_to_process_2)]
        bc_dependencies_1 = MagicMock(l3a_data=[])
        bc_dependencies_2 = MagicMock(l3a_data=[])
        mock_glows_l3bc_dependencies_class.from_cr_to_process.side_effect = [bc_dependencies_1, bc_dependencies_2]

        mock_archive_dependencies.side_effect = [sentinel.zip_file_1, sentinel.zip_file_2]

        l3b_path = "path_to/l3b"
        l3c_path = "path_to/l3c"
        mock_save_data.side_effect = [l3b_path, l3c_path]

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001')

        processor = GlowsProcessor(Mock(), input_metadata)

        mock_l3b = MagicMock()
        mock_l3c = MagicMock()

        processor.process_l3bc = Mock()
        processor.process_l3bc.side_effect = [
            CannotProcessCarringtonRotationError("All days for Carrington Rotation are in a bad season."),
            (mock_l3b, mock_l3c)
        ]
        products = processor.process()

        mock_archive_dependencies.assert_has_calls([call(cr_to_process_1, 1), call(cr_to_process_2, 2)])

        self.assertEqual(2, mock_glows_l3bc_dependencies_class.from_cr_to_process.call_count)
        mock_glows_l3bc_dependencies_class.from_cr_to_process.assert_has_calls([
            call(cr_to_process_1),
            call(cr_to_process_2)
        ])

        self.assertEqual(2, mock_save_data.call_count)
        mock_save_data.assert_has_calls([
            call(mock_l3b, cr_number=2),
            call(mock_l3c, cr_number=2),
        ])

        self.assertEqual(3, len(products))
        self.assertEqual([
            l3b_path,
            l3c_path,
            sentinel.zip_file_2
        ], products)

    @patch('imap_l3_processing.glows.glows_processor.archive_dependencies')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCDependencies.from_cr_to_process")
    def test_process_l3bc_all_data_in_bad_season_catches_exception_and_continues(self,
                                                                                 mock_glows_l3bc_dependencies_from_cr_to_process,
                                                                                 _):
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [
            create_glows_l3a_dictionary_from_cdf(
                l3a_data_folder_path / 'imap_glows_l3a_hist_20100201-repoint00032_v001.cdf')]

        external_dependencies = ExternalDependencies(
            f107_index_file_path=get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            omni2_data_path=get_test_instrument_team_data_path('glows/omni_2010.dat'),
            lyman_alpha_path=None
        )
        cr_to_process = Mock()

        mock_deps = GlowsL3BCDependencies(
            l3a_data=l3a_data,
            ancillary_files={
                'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
                'WawHelioIonMP_parameters': get_test_data_path('glows/imap_glows_WawHelioIonMP_20100101_v002.json'),
                'bad_days_list': get_test_data_path('glows/imap_glows_bad-days-list_v001.dat'),
                'pipeline_settings': get_test_instrument_team_data_path(
                    'glows/imap_glows_pipeline-settings-L3bc_20250707_v002.json')
            },
            external_files={
                'f107_raw_data': get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
                'omni_raw_data': get_test_instrument_team_data_path('glows/omni_2010.dat'),
            },
            carrington_rotation_number=1
        )

        mock_glows_l3bc_dependencies_from_cr_to_process.return_value = mock_deps

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=Mock())
        processor_output = processor.process_l3bc(external_dependencies, [(2, cr_to_process)])

        mock_glows_l3bc_dependencies_from_cr_to_process.assert_called_once_with(cr_to_process, external_dependencies)
        self.assertEqual([], processor_output.data_products)

    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.determine_l3e_files_to_produce')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    @patch("imap_l3_processing.glows.glows_processor.get_pointing_date_range")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_ultra(self, mock_shutil, mock_get_repoint_date_range, mock_l3e_dependencies,
                               mock_determine_call_args,
                               mock_run, mock_convert_dat_to_glows_l3e_ul_product, mock_save_data,
                               mock_determine_l3e_files_to_produce, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='survival-probability-ul')
        dependencies = Mock()

        l3e_dependencies = MagicMock(spec=GlowsL3EDependencies)
        l3e_dependencies.pipeline_settings = {'start_cr': 2090}
        l3e_dependencies.repointing_file = get_test_data_path("fake_2_day_repointing_on_may18_file.csv")
        cr_number = 2092

        mock_determine_l3e_files_to_produce.return_value = [20, 21]
        mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, cr_number)
        epoch_1 = np.datetime64(datetime(year=2024, month=10, day=7))
        epoch_1_end_date = np.datetime64(datetime(year=2024, month=10, day=7, hour=23))
        epoch_2 = np.datetime64(datetime(year=2024, month=10, day=8))
        epoch_2_end_date = np.datetime64(datetime(year=2024, month=10, day=8, hour=23))
        epochs = [(epoch_1, epoch_1_end_date), (epoch_2, epoch_2_end_date)]
        mock_get_repoint_date_range.side_effect = epochs

        epoch_deltas = [(end_date - epoch) / 2 for epoch, end_date in epochs]

        ultra_args = [["20241007_000000", "date.001", "vx", "vy", "vz", "30.000"],
                      ["20241008_000000", "date.002", "vx", "vy", "vz", "30.000"]]

        call_args_object = MagicMock(spec=GlowsL3eCallArguments)
        call_args_object.to_argument_list.side_effect = ultra_args
        mock_determine_call_args.side_effect = [call_args_object, call_args_object]

        mock_convert_dat_to_glows_l3e_ul_product.side_effect = [sentinel.ultra_data_1, sentinel.ultra_data_2]

        mock_save_data.side_effect = ["imap_glows_l3e_survival-probability-ul_20241007-repoint00020_v001.cdf",
                                      "imap_glows_l3e_survival-probability-ul_20241008-repoint00021_v001.cdf"]

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        products = processor.process()

        mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies, input_metadata.descriptor)

        mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

        mock_determine_l3e_files_to_produce.assert_called_once_with(input_metadata.descriptor, 2090, cr_number, 'v001',
                                                                    get_test_data_path(
                                                                        "fake_2_day_repointing_on_may18_file.csv"))

        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

        mock_determine_call_args.assert_has_calls([
            call(epoch_1, epoch_1 + epoch_deltas[0], 30),
            call(epoch_2, epoch_2 + epoch_deltas[1], 30),
        ])

        mock_run.assert_has_calls([call(["./survProbUltra"] + args) for args in ultra_args])

        first_output_data_path = Path("probSur.Imap.Ul_20241007_000000_date.001.dat")
        second_output_data_path = Path("probSur.Imap.Ul_20241008_000000_date.002.dat")

        mock_convert_dat_to_glows_l3e_ul_product.assert_has_calls([
            call(input_metadata, first_output_data_path,
                 np.array([epoch_1]), call_args_object),
            call(input_metadata, second_output_data_path,
                 np.array([epoch_2]), call_args_object)])

        expected_first_data_path = "imap_glows_l3e_survival-probability-ul-raw_20241007-repoint00020_v001.dat"
        expected_second_data_path = "imap_glows_l3e_survival-probability-ul-raw_20241008-repoint00021_v001.dat"

        mock_shutil.move.assert_has_calls(
            [call(first_output_data_path, Path(expected_first_data_path)),
             call(second_output_data_path, Path(expected_second_data_path))]
        )

        mock_save_data.assert_has_calls([call(sentinel.ultra_data_1), call(sentinel.ultra_data_2)])
        survival_data_product: GlowsL3EUltraData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        survival_data_product_2: GlowsL3EUltraData = mock_save_data.call_args_list[1].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product_2.parent_file_names)

        self.assertEqual(products, ["imap_glows_l3e_survival-probability-ul_20241007-repoint00020_v001.cdf",
                                    Path(expected_first_data_path),
                                    "imap_glows_l3e_survival-probability-ul_20241008-repoint00021_v001.cdf",
                                    Path(expected_second_data_path)])

        mock_get_repoint_date_range.assert_has_calls([call(20), call(21)])

    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.determine_l3e_files_to_produce')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.get_pointing_date_range")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_hi(self, mock_shutil, mock_l3e_dependencies,
                            mock_get_repoint_date_range, mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_hi_product, mock_save_data,
                            mock_determine_l3e_files_to_produce,
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
                mock_determine_l3e_files_to_produce.reset_mock()
                mock_shutil.reset_mock()

                input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                               datetime(2024, 10, 8, 10, 00, 00),
                                               'v001', descriptor=f'survival-probability-hi-{descriptor}')
                dependencies = Mock()

                l3e_dependencies = MagicMock(spec=GlowsL3EDependencies)
                l3e_dependencies.pipeline_settings = {'start_cr': 2090}
                l3e_dependencies.repointing_file = get_test_data_path("fake_2_day_repointing_on_may18_file.csv")

                cr_number = 2092
                mock_determine_l3e_files_to_produce.return_value = [20, 21]

                mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, cr_number)
                epoch_1 = np.datetime64(datetime(year=2024, month=10, day=7))
                epoch_1_end_date = np.datetime64(datetime(year=2024, month=10, day=7, hour=23))
                epoch_2 = np.datetime64(datetime(year=2024, month=10, day=8))
                epoch_2_end_date = np.datetime64(datetime(year=2024, month=10, day=8, hour=23))
                epochs = [(epoch_1, epoch_1_end_date), (epoch_2, epoch_2_end_date)]
                mock_get_repoint_date_range.side_effect = epochs

                epoch_deltas = [(end_date - epoch) / 2 for epoch, end_date in epochs]

                hi_args = [["20241007_000000", "date.001", "vx", "vy", "vz", elongation],
                           ["20241008_000000", "date.002", "vx", "vy", "vz", elongation]]

                call_args_object = MagicMock(spec=GlowsL3eCallArguments)
                call_args_object.to_argument_list.side_effect = hi_args
                mock_determine_call_args.side_effect = [call_args_object, call_args_object]

                mock_convert_dat_to_glows_l3e_hi_product.side_effect = [sentinel.hi_data_1, sentinel.hi_data_2]

                mock_save_data.side_effect = [
                    f"imap_glows_l3e_survival-probability-hi{descriptor}_20241007-repoint00020_v001.cdf",
                    f"imap_glows_l3e_survival-probability-hi{descriptor}_20241008-repoint00021_v001.cdf"]

                processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
                products = processor.process()

                mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies,
                                                                                 input_metadata.descriptor)

                mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

                mock_determine_l3e_files_to_produce.assert_called_once_with(input_metadata.descriptor, 2090, cr_number,
                                                                            'v001',
                                                                            get_test_data_path(
                                                                                "fake_2_day_repointing_on_may18_file.csv"))

                self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
                self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

                mock_determine_call_args.assert_has_calls([
                    call(epoch_1, epoch_1 + epoch_deltas[0], float(elongation)),
                    call(epoch_2, epoch_2 + epoch_deltas[1], float(elongation)),
                ])

                mock_run.assert_has_calls([call(["./survProbHi"] + args) for args in hi_args])

                first_output_data_path = f"probSur.Imap.Hi_20241007_000000_date.001_{elongation[:5]}.dat"
                second_output_data_path = f"probSur.Imap.Hi_20241008_000000_date.002_{elongation[:5]}.dat"

                mock_convert_dat_to_glows_l3e_hi_product.assert_has_calls([
                    call(input_metadata, Path(first_output_data_path),
                         np.array([epoch_1]), call_args_object),
                    call(input_metadata, Path(second_output_data_path),
                         np.array([epoch_2]), call_args_object)])

                mock_save_data.assert_has_calls([call(sentinel.hi_data_1), call(sentinel.hi_data_2)])
                survival_data_product: GlowsL3EHiData = mock_save_data.call_args_list[0].args[0]
                self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                                 survival_data_product.parent_file_names)

                survival_data_product_2: GlowsL3EHiData = mock_save_data.call_args_list[1].args[0]
                self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                                 survival_data_product_2.parent_file_names)

                expected_first_output_data_path = f"imap_glows_l3e_survival-probability-hi{descriptor}-raw_20241007-repoint00020_v001.dat"
                expected_second_output_data_path = f"imap_glows_l3e_survival-probability-hi{descriptor}-raw_20241008-repoint00021_v001.dat"

                mock_shutil.move.assert_has_calls([
                    call(Path(first_output_data_path), Path(expected_first_output_data_path)),
                    call(Path(second_output_data_path), Path(expected_second_output_data_path)),
                ])

                self.assertEqual(
                    [f"imap_glows_l3e_survival-probability-hi{descriptor}_20241007-repoint00020_v001.cdf",
                     Path(expected_first_output_data_path),
                     f"imap_glows_l3e_survival-probability-hi{descriptor}_20241008-repoint00021_v001.cdf",
                     Path(expected_second_output_data_path)], products)

                mock_get_repoint_date_range.assert_has_calls([call(20), call(21)])

    @patch('imap_l3_processing.glows.glows_processor.determine_l3e_files_to_produce')
    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.get_pointing_date_range")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_lo(self, mock_shutil, mock_l3e_dependencies, mock_get_repoint_date_range,
                            mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_lo_product, mock_save_data,
                            mock_get_parent_file_names, mock_determine_l3e_files_to_produce):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='survival-probability-lo')
        dependencies = Mock()

        l3e_dependencies = MagicMock(spec=GlowsL3EDependencies)
        l3e_dependencies.elongation = {'2024020': 75, '2024021': 105}

        l3e_dependencies.pipeline_settings = {'start_cr': 2090}
        l3e_dependencies.repointing_file = get_test_data_path("fake_2_day_repointing_on_may18_file.csv")

        cr_number = 2092

        mock_determine_l3e_files_to_produce.return_value = [20, 21, 22]

        mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, cr_number)
        epoch_1 = np.datetime64(datetime(year=2024, month=10, day=7))
        epoch_1_end_date = np.datetime64(datetime(year=2024, month=10, day=7, hour=23))
        epoch_2 = np.datetime64(datetime(year=2024, month=10, day=8))
        epoch_2_end_date = np.datetime64(datetime(year=2024, month=10, day=8, hour=23))
        epoch_3 = np.datetime64(datetime(year=2024, month=10, day=9))
        epoch_3_end_date = np.datetime64(datetime(year=2024, month=10, day=9, hour=23))
        epochs = [(epoch_1, epoch_1_end_date), (epoch_2, epoch_2_end_date), (epoch_3, epoch_3_end_date)]

        mock_get_repoint_date_range.side_effect = epochs

        epoch_deltas = [(end_date - epoch) / 2 for epoch, end_date in epochs]

        lo_call_args = [
            ["20241007_000000", "date.100", "vx", "vy", "vz", "75.000"],
            ["20241008_000000", "date.200", "vx", "vy", "vz", "105.000"]]

        call_args_object = MagicMock(spec=GlowsL3eCallArguments)
        call_args_object.to_argument_list.side_effect = lo_call_args
        mock_determine_call_args.side_effect = [call_args_object, call_args_object]

        lo_data_1 = Mock()
        lo_data_2 = Mock()
        mock_convert_dat_to_glows_l3e_lo_product.side_effect = [lo_data_1, lo_data_2]

        mock_save_data.side_effect = ["imap_glows_l3e_survival-probability-lo_20241007-repoint00020_v001.cdf",
                                      "imap_glows_l3e_survival-probability-lo_20241008-repoint00021_v001.cdf"]

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        products = processor.process()

        mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies, input_metadata.descriptor)

        mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

        mock_get_repoint_date_range.assert_has_calls([call(20), call(21)])

        mock_determine_l3e_files_to_produce.assert_called_once_with(input_metadata.descriptor, 2090, cr_number, 'v001',
                                                                    get_test_data_path(
                                                                        "fake_2_day_repointing_on_may18_file.csv"))

        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

        mock_determine_call_args.assert_has_calls([
            call(epoch_1, epoch_1 + epoch_deltas[0], 75),
            call(epoch_2, epoch_2 + epoch_deltas[1], 105)
        ])

        mock_run.assert_has_calls([
            call(["./survProbLo"] + lo_call_args[0]),
            call(["./survProbLo"] + lo_call_args[1]),
        ])

        first_output_file_path = "probSur.Imap.Lo_20241007_000000_date.100_75.00.dat"
        second_output_file_path = "probSur.Imap.Lo_20241008_000000_date.200_105.0.dat"

        mock_convert_dat_to_glows_l3e_lo_product.assert_has_calls([
            call(input_metadata, Path(first_output_file_path),
                 np.array([epoch_1.astype(datetime)]), 75, call_args_object),
            call(input_metadata, Path(second_output_file_path),
                 np.array([epoch_2.astype(datetime)]), 105, call_args_object),
        ])

        expected_first_output_file_path = "imap_glows_l3e_survival-probability-lo-raw_20241007-repoint00020_v001.dat"
        expected_second_output_file_path = "imap_glows_l3e_survival-probability-lo-raw_20241008-repoint00021_v001.dat"

        mock_shutil.move.assert_has_calls([
            call(Path(first_output_file_path), Path(expected_first_output_file_path)),
            call(Path(second_output_file_path), Path(expected_second_output_file_path)),
        ])

        mock_save_data.assert_has_calls([call(lo_data_1), call(lo_data_2)])
        survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[1].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        self.assertEqual(["imap_glows_l3e_survival-probability-lo_20241007-repoint00020_v001.cdf",
                          Path(expected_first_output_file_path),
                          "imap_glows_l3e_survival-probability-lo_20241008-repoint00021_v001.cdf",
                          Path(expected_second_output_file_path)], products)

    @patch('imap_l3_processing.glows.glows_processor.determine_l3e_files_to_produce')
    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.get_pointing_date_range")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EDependencies")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_skips_repointing_on_exception(self, mock_shutil, mock_l3e_dependencies,
                                                       mock_get_repoint_date_range,
                                                       mock_determine_call_args,
                                                       mock_run, mock_convert_dat_to_glows_l3e_lo_product,
                                                       mock_save_data, mock_get_parent_file_names,
                                                       mock_determine_l3e_files_to_produce):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        input_metadata = InputMetadata('glows', "l3e", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='survival-probability-lo')

        dependencies = Mock()

        l3e_dependencies = MagicMock(spec=GlowsL3EDependencies)
        l3e_dependencies.elongation = {'2024020': 75, '2024021': 105}

        l3e_dependencies.pipeline_settings = {'start_cr': 2090}
        l3e_dependencies.repointing_file = get_test_data_path("fake_2_day_repointing_on_may18_file.csv")
        cr_number = 2092

        mock_determine_l3e_files_to_produce.return_value = [20, 21, 22]

        mock_l3e_dependencies.fetch_dependencies.return_value = (l3e_dependencies, cr_number)
        epoch_1 = np.datetime64(datetime(year=2024, month=10, day=7))
        epoch_1_end_date = np.datetime64(datetime(year=2024, month=10, day=7, hour=23))
        epoch_2 = np.datetime64(datetime(year=2024, month=10, day=8))
        epoch_2_end_date = np.datetime64(datetime(year=2024, month=10, day=8, hour=23))
        epoch_3 = np.datetime64(datetime(year=2024, month=10, day=9))
        epoch_3_end_date = np.datetime64(datetime(year=2024, month=10, day=9, hour=23))
        epochs = [(epoch_1, epoch_1_end_date), (epoch_2, epoch_2_end_date), (epoch_3, epoch_3_end_date)]

        mock_get_repoint_date_range.side_effect = epochs

        epoch_deltas = [(end_date - epoch) / 2 for epoch, end_date in epochs]

        lo_call_args = [
            ["20241007_000000", "date.100", "vx", "vy", "vz", "75.000"],
            ["20241008_000000", "date.200", "vx", "vy", "vz", "105.000"]]

        call_args_object = MagicMock(spec=GlowsL3eCallArguments)
        call_args_object.to_argument_list.side_effect = lo_call_args
        mock_determine_call_args.side_effect = [call_args_object, call_args_object]

        expected_cdf_output = "imap_glows_l3e_survival-probability-lo_20241008-repoint00021_v001.cdf"

        lo_data_1 = Mock()
        lo_data_2 = Mock()
        mock_convert_dat_to_glows_l3e_lo_product.side_effect = [lo_data_1, lo_data_2]
        mock_save_data.side_effect = [ValueError(),
                                      expected_cdf_output]

        processor = GlowsProcessor(dependencies=dependencies, input_metadata=input_metadata)
        products = processor.process()

        mock_l3e_dependencies.fetch_dependencies.assert_called_once_with(dependencies, input_metadata.descriptor)

        mock_l3e_dependencies.fetch_dependencies.return_value[0].rename_dependencies.assert_called_once()

        mock_get_repoint_date_range.assert_has_calls([call(20), call(21)])

        mock_determine_l3e_files_to_produce.assert_called_once_with(input_metadata.descriptor, 2090, cr_number, 'v001',
                                                                    get_test_data_path(
                                                                        "fake_2_day_repointing_on_may18_file.csv"))

        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][0], datetime)
        self.assertIsInstance(mock_determine_call_args.call_args_list[0][0][1], datetime)

        mock_determine_call_args.assert_has_calls([
            call(epoch_1, epoch_1 + epoch_deltas[0], 75),
            call(epoch_2, epoch_2 + epoch_deltas[1], 105)
        ])

        mock_run.assert_has_calls([
            call(["./survProbLo"] + lo_call_args[0]),
            call(["./survProbLo"] + lo_call_args[1]),
        ])

        second_output_dat_file = Path("probSur.Imap.Lo_20241008_000000_date.200_105.0.dat")

        mock_convert_dat_to_glows_l3e_lo_product.assert_has_calls([
            call(input_metadata, Path("probSur.Imap.Lo_20241007_000000_date.100_75.00.dat"),
                 np.array([epoch_1.astype(datetime)]), 75, call_args_object),
            call(input_metadata, second_output_dat_file,
                 np.array([epoch_2.astype(datetime)]), 105, call_args_object),
        ])

        expected_second_output_dat_file = Path(
            "imap_glows_l3e_survival-probability-lo-raw_20241008-repoint00021_v001.dat")

        mock_shutil.move.assert_called_once_with(second_output_dat_file, expected_second_output_dat_file)

        mock_save_data.assert_has_calls([call(lo_data_1), call(lo_data_2)])
        survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[1].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        self.assertEqual(products, [expected_cdf_output, expected_second_output_dat_file])

    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3DDependencies.fetch_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.set_version_on_txt_files')
    @patch('imap_l3_processing.glows.glows_processor.json')
    def test_processor_handles_l3d(self, mock_json, mock_set_version_on_txt_files, mock_save_data,
                                   mock_fetch_dependencies, mock_convert_json_to_l3d,
                                   mock_run, mock_os, _, __):

        mock_deps = Mock()
        mock_deps.ancillary_files = {'pipeline_settings': get_test_data_path(
            "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json")}
        mock_deps.l3b_file_paths = []
        mock_deps.l3c_file_paths = []
        processing_input_collection = mock_deps

        mock_json.load.return_value = {'start_cr': 2091}

        input_metadata = InputMetadata('glows', "l3d", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='solar-params-history')

        mock_os.listdir.return_value = ['2096_txt_file_1',
                                        '2096_txt_file_2',
                                        '2096_txt_file_3',
                                        '2096_txt_file_4',
                                        '2096_txt_file_5',
                                        '2096_txt_file_6'
                                        ]

        mock_set_version_on_txt_files.return_value = [
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_1_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_2_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_3_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_4_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_5_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_6_v001'
        ]
        expected_cr = 2096
        mock_run.side_effect = [CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {expected_cr}'),
                                CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)]

        processor = GlowsProcessor(processing_input_collection, input_metadata)
        products = processor.process()
        mock_fetch_dependencies.assert_called_once_with(processing_input_collection)
        mock_save_data.assert_called_once_with(mock_convert_json_to_l3d.return_value, cr_number=expected_cr)

        mock_set_version_on_txt_files.assert_called_once_with([PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_1',
                                                               PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_2',
                                                               PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_3',
                                                               PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_4',
                                                               PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_5',
                                                               PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_6'],
                                                              'v001'
                                                              )

        self.assertEqual([
            mock_save_data.return_value,
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_1_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_2_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_3_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_4_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_5_v001',
            PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / '2096_txt_file_6_v001'
        ], products)

    @patch('imap_l3_processing.glows.glows_processor.set_version_on_txt_files')
    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.glows_processor.json.load')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3c_json_file_from_cdf")
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3b_json_file_from_cdf")
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3DDependencies")
    def test_process_l3d(self, mock_l3d_dependencies_constructor, mock_create_glows_l3b_json_file_from_cdf,
                         mock_create_glows_l3c_json_file_from_cdf, mock_os, mock_run,
                         mock_convert_json_to_l3d_data_product, mock_get_parent_file_names_from_l3d_json, mock_json,
                         _, mock_set_version):

        input_metadata = InputMetadata('glows', "l3d", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='solar-params-history')

        input_data_collection = Mock()
        mock_json.return_value = {'start_cr': 2092}

        mock_l3d_dependencies = Mock(spec=GlowsL3DDependencies)
        mock_l3d_dependencies.ancillary_files = {
            'pipeline_settings':
                'glows/imap_glows_pipeline-settings-l3bcde_20250514_v004.json',
            'WawHelioIon': {
                'speed': 'path/to/speed',
                'p-dens': 'path/to/p-dens',
                'uv-anis': 'path/to/uv-anis',
                'phion': 'path/to/phion',
                'lya': 'path/to/lya',
                'e-dens': 'path/to/e-dens'
            }
        }
        mock_l3d_dependencies.external_files = {
            'lya_raw_data': 'path/to/lya',
        }
        mock_l3d_dependencies.l3b_file_paths = [sentinel.l3b_file_1, sentinel.l3b_file_2]
        mock_l3d_dependencies.l3c_file_paths = [sentinel.l3c_file_1, sentinel.l3c_file_2]
        mock_l3d_dependencies_constructor.fetch_dependencies.return_value = mock_l3d_dependencies

        cr_number = 2092

        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number}'),
            CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number + 1}'),
            CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)]

        mock_os.listdir.return_value = [f'{cr_number}_txt_file_1',
                                        f'{cr_number}_txt_file_2',
                                        f'{cr_number}_txt_file_3',
                                        f'{cr_number}_txt_file_4',
                                        f'{cr_number}_txt_file_5',
                                        f'{cr_number}_txt_file_6',
                                        f'{cr_number + 1}_txt_file_1',
                                        f'{cr_number + 1}_txt_file_2',
                                        f'{cr_number + 1}_txt_file_3',
                                        f'{cr_number + 1}_txt_file_4',
                                        f'{cr_number + 1}_txt_file_5',
                                        f'{cr_number + 1}_txt_file_6'
                                        ]

        mock_convert_json_to_l3d_data_product.return_value = sentinel.l3d_data_product

        mock_set_version.return_value = sentinel.out_paths

        processor = GlowsProcessor(input_data_collection, input_metadata)
        actual_l3d_data_product, actual_l3d_txt_files, last_processed_cr = processor.process_l3d(mock_l3d_dependencies)

        self.assertEqual(actual_l3d_txt_files, sentinel.out_paths)
        self.assertEqual(last_processed_cr, cr_number + 1)

        self.assertEqual(2, mock_os.makedirs.call_count)
        mock_os.makedirs.assert_has_calls([
            call(PATH_TO_L3D_TOOLKIT / 'data_l3d', exist_ok=True),
            call(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt', exist_ok=True),
        ])

        mock_os.listdir.assert_called_once_with(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt')
        expected_dependencies = {
            'ancillary_files': mock_l3d_dependencies.ancillary_files,
            'external_files': mock_l3d_dependencies.external_files
        }

        mock_create_glows_l3c_json_file_from_cdf.assert_has_calls([
            call(sentinel.l3c_file_1), call(sentinel.l3c_file_2)
        ])

        mock_create_glows_l3b_json_file_from_cdf.assert_has_calls([
            call(sentinel.l3b_file_1), call(sentinel.l3b_file_2)
        ])

        expected_working_directory = Path(l3d.__file__).parent / 'science'

        self.assertEqual(3, mock_run.call_count)
        mock_run.assert_has_calls([
            call([sys.executable, './generate_l3d.py', f'{cr_number}', json.dumps(expected_dependencies)],
                 cwd=str(expected_working_directory), check=True,
                 capture_output=True, text=True),
            call([sys.executable, './generate_l3d.py', f'{cr_number + 1}', json.dumps(expected_dependencies)],
                 cwd=str(expected_working_directory),
                 check=True,
                 capture_output=True, text=True),
            call([sys.executable, './generate_l3d.py', f'{cr_number + 2}', json.dumps(expected_dependencies)],
                 cwd=str(expected_working_directory),
                 check=True,
                 capture_output=True, text=True),
        ])

        mock_get_parent_file_names_from_l3d_json.assert_called_once_with(expected_working_directory / 'data_l3d')

        mock_convert_json_to_l3d_data_product.assert_called_once_with(
            expected_working_directory / 'data_l3d' / f'imap_glows_l3d_solar-params-history_19470303-cr0{cr_number + 1}_v00.json',
            input_metadata,
            mock_get_parent_file_names_from_l3d_json.return_value)

        expected_l3d_txt_paths = [
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{cr_number + 1}_txt_file_1', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{cr_number + 1}_txt_file_2', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{cr_number + 1}_txt_file_3', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{cr_number + 1}_txt_file_4', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{cr_number + 1}_txt_file_5', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{cr_number + 1}_txt_file_6', ),
        ]

        self.assertEqual(sentinel.l3d_data_product, actual_l3d_data_product)

        mock_set_version.assert_has_calls([call(expected_l3d_txt_paths, 'v001')])

    @patch('imap_l3_processing.glows.glows_processor.set_version_on_txt_files')
    @patch('imap_l3_processing.glows.glows_processor.PATH_TO_L3D_TOOLKIT', get_test_data_path('glows/science'))
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3d_adds_parent_file_names_to_output(self, mock_spicepy, mock_run, mock_set_version):
        mock_spicepy.ktotal.return_value = 0
        l3b_path_1 = get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100422_v011.cdf')
        l3b_path_2 = get_test_data_path('glows/imap_glows_l3b_ion-rate-profile_20100519_v011.cdf')
        l3c_path_1 = get_test_data_path('glows/imap_glows_l3c_sw-profile_20100422_v011.cdf')
        l3c_path_2 = get_test_data_path('glows/imap_glows_l3c_sw-profile_20100519_v011.cdf')

        pipeline_settings_path = get_test_data_path(
            'glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json')
        speed_path = get_test_data_path('glows/imap_glows_plasma-speed-Legendre-2010a_v001.dat')
        p_dens_path = get_test_data_path('glows/imap_glows_proton-density-Legendre-2010a_v001.dat')
        uv_anis_path = get_test_data_path('glows/imap_glows_uv-anisotropy-2010a_v001.dat')
        phion_path = get_test_data_path('glows/imap_glows_photoion-2010a_v001.dat')
        lya_path = get_test_data_path('glows/imap_glows_lya-2010a_v001.dat')
        e_dens_path = get_test_data_path('glows/imap_glows_electron-density-2010a_v001.dat')

        lyman_alpha_composite_path = get_test_data_path('glows/lyman_alpha_composite.nc')

        l3b_file_paths = [l3b_path_1, l3b_path_2]
        l3c_file_paths = [l3c_path_1, l3c_path_2]
        ancillary_inputs = {
            'pipeline_settings': pipeline_settings_path,
            'WawHelioIon': {
                'speed': speed_path,
                'p-dens': p_dens_path,
                'uv-anis': uv_anis_path,
                'phion': phion_path,
                'lya': lya_path,
                'e-dens': e_dens_path
            }
        }
        external_inputs = {
            'lya_raw_data': lyman_alpha_composite_path
        }

        cr_number = 2095
        mock_run.side_effect = [CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number}'),
                                CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)]

        l3d_dependencies = GlowsL3DDependencies(l3b_file_paths=l3b_file_paths, l3c_file_paths=l3c_file_paths,
                                                ancillary_files=ancillary_inputs, external_files=external_inputs)

        mock_set_version.return_value = sentinel.out_paths

        input_metadata = Mock()
        input_metadata.version = 'v001'

        processor = GlowsProcessor(Mock(), input_metadata)
        actual_data_product, actual_l3d_txt_paths, last_processed_cr = processor.process_l3d(l3d_dependencies)
        self.assertEqual(last_processed_cr, cr_number)
        self.assertEqual(actual_l3d_txt_paths, sentinel.out_paths)

        expected_parent_file_names = [
            "imap_glows_plasma-speed-2010a_v003.dat",
            "imap_glows_proton-density-2010a_v003.dat",
            "imap_glows_uv-anisotropy-2010a_v003.dat",
            "imap_glows_photoion-2010a_v003.dat",
            "imap_glows_lya-2010a_v003.dat",
            "imap_glows_electron-density-2010a_v003.dat",
            "lyman_alpha_composite.nc",
            "imap_glows_l3b_ion-rate-profile_20100422_v011.cdf",
            "imap_glows_l3b_ion-rate-profile_20100519_v011.cdf",
            "imap_glows_l3c_sw-profile_20100422_v011.cdf",
            "imap_glows_l3c_sw-profile_20100519_v011.cdf"
        ]

        expected_l3d_txt_file_paths = [
            get_test_data_path('glows/science') / 'data_l3d_txt' / 'imap_glows_l3d_e-dens_19470303-cr02095_v00.dat',
            get_test_data_path('glows/science') / 'data_l3d_txt' / 'imap_glows_l3d_uv-anis_19470303-cr02095_v00.dat',
            get_test_data_path('glows/science') / 'data_l3d_txt' / 'imap_glows_l3d_phion_19470303-cr02095_v00.dat',
            get_test_data_path('glows/science') / 'data_l3d_txt' / 'imap_glows_l3d_p-dens_19470303-cr02095_v00.dat',
            get_test_data_path('glows/science') / 'data_l3d_txt' / 'imap_glows_l3d_lya_19470303-cr02095_v00.dat',
            get_test_data_path('glows/science') / 'data_l3d_txt' / 'imap_glows_l3d_speed_19470303-cr02095_v00.dat',
        ]

        self.assertCountEqual(expected_parent_file_names, actual_data_product.parent_file_names)
        self.assertCountEqual(expected_l3d_txt_file_paths, mock_set_version.call_args[0][0])
        self.assertEqual('v001', mock_set_version.call_args[0][1])

    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.glows_processor.set_version_on_txt_files')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.json')
    def test_process_l3d_returns_correctly_if_nothing_is_processed(self, mock_json, mock_run, _, __, ___):

        mock_json.load.return_value = {'start_cr': 2092}
        ancillary_files = {
            'pipeline_settings': get_test_data_path(
                "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
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

        mock_run.side_effect = [CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)]
        processor = GlowsProcessor(Mock(), Mock())

        actual_data_product, actual_text_files, actual_last_processed_crs = processor.process_l3d(l3d_dependencies)

        self.assertIsNone(actual_data_product)
        self.assertIsNone(actual_text_files)
        self.assertIsNone(actual_last_processed_crs)

    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.glows_processor.set_version_on_txt_files')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.json')
    def test_process_l3d_handles_unexpected_exception_from_science(self, mock_json,
                                                                   mock_convert_json_to_l3d,
                                                                   mock_run, mock_os, _, __, ___):
        ancillary_files = {
            'pipeline_settings': get_test_data_path(
                "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
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
        mock_json.load.return_value = {'start_cr': 2091}
        l3b_file_paths = []
        l3c_file_paths = []
        l3d_dependencies = GlowsL3DDependencies(ancillary_files=ancillary_files,
                                                external_files=external_files,
                                                l3b_file_paths=l3b_file_paths,
                                                l3c_file_paths=l3c_file_paths)

        processing_input_collection = Mock()
        input_metadata = InputMetadata('glows', "l3d", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='solar-params-history')

        mock_os.listdir.return_value = ['2096_txt_file_1',
                                        '2096_txt_file_2',
                                        '2096_txt_file_3',
                                        '2096_txt_file_4',
                                        '2096_txt_file_5',
                                        '2096_txt_file_6'
                                        ]

        expected_cr = 2096

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

        mock_run.side_effect = [CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {expected_cr}'),
                                CalledProcessError(cmd="", returncode=1, stderr=unexpected_exception)]

        processor = GlowsProcessor(processing_input_collection, input_metadata)

        with self.assertRaises(Exception) as context:
            processor.process_l3d(l3d_dependencies)
        self.assertEqual(unexpected_exception, str(context.exception))

        mock_convert_json_to_l3d.assert_not_called()

    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.json')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3DDependencies.fetch_dependencies')
    def test_process_l3d_does_not_save_if_nothing_processed(self, mock_fetch_dependencies, mock_json, mock_save_data,
                                                            mock_run, mock_os, _, __,
                                                            ___):
        processing_input_collection = Mock()
        input_metadata = InputMetadata('glows', "l3d", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001', descriptor='solar-params-history')

        mock_os.listdir.return_value = ['2096_txt_file_1',
                                        '2096_txt_file_2',
                                        '2096_txt_file_3',
                                        '2096_txt_file_4',
                                        '2096_txt_file_5',
                                        '2096_txt_file_6'
                                        ]

        mock_run.side_effect = [CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)]

        mock_json.load.return_value = {'start_cr': 2091}
        mock_deps = MagicMock()
        mock_fetch_dependencies.return_value = mock_deps

        processor = GlowsProcessor(processing_input_collection, input_metadata)
        products = processor.process()
        mock_save_data.assert_not_called()
        self.assertEqual([], products)

    @patch("imap_l3_processing.glows.l3d.glows_l3d_dependencies.download")
    @patch("imap_l3_processing.glows.l3d.glows_l3d_dependencies.query")
    def test_process_glows_l3d_drift(self, mock_query, mock_download):
        expected_txt_filenames = ["imap_glows_l3d_e-dens_19470303-cr02096_v004.dat",
                                  "imap_glows_l3d_lya_19470303-cr02096_v004.dat",
                                  "imap_glows_l3d_p-dens_19470303-cr02096_v004.dat",
                                  "imap_glows_l3d_phion_19470303-cr02096_v004.dat",
                                  "imap_glows_l3d_speed_19470303-cr02096_v004.dat",
                                  "imap_glows_l3d_uv-anis_19470303-cr02096_v004.dat"]
        expected_cdf_filename = "imap_glows_l3d_solar-hist_19470303-cr02096_v004.cdf"

        glows_output_file_path = config["DATA_DIR"] / "imap" / "glows" / "l3d" / "1947" / "03"
        if os.path.exists(glows_output_file_path / expected_cdf_filename): os.remove(
            glows_output_file_path / expected_cdf_filename)

        input_metadata = InputMetadata("glows", "l3d", datetime(2010, 1, 1), datetime(2010, 1, 2), version="v004",
                                       descriptor="solar-hist")
        processing_input_collection = ProcessingInputCollection(
            ScienceInput("imap_glows_l3b_ion-rate-profile_20100422_v013.cdf"),
            ScienceInput("imap_glows_l3c_sw-profile_20100422_v012.cdf"),
            AncillaryInput("imap_glows_plasma-speed-2010a_20100101_v003.dat"),
            AncillaryInput("imap_glows_proton-density-2010a_20100101_v003.dat"),
            AncillaryInput("imap_glows_uv-anisotropy-2010a_20100101_v003.dat"),
            AncillaryInput("imap_glows_photoion-2010a_20100101_v003.dat"),
            AncillaryInput("imap_glows_electron-density-2010a_20100101_v003.dat"),
            AncillaryInput("imap_glows_lya-2010a_20100101_v003.dat"),
            AncillaryInput("imap_glows_l3b-archive_20100422_v011.zip"),
            AncillaryInput("imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
        )

        mock_download.side_effect = [
            get_test_data_path("glows/l3d_drift_test/imap_glows_plasma-speed-2010a_20100101_v003.dat"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_proton-density-2010a_20100101_v003.dat"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_uv-anisotropy-2010a_20100101_v003.dat"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_photoion-2010a_20100101_v003.dat"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_lya-2010a_20100101_v003.dat"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_electron-density-2010a_20100101_v003.dat"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_l3b-archive_20100519_v011.zip"),

            get_test_data_path("glows/l3d_drift_test/imap_glows_l3b_ion-rate-profile_20100422_v013.cdf"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_l3b_ion-rate-profile_20100519_v013.cdf"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_l3c_sw-profile_20100422_v012.cdf"),
            get_test_data_path("glows/l3d_drift_test/imap_glows_l3c_sw-profile_20100519_v012.cdf"),
        ]
        mock_query.side_effect = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]

        processor = GlowsProcessor(processing_input_collection, input_metadata)
        products = processor.process()

        self.assertEqual(7, len(products))

        self.assertIn(glows_output_file_path / expected_cdf_filename, products)
        for file in expected_txt_filenames:
            self.assertIn(PATH_TO_L3D_TOOLKIT / "data_l3d_txt" / file, products)

        test_cases = [
            (expected_txt_filenames[0], "electron_density", 847, [1947.167990000000, -1, 1250.5],
             [2010.343068904109, 5.764092012146565e+00, 2096.5]),
            (expected_txt_filenames[1], "lyman_alpha", 847, [1947.167990000000, 6.199757637899209e+11, 1250.5],
             [2010.343068904109, 3.976427956000000e+11, 2096.5]),
            (expected_txt_filenames[2], "proton_density", 847,
             [1.94716799e+03, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              1.25050000e+03, -1.00000000e+00],
             [2.01034307e+03, 3.38901240e+00, 3.35060190e+00, 3.25146686e+00, 3.69331370e+00, 4.69194170e+00,
              5.59545144e+00, 6.38779612e+00, 6.79219078e+00, 7.44907664e+00, 7.92968821e+00, 7.69531028e+00,
              7.31272442e+00, 6.37330975e+00, 5.46350990e+00, 4.50035049e+00, 3.33430038e+00, 2.77020067e+00,
              2.71814660e+00, 2.71783185e+00, 2.09650000e+03, 1.00000000e+04]),
            (expected_txt_filenames[3], "phion", 847, [1947.167990000000, 1.830438692701064e-07, 1250.5],
             [2010.343068904109, 1.049049881153713e-07, 2096.5]),
            (expected_txt_filenames[4], "plasma_speed", 847,
             [1.94716799e+03, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              1.25050000e+03, -1],
             [2010.3430689, 419.92192821, 422.74228283, 430.19209339, 398.7140042, 343.04238403, 304.35629938,
              275.90330762, 261.5337126, 241.86406135, 227.84491603, 235.56448958, 247.05106709, 275.35344769,
              307.8808557, 350.7627346, 422.76124576, 470.9791981, 476.1416525, 476.18197781, 2096.5, 20000.]),
            (expected_txt_filenames[5], "uv_anisotropy", 847,
             [1.94716799e+03, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
              1.25050000e+03, -1],
             [2.01034307e+03, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2.09650000e+03, 1.00000000e+03]),
        ]

        with CDF(str(glows_output_file_path / expected_cdf_filename)) as actual_cdf:
            for filename, cdf_var_name, length_of_data, first_line, last_line in test_cases:
                with self.subTest(msg=filename):
                    actual = np.loadtxt(PATH_TO_L3D_TOOLKIT / "data_l3d_txt" / filename)

                    self.assertEqual(length_of_data, len(actual))
                    self.assertEqual(length_of_data, len(actual_cdf[cdf_var_name][...]))

                    np.testing.assert_allclose(actual[0], first_line)

                    np.testing.assert_allclose(actual[-1], last_line)


if __name__ == '__main__':
    unittest.main()
