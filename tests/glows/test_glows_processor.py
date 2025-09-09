import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import patch, Mock, sentinel, call, MagicMock
from zipfile import ZIP_DEFLATED

import imap_data_access
import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows import l3d
from imap_l3_processing.glows.descriptors import GLOWS_L3A_DESCRIPTOR
from imap_l3_processing.glows.glows_processor import GlowsProcessor, process_l3d, process_l3e, process_l3bc, \
    process_l3e_ul, process_l3e_hi, process_l3e_lo
from imap_l3_processing.glows.l3a.utils import create_glows_l3a_dictionary_from_cdf, create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3bc.cannot_process_carrington_rotation_error import CannotProcessCarringtonRotationError
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.glows_l3bc_initializer import GlowsL3BCInitializerData
from imap_l3_processing.glows.l3bc.models import ExternalDependencies
from imap_l3_processing.glows.l3d.glows_l3d_dependencies import GlowsL3DDependencies
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_dependencies import GlowsL3EDependencies
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.glows.l3e.glows_l3e_initializer import GlowsL3EInitializer, GlowsL3EInitializerOutput
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.glows.l3e.glows_l3e_utils import GlowsL3eRepointings, determine_call_args_for_l3e_executable
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import save_data
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

    def setUp(self):
        self.l3bc_initializer_patcher = patch(
            "imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer.get_crs_to_process")
        self.mock_l3bc_initializer = self.l3bc_initializer_patcher.start()

        self.mock_external_deps = Mock()
        self.mock_l3bc_initializer.return_value = GlowsL3BCInitializerData(
            external_dependencies=self.mock_external_deps,
            l3bc_dependencies=[],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=Path("imap_2001_052_001.repoint.csv"),
        )

        self.l3d_initializer_patcher = patch(
            "imap_l3_processing.glows.glows_processor.GlowsL3DInitializer.should_process_l3d")
        self.mock_l3d_initializer = self.l3d_initializer_patcher.start()
        self.mock_l3d_initializer.return_value = None

        self.mock_l3e_initializer_patcher = patch("imap_l3_processing.glows.glows_processor.GlowsL3EInitializer.get_repointings_to_process")
        self.mock_l3e_initializer = self.mock_l3e_initializer_patcher.start()
        self.mock_l3e_initializer.return_value = GlowsL3EInitializerOutput(
            dependencies=Mock(),
            repointings=GlowsL3eRepointings(
                repointing_numbers=[],
                hi_90_repointings={},
                hi_45_repointings={},
                lo_repointings={},
                ultra_repointings={},
            )
        )

    def tearDown(self):
        self.l3bc_initializer_patcher.stop()
        self.l3d_initializer_patcher.stop()
        self.mock_l3e_initializer_patcher.stop()

        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3b'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3b')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3c'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3c')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3d'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3d')
        if os.path.exists(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt'): shutil.rmtree(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt')

    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3ADependencies')
    @patch('imap_l3_processing.glows.glows_processor.L3aData')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.utils.spiceypy')
    def test_processor_handles_l3a(self, mock_spiceypy, mock_save_data, mock_l3a_data,
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
            mock_l3a_data.return_value.data = json.load(f)
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
            mock_l3a_data.return_value.data, replace(input_metadata, descriptor=GLOWS_L3A_DESCRIPTOR))
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

    @patch("imap_l3_processing.utils.spiceypy")
    @patch("imap_l3_processing.glows.glows_processor.save_data")
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind')
    @patch('imap_l3_processing.glows.glows_processor.filter_l3a_files')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer')
    def test_process_l3bc(self, mock_glows_l3bc_initializer, mock_archive_dependencies, mock_generate_l3bc,
                          mock_filter_bad_days,
                          mock_l3c_model_class, mock_l3b_model_class, mock_save_data,
                          mock_spiceypy):
        mock_spiceypy.ktotal.return_value = 1
        mock_spiceypy.kdata.return_value = ['kernel_1', 'type', 'source', 'handle']

        first_cr_number_to_process = 2091
        second_cr_number_to_process = 2092

        external_deps = ExternalDependencies(f107_index_file_path=sentinel.f107_index_file_path,
                                             lyman_alpha_path=sentinel.lyman_alpha_path,
                                             omni2_data_path=sentinel.omni2_data_path)

        mock_archive_dependencies.side_effect = [Path("path1.zip"), Path("path2.zip")]

        l3a_data_1 = {"filename": "some_file"}
        first_dependency = GlowsL3BCDependencies(l3a_data=[l3a_data_1],
                                                 external_files=sentinel.external_files_1,
                                                 ancillary_files={
                                                     'bad_days_list': sentinel.bad_days_list_1,
                                                 },
                                                 carrington_rotation_number=first_cr_number_to_process,
                                                 version=1,
                                                 start_date=Mock(),
                                                 end_date=Mock(),
                                                 repointing_file_path=sentinel.repointing_file_path, )

        l3a_data_2 = {"filename": "some_file"}
        second_dependency = GlowsL3BCDependencies(l3a_data=[l3a_data_2],
                                                  external_files=sentinel.external_files_2,
                                                  ancillary_files={
                                                      'bad_days_list': sentinel.bad_days_list_2,
                                                  },
                                                  carrington_rotation_number=second_cr_number_to_process,
                                                  version=2,
                                                  start_date=Mock(),
                                                  end_date=Mock(),
                                                  repointing_file_path=sentinel.repointing_file_path)

        mock_generate_l3bc.side_effect = [(sentinel.l3b_data_1, sentinel.l3c_data_1),
                                          (sentinel.l3b_data_2, sentinel.l3c_data_2)]
        mock_filter_bad_days.side_effect = [sentinel.filtered_days_1, sentinel.filtered_days_2]

        l3b_model_1 = Mock(parent_file_names=["file1"])
        l3b_model_2 = Mock(parent_file_names=["file2"])
        l3c_model_1 = Mock(parent_file_names=["file3"])
        l3c_model_2 = Mock(parent_file_names=["file4"])
        mock_l3b_model_class.from_instrument_team_dictionary.side_effect = [l3b_model_1, l3b_model_2]
        mock_l3c_model_class.from_instrument_team_dictionary.side_effect = [l3c_model_1, l3c_model_2]
        mock_save_data.side_effect = [Path("path/to/l3b_file_1.cdf"),
                                      Path("path/to/l3c_file_1.cdf"),
                                      Path("path/to/l3b_file_2.cdf"),
                                      Path("path/to/l3c_file_2.cdf")]

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v02')

        mock_glows_l3bc_initializer.get_crs_to_process.return_value = GlowsL3BCInitializerData(
            external_dependencies=external_deps,
            l3bc_dependencies=[first_dependency, second_dependency],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=input_metadata)
        products = processor.process()

        self.assertEqual([Path("path/to/l3b_file_1.cdf"),
                          Path("path/to/l3c_file_1.cdf"),
                          Path("path1.zip"),
                          Path("path/to/l3b_file_2.cdf"),
                          Path("path/to/l3c_file_2.cdf"),
                          Path("path2.zip")
                          ], products)

        dependencies_with_filtered_list_1 = replace(first_dependency, l3a_data=sentinel.filtered_days_1)
        dependencies_with_filtered_list_2 = replace(second_dependency, l3a_data=sentinel.filtered_days_2)

        mock_filter_bad_days.assert_has_calls(
            [call([l3a_data_1], sentinel.bad_days_list_1, first_cr_number_to_process),
             call([l3a_data_2], sentinel.bad_days_list_2, second_cr_number_to_process)])

        mock_generate_l3bc.assert_has_calls(
            [call(dependencies_with_filtered_list_1), call(dependencies_with_filtered_list_2)])

        expected_l3b_metadata_1 = InputMetadata("glows", "l3b", first_dependency.start_date,
                                                first_dependency.end_date, 'v001', "ion-rate-profile")
        expected_l3b_metadata_2 = InputMetadata("glows", "l3b", second_dependency.start_date,
                                                second_dependency.end_date, 'v002', "ion-rate-profile")
        expected_l3c_metadata_1 = InputMetadata("glows", "l3c", first_dependency.start_date,
                                                first_dependency.end_date, 'v001', "sw-profile")
        expected_l3c_metadata_2 = InputMetadata("glows", "l3c", second_dependency.start_date,
                                                second_dependency.end_date, 'v002', "sw-profile")
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
            call(l3bc_deps=first_dependency, external_dependencies=external_deps),
            call(l3bc_deps=second_dependency, external_dependencies=external_deps),
        ])

        self.assertEqual(["file1", "path1.zip", "kernel_1"], l3b_model_1.parent_file_names)
        self.assertEqual(["file2", "path2.zip", "kernel_1"], l3b_model_2.parent_file_names)
        self.assertEqual(["file3", "path1.zip", "l3b_file_1.cdf", "kernel_1", ], l3c_model_1.parent_file_names)
        self.assertEqual(["file4", "path2.zip", "l3b_file_2.cdf", "kernel_1", ], l3c_model_2.parent_file_names)

    @patch('imap_l3_processing.glows.glows_processor.filter_l3a_files')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3BIonizationRate.from_instrument_team_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3CSolarWind.from_instrument_team_dictionary')
    @patch('imap_l3_processing.glows.glows_processor.generate_l3bc')
    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3BCInitializer")
    def test_process_l3bc_catches_no_data_error_and_continues(self, mock_glows_initializer_class, mock_save_data,
                                                           mock_archive_dependencies, mock_generate_l3bc,
                                                           mock_l3c_from_instrument_team_dictionary,
                                                           mock_l3b_from_instrument_team_dictionary, _):
        bc_dependencies_1 = GlowsL3BCDependencies(version=1,
                                                  carrington_rotation_number=2096,
                                                  start_date=datetime(year=2021, month=1, day=1),
                                                  end_date=datetime(year=2021, month=1, day=1),
                                                  l3a_data=[],
                                                  external_files=defaultdict(Mock),
                                                  ancillary_files=defaultdict(Mock),
                                                  repointing_file_path=sentinel.repointing_file_path
                                                  )
        bc_dependencies_2 = GlowsL3BCDependencies(version=1,
                                                  carrington_rotation_number=2096,
                                                  start_date=datetime(year=2021, month=1, day=1),
                                                  end_date=datetime(year=2021, month=1, day=1),
                                                  l3a_data=[],
                                                  external_files=defaultdict(Mock),
                                                  ancillary_files=defaultdict(Mock),
                                                  repointing_file_path=sentinel.repointing_file_path
                                                  )
        external_dependencies = ExternalDependencies(None, None, None)
        mock_glows_initializer_class.get_crs_to_process.return_value = GlowsL3BCInitializerData(
            external_dependencies=external_dependencies,
            l3bc_dependencies=[bc_dependencies_1, bc_dependencies_2],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        zip_file_1 = "imap_glows_archive_20210101_v001.zip"
        zip_file_2 = "imap_glows_archive_20210102_v001.zip"

        mock_archive_dependencies.side_effect = [zip_file_1, zip_file_2]

        l3b_path = Path("path_to/l3b")
        l3c_path = Path("path_to/l3c")
        mock_save_data.side_effect = [l3b_path, l3c_path]

        input_metadata = InputMetadata('glows', "l3b", datetime(2024, 10, 7, 10, 00, 00),
                                       datetime(2024, 10, 8, 10, 00, 00),
                                       'v001')

        processor = GlowsProcessor(Mock(), input_metadata)

        mock_generate_l3bc.side_effect = [
            CannotProcessCarringtonRotationError("All days for Carrington Rotation are in a bad season."),
            ({}, {})
        ]

        products = processor.process()

        mock_l3b_from_instrument_team_dictionary.assert_called_once()
        mock_l3c_from_instrument_team_dictionary.assert_called_once()

        mock_archive_dependencies.assert_has_calls([
            call(l3bc_deps=bc_dependencies_1, external_dependencies=external_dependencies),
            call(l3bc_deps=bc_dependencies_2, external_dependencies=external_dependencies)
        ])

        self.assertEqual(2, mock_save_data.call_count)
        mock_save_data.assert_has_calls([
            call(mock_l3b_from_instrument_team_dictionary.return_value, cr_number=2096),
            call(mock_l3c_from_instrument_team_dictionary.return_value, cr_number=2096),
        ])

        self.assertEqual(3, len(products))
        self.assertEqual([
            l3b_path,
            l3c_path,
            zip_file_2
        ], products)

    @patch('imap_l3_processing.glows.glows_processor.GlowsProcessor.archive_dependencies')
    def test_process_l3bc_all_data_in_bad_season_catches_exception_and_continues(self, _):
        l3a_data_folder_path = get_test_data_path('glows/l3a_products')
        l3a_data = [
            create_glows_l3a_dictionary_from_cdf(
                l3a_data_folder_path / 'imap_glows_l3a_hist_20100201-repoint00032_v001.cdf')]

        external_dependencies = ExternalDependencies(
            f107_index_file_path=get_test_instrument_team_data_path('glows/f107_fluxtable.txt'),
            omni2_data_path=get_test_instrument_team_data_path('glows/omni_2010.dat'),
            lyman_alpha_path=None
        )

        l3bc_deps = GlowsL3BCDependencies(
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
            carrington_rotation_number=1,
            start_date=Mock(), end_date=Mock(), version=1,
            repointing_file_path=sentinel.repointing_file_path
        )

        initializer_data = GlowsL3BCInitializerData(
            external_dependencies=external_dependencies,
            l3bc_dependencies=[l3bc_deps],
            l3bs_by_cr={},
            l3cs_by_cr={},
            repoint_file_path=sentinel.repoint_file_path
        )

        processor = GlowsProcessor(dependencies=Mock(), input_metadata=Mock())
        processor_output = process_l3bc(processor, initializer_data)

        self.assertEqual([], processor_output.data_products)


    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3b_json_file_from_cdf")
    @patch("imap_l3_processing.glows.glows_processor.create_glows_l3c_json_file_from_cdf")
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.shutil')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3DInitializer")
    @patch("imap_l3_processing.glows.glows_processor.read_pipeline_settings")
    def test_process_l3d(self, mock_read_pipeline_settings, mock_glows_l3d_initializer, mock_os, mock_shutil, mock_run,
                         mock_convert_json_to_l3d_data_product, mock_get_parent_file_names_from_l3d_json,
                         mock_rename_l3d, mock_save_data, mock_convert_l3c_to_json, mock_convert_l3b_to_json):

        cr_number = 2092
        mock_read_pipeline_settings.return_value = {'start_cr': cr_number}

        glows_l3d_dependencies = GlowsL3DDependencies(
            external_files={
                'lya_raw_data': Path('path/to/lya'),
            },
            ancillary_files={
                'pipeline_settings':
                    Path('glows/imap_glows_pipeline-settings-l3bcde_20250514_v004.json'),
                'WawHelioIon': {
                    'speed': Path('path/to/speed'),
                    'p-dens': Path('path/to/p-dens'),
                    'uv-anis': Path('path/to/uv-anis'),
                    'phion': Path('path/to/phion'),
                    'lya': Path('path/to/lya'),
                    'e-dens': Path('path/to/e-dens')
                }
            },
            l3b_file_paths=[sentinel.l3b_file_1, sentinel.l3b_file_2],
            l3c_file_paths=[sentinel.l3c_file_1, sentinel.l3c_file_2]
        )

        old_l3d = Path('imap_glows_l3d_solar-hist_19470303-cr02090_v001.cdf')
        l3d_output_version = 5
        mock_glows_l3d_initializer.should_process_l3d.return_value = (
            l3d_output_version, glows_l3d_dependencies, old_l3d)

        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number}'),
            CompletedProcess(args=[], returncode=0, stdout=f'Processed CR= {cr_number + 1}'),
            CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)
        ]

        mock_os.listdir.return_value = [
            f'{cr_number}_txt_file_1',
            f'{cr_number}_txt_file_2',
            f'{cr_number + 1}_txt_file_1',
            f'{cr_number + 1}_txt_file_2',
        ]

        mock_convert_json_to_l3d_data_product.return_value = sentinel.l3d_data_product

        mock_rename_l3d.return_value = [
            Path("imap_glows_e-dens_19470303_20100101_v000.dat"),
            Path("imap_glows_lya_19470303_20100101_v000.dat")
        ]

        mock_save_data.return_value = Path("l3d_cdf.cdf")

        processor = GlowsProcessor(Mock(), Mock(data_level="l3b"))
        products = processor.process()

        mock_convert_l3b_to_json.assert_has_calls([call(sentinel.l3b_file_1), call(sentinel.l3b_file_2)])
        mock_convert_l3c_to_json.assert_has_calls([call(sentinel.l3c_file_1), call(sentinel.l3c_file_2)])

        self.assertEqual([
            Path("imap_glows_e-dens_19470303_20100101_v000.dat"),
            Path("imap_glows_lya_19470303_20100101_v000.dat"),
            Path("l3d_cdf.cdf")
        ], products)

        self.assertEqual(2, mock_shutil.copy.call_count)
        mock_shutil.copy.assert_has_calls([
            call(Path("imap_glows_e-dens_19470303_20100101_v000.dat"),
                 imap_data_access.config["DATA_DIR"] / "imap/ancillary/glows/imap_glows_e-dens_19470303_20100101_v000.dat"),
            call(Path("imap_glows_lya_19470303_20100101_v000.dat"),
                 imap_data_access.config["DATA_DIR"] / "imap/ancillary/glows/imap_glows_lya_19470303_20100101_v000.dat")
        ])

        self.assertEqual(2, mock_os.makedirs.call_count)
        mock_os.makedirs.assert_has_calls([
            call(PATH_TO_L3D_TOOLKIT / 'data_l3d', exist_ok=True),
            call(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt', exist_ok=True),
        ])

        expected_dependencies = json.dumps({
            'external_files': {
                'lya_raw_data': str(Path('path/to/lya')),
            },
            'ancillary_files': {
                'pipeline_settings':
                    str(Path('glows/imap_glows_pipeline-settings-l3bcde_20250514_v004.json')),
                'WawHelioIon': {
                    'speed': str(Path('path/to/speed')),
                    'p-dens': str(Path('path/to/p-dens')),
                    'uv-anis': str(Path('path/to/uv-anis')),
                    'phion': str(Path('path/to/phion')),
                    'lya': str(Path('path/to/lya')),
                    'e-dens': str(Path('path/to/e-dens'))
                }
            }
        })

        expected_working_directory = Path(l3d.__file__).parent / 'science'

        self.assertEqual(3, mock_run.call_count)
        mock_run.assert_has_calls([
            call([sys.executable, './generate_l3d.py', f'{cr_number}', expected_dependencies],
                 cwd=str(expected_working_directory), check=True,
                 capture_output=True, text=True),
            call([sys.executable, './generate_l3d.py', f'{cr_number + 1}', expected_dependencies],
                 cwd=str(expected_working_directory),
                 check=True,
                 capture_output=True, text=True),
            call([sys.executable, './generate_l3d.py', f'{cr_number + 2}', expected_dependencies],
                 cwd=str(expected_working_directory),
                 check=True,
                 capture_output=True, text=True),
        ])

        mock_os.listdir.assert_called_once_with(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt')

        expected_output_cr = cr_number + 1
        expected_l3d_txt_paths = [
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{expected_output_cr}_txt_file_1', ),
            Path(PATH_TO_L3D_TOOLKIT / 'data_l3d_txt' / f'{expected_output_cr}_txt_file_2', ),
        ]
        mock_rename_l3d.assert_has_calls([call(expected_l3d_txt_paths, f'v00{l3d_output_version}')])

        mock_get_parent_file_names_from_l3d_json.assert_called_once_with(expected_working_directory / 'data_l3d')

        expected_data_product_metadata = InputMetadata(instrument="glows", data_level="l3d", descriptor="solar-hist",
                                                       start_date=datetime(1947, 3, 3),
                                                       end_date=datetime(1947, 3, 3),
                                                       version=f"v00{l3d_output_version}")

        mock_convert_json_to_l3d_data_product.assert_called_once_with(
            expected_working_directory / 'data_l3d' / f'imap_glows_l3d_solar-params-history_19470303-cr0{expected_output_cr}_v00.json',
            expected_data_product_metadata,
            mock_get_parent_file_names_from_l3d_json.return_value)

        mock_save_data.assert_called_once_with(sentinel.l3d_data_product, cr_number=expected_output_cr)

    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.PATH_TO_L3D_TOOLKIT', get_test_data_path('glows/science'))
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3d_adds_parent_file_names_to_output(self, mock_spicepy, mock_run, mock_rename_l3d_text_output,
                                                          mock_save_data):
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

        glows_l3d_output = process_l3d(l3d_dependencies, 1)

        [save_data_call_args] = mock_save_data.call_args_list
        actual_data_product = save_data_call_args.args[0]

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
        self.assertCountEqual(expected_l3d_txt_file_paths, mock_rename_l3d_text_output.call_args[0][0])
        self.assertEqual('v001', mock_rename_l3d_text_output.call_args[0][1])

    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.read_pipeline_settings')
    def test_process_l3d_returns_correctly_if_nothing_is_processed(self, mock_read_pipeline_settings, mock_run, _, __):

        mock_read_pipeline_settings.return_value = {'start_cr': 2092}
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

        processor_return = process_l3d(l3d_dependencies, 1)

        self.assertIsNone(processor_return)

    @patch('imap_l3_processing.glows.glows_processor.rename_l3d_text_outputs')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.read_pipeline_settings')
    def test_process_l3d_handles_unexpected_exception_from_science(self, mock_read_pipeline_settings,
                                                                   mock_convert_json_to_l3d,
                                                                   mock_run, mock_os, _, __):
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
        mock_read_pipeline_settings.return_value = {'start_cr': 2091}
        l3b_file_paths = []
        l3c_file_paths = []
        l3d_dependencies = GlowsL3DDependencies(ancillary_files=ancillary_files,
                                                external_files=external_files,
                                                l3b_file_paths=l3b_file_paths,
                                                l3c_file_paths=l3c_file_paths)

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

        with self.assertRaises(Exception) as context:
            process_l3d(l3d_dependencies, 1)
        self.assertEqual(unexpected_exception, str(context.exception))

        mock_convert_json_to_l3d.assert_not_called()

    @patch('imap_l3_processing.glows.glows_processor.convert_json_to_l3d_data_product')
    @patch('imap_l3_processing.glows.glows_processor.get_parent_file_names_from_l3d_json')
    @patch('imap_l3_processing.glows.glows_processor.os')
    @patch('imap_l3_processing.glows.glows_processor.run')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch('imap_l3_processing.glows.glows_processor.read_pipeline_settings')
    @patch('imap_l3_processing.glows.glows_processor.GlowsL3DInitializer.should_process_l3d')
    def test_process_l3d_does_not_save_if_nothing_processed(self, mock_should_process_l3d, mock_read_pipeline_settings,
                                                            mock_save_data,
                                                            mock_run, mock_os, _, __,
                                                            ):

        l3d_dependencies = MagicMock()
        mock_should_process_l3d.return_value = sentinel.l3d_version, l3d_dependencies

        mock_os.listdir.return_value = ['2096_txt_file_1',
                                        '2096_txt_file_2',
                                        '2096_txt_file_3',
                                        '2096_txt_file_4',
                                        '2096_txt_file_5',
                                        '2096_txt_file_6'
                                        ]

        mock_run.side_effect = [CalledProcessError(cmd="", returncode=1, stderr=self.ran_out_of_l3b_exception)]

        mock_read_pipeline_settings.load.return_value = {'start_cr': 2091}

        products = process_l3d(dependencies=l3d_dependencies, version=1)

        mock_run.assert_called_once()

        mock_save_data.assert_not_called()
        self.assertEqual(None, products)

    @patch('imap_l3_processing.glows.glows_processor.shutil')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    def test_process_glows_l3d_drift(self, mock_save_data, _ ):
        lyman_alpha_composite = get_test_data_path("glows/l3d_drift_test/lyman_alpha_composite.nc")

        l3b_1 = get_test_data_path("glows/l3d_drift_test/imap_glows_l3b_ion-rate-profile_20100422_v013.cdf")
        l3b_2 = get_test_data_path("glows/l3d_drift_test/imap_glows_l3b_ion-rate-profile_20100519_v013.cdf")
        l3c_1 = get_test_data_path("glows/l3d_drift_test/imap_glows_l3c_sw-profile_20100422_v012.cdf")
        l3c_2 = get_test_data_path("glows/l3d_drift_test/imap_glows_l3c_sw-profile_20100519_v012.cdf")

        plasma_speed_legendre = get_test_data_path(
            "glows/l3d_drift_test/imap_glows_plasma-speed-2010a_20100101_v003.dat")
        proton_density_legendre = get_test_data_path(
            "glows/l3d_drift_test/imap_glows_proton-density-2010a_20100101_v003.dat")
        uv_anisotropy = get_test_data_path("glows/l3d_drift_test/imap_glows_uv-anisotropy-2010a_20100101_v003.dat")
        photoion = get_test_data_path("glows/l3d_drift_test/imap_glows_photoion-2010a_20100101_v003.dat")
        lya_2010a = get_test_data_path("glows/l3d_drift_test/imap_glows_lya-2010a_20100101_v003.dat")
        electron_density = get_test_data_path(
            "glows/l3d_drift_test/imap_glows_electron-density-2010a_20100101_v003.dat")
        pipeline_settings = get_test_data_path(
            "glows/l3d_drift_test/imap_glows_pipeline-settings-l3bcde_20100101_v006.json")

        l3d_dependencies = GlowsL3DDependencies(
            external_files={
                'lya_raw_data': lyman_alpha_composite,
            },
            ancillary_files={
                'pipeline_settings': pipeline_settings,
                'WawHelioIon': {
                    'speed': plasma_speed_legendre,
                    'p-dens': proton_density_legendre,
                    'uv-anis': uv_anisotropy,
                    'phion': photoion,
                    'lya': lya_2010a,
                    'e-dens': electron_density
                }
            },
            l3b_file_paths=[l3b_1, l3b_2],
            l3c_file_paths=[l3c_1, l3c_2],
        )

        version = 4
        glows_l3d_output = process_l3d(l3d_dependencies, version)

        expected_txt_filenames = ["imap_glows_e-dens_19470303_20100506_v004.dat",
                                  "imap_glows_lya_19470303_20100506_v004.dat",
                                  "imap_glows_p-dens_19470303_20100506_v004.dat",
                                  "imap_glows_phion_19470303_20100506_v004.dat",
                                  "imap_glows_speed_19470303_20100506_v004.dat",
                                  "imap_glows_uv-anis_19470303_20100506_v004.dat"]
        for file in expected_txt_filenames:
            self.assertIn(PATH_TO_L3D_TOOLKIT / "data_l3d_txt" / file, glows_l3d_output.l3d_text_file_paths)

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

        [save_data_call_args] = mock_save_data.call_args_list
        l3d_data_product = save_data_call_args.args[0]

        expected_cdf_filename = "imap_glows_l3d_solar-hist_19470303-cr02096_v004.cdf"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            save_data(l3d_data_product, cr_number=2096, folder_path=tmp_dir)

            with CDF(str(tmp_dir / expected_cdf_filename)) as actual_cdf:
                for filename, cdf_var_name, length_of_data, first_line, last_line in test_cases:
                    with self.subTest(msg=filename):
                        actual = np.loadtxt(PATH_TO_L3D_TOOLKIT / "data_l3d_txt" / filename)

                        self.assertEqual(length_of_data, len(actual))
                        self.assertEqual(length_of_data, len(actual_cdf[cdf_var_name][...]))

                        np.testing.assert_allclose(actual[0], first_line)
                        np.testing.assert_allclose(actual[-1], last_line)

    @patch('imap_l3_processing.glows.glows_processor.imap_data_access.download')
    @patch('imap_l3_processing.glows.glows_processor.spiceypy')
    @patch('imap_l3_processing.glows.glows_processor.find_l3e_spice_kernels_for_time_range')
    @patch('imap_l3_processing.glows.glows_processor.get_pointing_date_range')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_hi')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_lo')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul')
    def test_process_l3e(self, mock_process_ultra, mock_process_lo, mock_process_hi, mock_get_pointing_date_range,
                         mock_find_l3e_spice_kernels_for_time_range, mock_spiceypy, mock_download):
        mock_process_hi.side_effect = [
            [Path('path/to/first_hi_l3e')],
            [Path('path/to/second_hi_l3e')],
        ]
        mock_process_lo.return_value = [Path('path/to/lo_l3e')]
        mock_process_ultra.return_value = [Path('path/to/ultra_l3e')]

        expected_l3e_products = [
            Path('path/to/lo_l3e'),
            Path('path/to/first_hi_l3e'), Path('path/to/second_hi_l3e'),
            Path('path/to/ultra_l3e')
        ]

        start_epoch = datetime(2020, 1, 1)
        end_epoch = datetime(2020, 1, 2)
        epoch_delta = timedelta(hours=12)
        mock_get_pointing_date_range.return_value = (start_epoch, end_epoch)

        mock_dependencies = Mock(elongation={"2020025": sentinel.elongation})

        mock_dependencies.get_hi_parents.return_value = ["hi_ancillary.dat"]
        mock_dependencies.get_lo_parents.return_value = ["lo_ancillary.dat"]
        mock_dependencies.get_ul_parents.return_value = ["ul_ancillary.dat"]

        mock_find_l3e_spice_kernels_for_time_range.return_value = ["spice_kernel_1.tf"]

        initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_dependencies,
            repointings=GlowsL3eRepointings(
                repointing_numbers=[25],
                hi_90_repointings={25: 1},
                hi_45_repointings={25: 2},
                lo_repointings={25: 3},
                ultra_repointings={25: 4}
            )
        )

        actual_l3e_products = process_l3e(initializer_data)
        mock_download.assert_called_once_with("spice_kernel_1.tf")
        mock_find_l3e_spice_kernels_for_time_range.assert_called_once_with(start_epoch, end_epoch)
        mock_spiceypy.furnsh.assert_called_once_with(str(mock_download.return_value))
        mock_get_pointing_date_range.assert_called_once_with(25)

        mock_process_hi.assert_has_calls([
            call(["spice_kernel_1.tf", "hi_ancillary.dat"], 25, start_epoch, epoch_delta, 90, 1),
            call(["spice_kernel_1.tf", "hi_ancillary.dat"], 25, start_epoch, epoch_delta, 135, 2)
        ])
        mock_process_lo.assert_called_once_with(["spice_kernel_1.tf", "lo_ancillary.dat"], 25, start_epoch, epoch_delta, sentinel.elongation, 3)
        mock_process_ultra.assert_called_once_with(["spice_kernel_1.tf", "ul_ancillary.dat"], 25, start_epoch, epoch_delta, 4)

        self.assertEqual(expected_l3e_products, actual_l3e_products)

    @patch('imap_l3_processing.glows.glows_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_ultra(self, mock_shutil, mock_determine_call_args, mock_run,
                               mock_convert_dat_to_glows_l3e_ul_product,
                               mock_save_data, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]

        epoch_start_date = datetime(year=2024, month=10, day=7)
        epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
        epoch_delta = (epoch_end_date - epoch_start_date) / 2
        repointing = 20
        version = 12

        input_metadata = InputMetadata('glows', "l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                                       version='v012', descriptor='survival-probability-ul', repointing=repointing)

        ultra_args = ["20241007_000000", "date.001", "vx", "vy", "vz", "30.000"]

        call_args_object = MagicMock(spec=GlowsL3eCallArguments)
        call_args_object.to_argument_list.return_value = ultra_args
        mock_determine_call_args.return_value = call_args_object

        mock_convert_dat_to_glows_l3e_ul_product.return_value = sentinel.ultra_data_1

        mock_save_data.return_value = "imap_glows_l3e_survival-probability-ul_20241007-repoint00020_v012.cdf"

        parent_file_names = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]
        products = process_l3e_ul(parent_file_names, repointing, epoch_start_date, epoch_delta, version)

        mock_determine_call_args.assert_called_once_with(epoch_start_date, epoch_start_date + epoch_delta, 30)

        mock_run.assert_called_once_with(["./survProbUltra"] + ultra_args)

        output_data_path = Path("probSur.Imap.Ul_20241007_000000_date.001.dat")

        mock_convert_dat_to_glows_l3e_ul_product.assert_called_once_with(
            input_metadata, output_data_path, np.array([epoch_start_date]), call_args_object)

        expected_first_data_path = "imap_glows_l3e_survival-probability-ul-raw_20241007-repoint00020_v012.dat"

        mock_shutil.move.assert_called_once_with(output_data_path, Path(expected_first_data_path))

        mock_save_data.assert_called_once_with(sentinel.ultra_data_1)
        survival_data_product: GlowsL3EUltraData = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"],
                         survival_data_product.parent_file_names)

        self.assertEqual(products, ["imap_glows_l3e_survival-probability-ul_20241007-repoint00020_v012.cdf",
                                    Path(expected_first_data_path)])

    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_hi(self, mock_shutil, mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_hi_product, mock_save_data):

        test_cases = [
            ("90", 90),
            ("45", 135)
        ]

        for descriptor_elongation, elongation in test_cases:
            with self.subTest(elongation=elongation):
                mock_determine_call_args.reset_mock()
                mock_run.reset_mock()
                mock_convert_dat_to_glows_l3e_hi_product.reset_mock()
                mock_save_data.reset_mock()
                mock_shutil.reset_mock()

                epoch_start_date = datetime(year=2024, month=10, day=7)
                epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
                epoch_delta = (epoch_end_date - epoch_start_date) / 2
                repointing = 20
                version = 12

                expected_input_metadata = InputMetadata(
                    instrument='glows', data_level="l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                    version='v012', descriptor=f'survival-probability-hi-{descriptor_elongation}', repointing=repointing)

                l3e_dependencies = MagicMock(spec=GlowsL3EDependencies)
                l3e_dependencies.pipeline_settings = {'start_cr': 2090}
                l3e_dependencies.repointing_file = get_test_data_path("fake_2_day_repointing_on_may18_file.csv")


                hi_args = ["20241007_000000", "date.001", "vx", "vy", "vz", descriptor_elongation]

                mock_call_args_object = MagicMock(spec=GlowsL3eCallArguments)
                mock_call_args_object.to_argument_list.return_value = hi_args
                mock_determine_call_args.return_value = mock_call_args_object
                mock_convert_dat_to_glows_l3e_hi_product.return_value = sentinel.hi_data_1

                saved_cdf_path = Path(f"imap_glows_l3e_survival-probability-hi-{descriptor_elongation}_20241007-repoint00020_v001.cdf")

                mock_save_data.return_value = saved_cdf_path

                parent_file_names = ["some_l3e_hi_parent.dat", "some_repointing_file.repoint.csv"]
                products = process_l3e_hi(parent_file_names, repointing, epoch_start_date, epoch_delta, elongation, version)

                mock_determine_call_args.assert_called_once_with(epoch_start_date, epoch_start_date + epoch_delta, float(elongation))

                mock_run.assert_called_once_with(["./survProbHi"] + hi_args)

                first_output_data_path = Path(f"probSur.Imap.Hi_20241007_000000_date.001_{descriptor_elongation[:5]}.dat")

                mock_convert_dat_to_glows_l3e_hi_product.assert_called_once_with(
                    expected_input_metadata,
                    first_output_data_path,
                    np.array([epoch_start_date]),
                    mock_call_args_object
                )

                mock_save_data.assert_called_once_with(sentinel.hi_data_1)

                survival_data_product: GlowsL3EHiData = mock_save_data.call_args_list[0].args[0]
                self.assertEqual(parent_file_names, survival_data_product.parent_file_names)

                expected_output_data_path = \
                Path(f"imap_glows_l3e_survival-probability-hi-{descriptor_elongation}-raw_20241007-repoint00020_v001.dat")

                mock_shutil.move.assert_called_once_with(first_output_data_path, expected_output_data_path)

                expected_products = [
                    saved_cdf_path,
                    expected_output_data_path
                ]
                self.assertEqual(expected_products, products)


    @patch('imap_l3_processing.glows.glows_processor.save_data')
    @patch("imap_l3_processing.glows.glows_processor.GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product")
    @patch("imap_l3_processing.glows.glows_processor.run")
    @patch("imap_l3_processing.glows.glows_processor.determine_call_args_for_l3e_executable")
    @patch("imap_l3_processing.glows.glows_processor.shutil")
    def test_process_l3e_lo(self, mock_shutil,
                            mock_determine_call_args,
                            mock_run, mock_convert_dat_to_glows_l3e_lo_product, mock_save_data):

        test_cases = [75,105]
        for elongation in test_cases:
            with self.subTest(elongation):
                epoch_start_date = datetime(year=2024, month=10, day=7)
                epoch_end_date = datetime(year=2024, month=10, day=7, hour=23)
                epoch_delta = (epoch_end_date - epoch_start_date) / 2
                repointing = 20
                version = 12

                expected_input_metadata = InputMetadata(
                    instrument='glows', data_level="l3e", start_date=epoch_start_date, end_date=epoch_end_date,
                    version='v012', descriptor=f'survival-probability-lo', repointing=repointing)

                lo_call_args = ["20241007_000000", "date.100", "vx", "vy", "vz", f"{elongation:.3f}"]

                l3e_args = GlowsL3eCallArguments(
                    formatted_date="20241007_000000",
                    decimal_date="date.100",
                    spacecraft_radius=np.float32(100.0),
                    spacecraft_longitude=np.float32(100.0),
                    spacecraft_latitude=np.float32(100.0),
                    spacecraft_velocity_x=np.float32(100.0),
                    spacecraft_velocity_y=np.float32(100.0),
                    spacecraft_velocity_z=np.float32(100.0),
                    spin_axis_longitude=np.float32(100.0),
                    spin_axis_latitude=np.float32(100.0),
                    elongation=elongation
                )
                l3e_args.to_argument_list = Mock(return_value=lo_call_args)
                mock_determine_call_args.return_value = l3e_args

                lo_data_1 = Mock()
                mock_convert_dat_to_glows_l3e_lo_product.return_value = lo_data_1

                output_cdf_path = Path("imap_glows_l3e_survival-probability-lo_20241007-repoint00020_v012.cdf")
                mock_save_data.return_value = output_cdf_path

                parent_file_names = ["l3d_file", "ancillary_1", "ancillary_2", "ancillary_3"]

                products = process_l3e_lo(parent_file_names, repointing, epoch_start_date, epoch_delta, elongation, version)

                mock_determine_call_args.assert_called_once_with(epoch_start_date, epoch_start_date + epoch_delta, elongation)

                mock_run.assert_called_once_with(["./survProbLo"] + lo_call_args)

                first_output_file_path = Path(f"probSur.Imap.Lo_20241007_000000_date.100_{elongation:.2f}.dat")

                mock_convert_dat_to_glows_l3e_lo_product.assert_called_once_with(expected_input_metadata, first_output_file_path,
                         np.array([epoch_start_date]), elongation, l3e_args)

                expected_first_output_file_path = Path("imap_glows_l3e_survival-probability-lo-raw_20241007-repoint00020_v012.dat")

                mock_shutil.move.assert_called_once_with(first_output_file_path, expected_first_output_file_path)

                mock_save_data.assert_called_once_with(lo_data_1)
                survival_data_product: GlowsL3ELoData = mock_save_data.call_args_list[0].args[0]
                self.assertEqual(parent_file_names,
                                 survival_data_product.parent_file_names)

                self.assertEqual([output_cdf_path,
                                  expected_first_output_file_path], products)

                mock_shutil.reset_mock()
                mock_determine_call_args.reset_mock()
                mock_run.reset_mock()
                mock_convert_dat_to_glows_l3e_lo_product.reset_mock()
                mock_save_data.reset_mock()

    @patch('imap_l3_processing.glows.glows_processor.find_l3e_spice_kernels_for_time_range', return_value=[])
    @patch('imap_l3_processing.glows.glows_processor.get_pointing_date_range')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_hi')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_lo')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul')
    def test_process_l3e_skips_repointing_on_exception(self, mock_process_ultra, mock_process_lo,
                                                       mock_process_hi, mock_get_pointing_date_range, _):


        mock_process_hi.side_effect = [
            ValueError("Failed to generate hi"), [Path('path/to/first_hi-45_l3e')],
            [Path('path/to/second_hi-90_l3e')], [Path('path/to/second_hi-45_l3e')],
            [Path('path/to/third_hi-90_l3e')], [Path('path/to/third_hi-45_l3e')],
        ]

        mock_process_lo.side_effect = [
            [Path('path/to/first_lo_l3e')],
            ValueError("Failed to generate lo"),
            [Path('path/to/third_lo_l3e')],
        ]

        mock_process_ultra.side_effect = [
            [Path('path/to/first_ultra_l3e')],
            [Path('path/to/second_ultra_l3e')],
            ValueError("Failed to generate ultra!")
        ]


        expected_l3e_products = [
            Path('path/to/first_lo_l3e'),
            Path('path/to/first_hi-45_l3e'),
            Path('path/to/first_ultra_l3e'),
            Path('path/to/second_hi-90_l3e'),
            Path('path/to/second_hi-45_l3e'),
            Path('path/to/second_ultra_l3e'),
            Path('path/to/third_lo_l3e'),
            Path('path/to/third_hi-90_l3e'),
            Path('path/to/third_hi-45_l3e'),
        ]

        start_epoch_1 = datetime(2020, 1, 1)
        end_epoch_1 = datetime(2020, 1, 2)
        epoch_delta_1 = timedelta(hours=12)

        start_epoch_2 = datetime(2020, 1, 2)
        end_epoch_2 = datetime(2020, 1, 4)
        epoch_delta_2 = timedelta(hours=24)

        start_epoch_3 = datetime(2020, 1, 4)
        end_epoch_3 = datetime(2020, 1, 5)
        epoch_delta_3 = timedelta(hours=12)

        mock_get_pointing_date_range.side_effect = [
            (start_epoch_1, end_epoch_1),
            (start_epoch_2, end_epoch_2),
            (start_epoch_3, end_epoch_3),
        ]

        mock_dependencies = Mock(elongation={"2020024": sentinel.elongation_1, "2020025": sentinel.elongation_2,
                                             "2020026": sentinel.elongation_3})

        hi_parents = ["imap_glows_hi-ancillary_20100101_v001.dat"]
        mock_dependencies.get_hi_parents.return_value = hi_parents
        lo_parents = ["imap_glows_lo-ancillary_20100101_v001.dat"]
        mock_dependencies.get_lo_parents.return_value = lo_parents
        ultra_parents = ["imap_glows_ul-ancillary_20100101_v001.dat"]
        mock_dependencies.get_ul_parents.return_value = ultra_parents

        initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_dependencies,
            repointings=GlowsL3eRepointings(
                repointing_numbers=[24 ,25, 26],
                hi_90_repointings={24: 1, 25: 1, 26: 1},
                hi_45_repointings={24: 2, 25: 2, 26: 2},
                lo_repointings={24: 3, 25: 3, 26: 3},
                ultra_repointings={24: 4, 25: 4, 26: 4}
            )
        )

        actual_l3e_products = process_l3e(initializer_data)

        mock_get_pointing_date_range.assert_has_calls([call(24), call(25), call(26)])

        mock_process_hi.assert_has_calls([
            call(hi_parents, 24, start_epoch_1, epoch_delta_1, 90, 1),
            call(hi_parents, 24, start_epoch_1, epoch_delta_1, 135, 2),
            call(hi_parents, 25, start_epoch_2, epoch_delta_2, 90, 1),
            call(hi_parents, 25, start_epoch_2, epoch_delta_2, 135, 2),
            call(hi_parents, 26, start_epoch_3, epoch_delta_3, 90, 1),
            call(hi_parents, 26, start_epoch_3, epoch_delta_3, 135, 2)
        ])
        mock_process_lo.assert_has_calls([
            call(lo_parents, 24, start_epoch_1, epoch_delta_1, sentinel.elongation_1, 3),
            call(lo_parents, 25, start_epoch_2, epoch_delta_2, sentinel.elongation_2, 3),
            call(lo_parents, 26, start_epoch_3, epoch_delta_3, sentinel.elongation_3, 3),
        ])

        mock_process_ultra.assert_has_calls([
            call(ultra_parents, 24, start_epoch_1, epoch_delta_1, 4),
            call(ultra_parents, 25, start_epoch_2, epoch_delta_2, 4),
            call(ultra_parents, 26, start_epoch_3, epoch_delta_3, 4),
        ])

        self.assertEqual(expected_l3e_products, actual_l3e_products)

    @patch('imap_l3_processing.glows.glows_processor.find_l3e_spice_kernels_for_time_range', return_value=[])
    @patch('imap_l3_processing.glows.glows_processor.get_pointing_date_range')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_hi')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_lo')
    @patch('imap_l3_processing.glows.glows_processor.process_l3e_ul')
    def test_process_l3e_missing_lo_elongation_skips_processing(self, mock_process_ultra, mock_process_lo,
                                                                mock_process_hi, mock_get_pointing_date_range, _):
        mock_process_hi.side_effect = [
            [Path('path/to/first_hi_l3e')],
            [Path('path/to/second_hi_l3e')],
        ]
        mock_process_ultra.return_value = [Path('path/to/ultra_l3e')]

        expected_l3e_products = [
            Path('path/to/first_hi_l3e'), Path('path/to/second_hi_l3e'),
            Path('path/to/ultra_l3e')
        ]

        start_epoch = datetime(2020, 1, 1)
        end_epoch = datetime(2020, 1, 2)
        epoch_delta = timedelta(hours=12)
        mock_get_pointing_date_range.return_value = (start_epoch, end_epoch)

        mock_dependencies = Mock(elongation={})

        mock_dependencies.get_hi_parents.return_value = ["hi_ancillary.dat"]
        mock_dependencies.get_lo_parents.return_value = ["lo_ancillary.dat"]
        mock_dependencies.get_ul_parents.return_value = ["ul_ancillary.dat"]

        initializer_data = GlowsL3EInitializerOutput(
            dependencies=mock_dependencies,
            repointings=GlowsL3eRepointings(
                repointing_numbers=[25],
                hi_90_repointings={25: 1},
                hi_45_repointings={25: 2},
                lo_repointings={25: 3},
                ultra_repointings={25: 4}
            )
        )

        actual_l3e_products = process_l3e(initializer_data)

        mock_get_pointing_date_range.assert_called_once_with(25)

        mock_process_hi.assert_has_calls([
            call(["hi_ancillary.dat"], 25, start_epoch, epoch_delta, 90, 1),
            call(["hi_ancillary.dat"], 25, start_epoch, epoch_delta, 135, 2)
        ])
        mock_process_lo.assert_not_called()
        mock_process_ultra.assert_called_once_with(["ul_ancillary.dat"], 25, start_epoch,
                                                   epoch_delta, 4)

        self.assertEqual(expected_l3e_products, actual_l3e_products)

    @patch("imap_l3_processing.glows.glows_processor.json")
    @patch("imap_l3_processing.glows.glows_processor.ZipFile")
    def test_archive_dependencies(self, mock_zip, mock_json):
        expected_filepath = TEMP_CDF_FOLDER_PATH / "imap_glows_l3b-archive_20250314_v001.zip"
        expected_json_filename = "cr_to_process.json"

        version_number = 1
        l3bc_dependencies = GlowsL3BCDependencies(
            version=version_number,
            carrington_rotation_number=2095,
            start_date=datetime.fromisoformat("2025-03-14 12:34:56.789"),
            end_date=datetime.fromisoformat("2025-03-24 12:34:56.789"),
            l3a_data=[{"filename": "imap_glows_l3a_hist_20250314-repoint000001_v001.cdf"},
                      {"filename": "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"}],
            external_files={},
            ancillary_files={
                'uv_anisotropy': Path("uv_anisotropy.dat"),
                'WawHelioIonMP_parameters': Path("waw_helio_ion.dat"),
                'bad_days_list': Path("bad_days_list.dat"),
                'pipeline_settings': Path("pipeline_settings.json"),
            },
            repointing_file_path=Path("repointing.csv")
        )

        external_dependencies = ExternalDependencies(
            f107_index_file_path=Path("f107_index_file_path"),
            lyman_alpha_path=Path("lyman_alpha_path"),
            omni2_data_path=Path("omni2_data_path"),
        )

        expected_json_to_serialize = {"cr_rotation_number": 2095,
                                      "l3a_paths": ["imap_glows_l3a_hist_20250314-repoint000001_v001.cdf",
                                                    "imap_glows_l3a_hist_20100102-repoint000002_v001.cdf"],
                                      "cr_start_date": "2025-03-14 12:34:56.789000",
                                      "cr_end_date": "2025-03-24 12:34:56.789000",
                                      "bad_days_list": "bad_days_list.dat",
                                      "pipeline_settings": "pipeline_settings.json",
                                      "waw_helioion_mp": "waw_helio_ion.dat",
                                      "uv_anisotropy": "uv_anisotropy.dat",
                                      "repointing_file": "repointing.csv",
                                      }

        mock_zip_file = MagicMock()
        mock_zip.return_value.__enter__.return_value = mock_zip_file

        actual_zip_file_name = GlowsProcessor.archive_dependencies(l3bc_dependencies, external_dependencies)

        self.assertEqual(expected_filepath, actual_zip_file_name)

        mock_zip.assert_called_with(expected_filepath, "w", ZIP_DEFLATED)

        mock_json.dumps.assert_called_once_with(expected_json_to_serialize)

        mock_zip_file.write.assert_has_calls([
            call(Path("lyman_alpha_path"), "lyman_alpha_composite.nc"),
            call(Path("omni2_data_path"), "omni2_all_years.dat"),
            call(Path("f107_index_file_path"), "f107_fluxtable.txt"),
        ])
        mock_zip_file.writestr.assert_called_once_with(expected_json_filename, mock_json.dumps.return_value)

if __name__ == '__main__':
    unittest.main()
