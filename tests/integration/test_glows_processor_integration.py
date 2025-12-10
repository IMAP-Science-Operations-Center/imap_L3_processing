import dataclasses
import json
import logging
import os
import shutil
import subprocess
import unittest
from datetime import timedelta, datetime
from functools import wraps
from pathlib import Path
from typing import Callable
from unittest import skip
from unittest.mock import patch

import numpy as np
import requests
import spiceypy
from imap_data_access import ProcessingInputCollection, RepointInput
from imap_data_access import ScienceInput, AncillaryInput
from imap_data_access.file_validation import ScienceFilePath, AncillaryFilePath
from spacepy.pycdf import CDF

import tests
from imap_l3_processing.constants import TTJ2000_EPOCH
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data, create_glows_l3a_from_dictionary
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import save_data
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, \
    with_tempdir, get_run_local_data_path, run_periodically

GLOWS_L3E_INTEGRATION_DATA_DIR = get_run_local_data_path("glows_l3bcde_integration_data_dir")
INTEGRATION_TEST_DATA = Path(__file__).parent / "test_data"
GLOWS_TEST_DATA = INTEGRATION_TEST_DATA / "glows"


def generate_test_function_import_path(fn: Callable) -> str:
    path_to_imap_processing_dir = Path(tests.__file__).parent.parent
    path_to_file = Path(fn.__code__.co_filename).relative_to(path_to_imap_processing_dir)
    path_in_import_style = str(path_to_file).replace('.py', '.').replace(os.path.sep, '.')
    return path_in_import_style + fn.__qualname__


def run_test_in_docker(test_to_run: Callable):
    @wraps(test_to_run)
    def decorated(self):
        if os.getenv("IN_GLOWS_INTEGRATION_DOCKER"):
            test_to_run(self)
        else:
            l3_processing_dir = Path(tests.__file__).parent.parent

            docker_build = subprocess.run(["docker", "build", "-q", "-f", "Dockerfile_glows_integration", "."],
                                          cwd=l3_processing_dir, capture_output=True)
            image_hash = docker_build.stdout.strip().decode('utf-8')

            print(f"Built docker container: {image_hash}")

            args = [
                "docker", "run", "--rm",
                "--mount", f'type=bind,src={l3_processing_dir}/temp_cdf_data,dst=/temp_cdf_data',
                "--mount", f'type=bind,src={l3_processing_dir}/run_local_input_data,dst=/run_local_input_data',
                image_hash, generate_test_function_import_path(test_to_run)
            ]

            subprocess.run(args, cwd=l3_processing_dir)

    return decorated


class TestGlowsProcessorIntegration(unittest.TestCase):
    @with_tempdir
    def test_glows_l3a(self, tmp_dir):
        input_l2_cdf_path = self._fill_official_l2_cdf_with_json_values(tmp_dir)

        shutil.copy(input_l2_cdf_path, get_test_data_path("glows") / input_l2_cdf_path.name)

        l2_science_file_path = ScienceFilePath(input_l2_cdf_path)

        date_in_path = l2_science_file_path.start_date
        start_date = datetime.strptime(date_in_path, "%Y%m%d")
        end_date = start_date + timedelta(days=1)
        input_metadata = InputMetadata(
            instrument='glows',
            data_level='l3a',
            descriptor='hist',
            start_date=start_date,
            end_date=end_date,
            version='v001',
            repointing=l2_science_file_path.repointing
        )

        expected_json_path = get_test_instrument_team_data_path(
            "glows/imap_glows_l3a_20130908085214_orbX_modX_p_v00.json")
        with open(expected_json_path) as f:
            instrument_team_dict = json.load(f)
        expected_output = create_glows_l3a_from_dictionary(instrument_team_dict, input_metadata)

        with CDF(str(input_l2_cdf_path)) as cdf_data:
            l2_glows_data = read_l2_glows_data(cdf_data)

        dependencies = GlowsL3ADependencies(l2_glows_data, {
            "calibration_data": get_test_instrument_team_data_path(
                "glows/imap_glows_calibration-data_20100101_v002.dat"),
            "settings": get_test_instrument_team_data_path("glows/imap_glows_pipeline-settings_20100101_v001.json"),
            "time_dependent_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_time-dep-bckgrd_20100101_v001.dat"),
            "extra_heliospheric_bckgrd": get_test_instrument_team_data_path(
                "glows/imap_glows_map-of-extra-helio-bckgrd_20100101_v002.dat"),
        })

        processor = GlowsProcessor(ProcessingInputCollection(), input_metadata)
        l3a_data = processor.process_l3a(dependencies)

        print(save_data(l3a_data, delete_if_present=True))

        expected_dict = dataclasses.asdict(expected_output)
        actual_dict = dataclasses.asdict(l3a_data)

        self.assertEqual(input_metadata.repointing, l3a_data.identifier)

        np.testing.assert_allclose(actual_dict['photon_flux'], expected_dict['photon_flux'], rtol=1e-3)
        np.testing.assert_allclose(actual_dict['photon_flux_uncertainty'], expected_dict['photon_flux_uncertainty'],
                                   rtol=1e-3)
        np.testing.assert_allclose(actual_dict['raw_histogram'], expected_dict['raw_histogram'])
        np.testing.assert_allclose(actual_dict['exposure_times'], expected_dict['exposure_times'], rtol=1e-3)
        np.testing.assert_allclose(actual_dict['number_of_bins'], expected_dict['number_of_bins'])
        self.assertEqual(actual_dict['epoch'], expected_dict['epoch'])
        np.testing.assert_allclose(actual_dict['epoch_delta'], expected_dict['epoch_delta'])
        np.testing.assert_allclose(actual_dict['spin_angle'], expected_dict['spin_angle'])
        np.testing.assert_allclose(actual_dict['spin_angle_delta'], expected_dict['spin_angle_delta'])
        np.testing.assert_allclose(actual_dict['latitude'], expected_dict['latitude'], atol=1e-3)
        np.testing.assert_allclose(actual_dict['longitude'], expected_dict['longitude'], atol=1e-3)
        # np.testing.assert_allclose(actual_dict['extra_heliospheric_background'], expected_dict['extra_heliospheric_background'])
        # np.testing.assert_allclose(actual_dict['time_dependent_background'], expected_dict['time_dependent_background'])
        np.testing.assert_allclose(actual_dict['filter_temperature_average'],
                                   expected_dict['filter_temperature_average'])
        np.testing.assert_allclose(actual_dict['filter_temperature_std_dev'],
                                   expected_dict['filter_temperature_std_dev'])
        np.testing.assert_allclose(actual_dict['hv_voltage_average'], expected_dict['hv_voltage_average'])
        np.testing.assert_allclose(actual_dict['hv_voltage_std_dev'], expected_dict['hv_voltage_std_dev'])
        np.testing.assert_allclose(actual_dict['spin_period_average'], expected_dict['spin_period_average'])
        np.testing.assert_allclose(actual_dict['spin_period_std_dev'], expected_dict['spin_period_std_dev'])
        np.testing.assert_allclose(actual_dict['spin_period_ground_average'],
                                   expected_dict['spin_period_ground_average'])
        np.testing.assert_allclose(actual_dict['spin_period_ground_std_dev'],
                                   expected_dict['spin_period_ground_std_dev'])
        np.testing.assert_allclose(actual_dict['pulse_length_average'], expected_dict['pulse_length_average'])
        np.testing.assert_allclose(actual_dict['pulse_length_std_dev'], expected_dict['pulse_length_std_dev'])
        np.testing.assert_allclose(actual_dict['position_angle_offset_average'],
                                   expected_dict['position_angle_offset_average'])
        np.testing.assert_allclose(actual_dict['position_angle_offset_std_dev'],
                                   expected_dict['position_angle_offset_std_dev'])
        np.testing.assert_allclose(actual_dict['spin_axis_orientation_average'],
                                   expected_dict['spin_axis_orientation_average'])
        np.testing.assert_allclose(actual_dict['spin_axis_orientation_std_dev'],
                                   expected_dict['spin_axis_orientation_std_dev'])
        np.testing.assert_allclose(actual_dict['spacecraft_location_average'],
                                   expected_dict['spacecraft_location_average'])
        np.testing.assert_allclose(actual_dict['spacecraft_location_std_dev'],
                                   expected_dict['spacecraft_location_std_dev'])

        self.assertEqual(actual_dict['input_metadata'], expected_dict['input_metadata'])

    @run_periodically(timedelta(days=14))
    @run_test_in_docker
    def test_l3bcde_first_time_processing(self):
        input_files = [
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250428-repoint01013_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250429-repoint01014_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250510-repoint01025_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250511-repoint01026_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250525-repoint01040_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250526-repoint01041_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250606-repoint01053_v001.cdf",
            GLOWS_TEST_DATA / "imap_glows_l3a_hist_20250607-repoint01054_v001.cdf",

            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-1CR_20100101_v001.json",
            GLOWS_TEST_DATA / "imap_glows_WawHelioIonMP_20100101_v001.json",
            GLOWS_TEST_DATA / "imap_glows_bad-days-list_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_pipeline-settings-l3bcde_20100101_v003.json",
            GLOWS_TEST_DATA / "imap_glows_plasma-speed-2010a_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_proton-density-2010a_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-2010a_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_photoion-2010a_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_lya-2010a_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_electron-density-2010a_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_ionization-files_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-lo_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-xyz-8_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_lo_elongation-data_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-hi_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_energy-grid-ultra_20100101_v001.dat",
            GLOWS_TEST_DATA / "imap_glows_tess-ang-16_20100101_v001.dat",
            INTEGRATION_TEST_DATA / "spice" / "imap_2026_269_05.repoint.csv",
            INTEGRATION_TEST_DATA / "spice" / "imap_2025_105_2026_105_01.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
            INTEGRATION_TEST_DATA / "spice" / "imap_science_108.tf",
            INTEGRATION_TEST_DATA / "spice" / "naif020.tls",
            INTEGRATION_TEST_DATA / "spice" / "imap_sclk_008.tsc",
            INTEGRATION_TEST_DATA / "spice" / "de440.bsp",
            INTEGRATION_TEST_DATA / "spice" / "imap_recon_20250415_20260415_v01.bsp",
        ]
        with mock_imap_data_access(GLOWS_L3E_INTEGRATION_DATA_DIR, input_files):

            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            for folder in ["data_l3b", "data_l3c", "data_l3d", "data_l3d_txt"]:
                path = PATH_TO_L3D_TOOLKIT / folder
                if path.exists():
                    shutil.rmtree(path)

            processing_input = ProcessingInputCollection(RepointInput("imap_2026_269_05.repoint.csv"))
            input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                           version="v001", start_date=datetime(2000, 1, 1),
                                           end_date=datetime(2000, 1, 1))

            processor = GlowsProcessor(processing_input, input_metadata)
            processor.process()

            expected_files = [
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20250425-cr02297_v001.cdf'),
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20250523-cr02298_v001.cdf'),

                ScienceFilePath('imap_glows_l3c_sw-profile_20250425-cr02297_v001.cdf'),
                ScienceFilePath('imap_glows_l3c_sw-profile_20250523-cr02298_v001.cdf'),

                ScienceFilePath('imap_glows_l3d_solar-hist_19470303-cr02297_v001.cdf'),
                AncillaryFilePath('imap_glows_uv-anis_19470303_20250509_v001.dat'),
                AncillaryFilePath('imap_glows_lya_19470303_20250509_v001.dat'),
                AncillaryFilePath('imap_glows_e-dens_19470303_20250509_v001.dat'),
                AncillaryFilePath('imap_glows_p-dens_19470303_20250509_v001.dat'),
                AncillaryFilePath('imap_glows_speed_19470303_20250509_v001.dat'),
                AncillaryFilePath('imap_glows_phion_19470303_20250509_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-ul-sf_20250425-repoint01010_v001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-ul-sf_20250426-repoint01011_v001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-ul-sf-raw_20250425_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-ul-sf-raw_20250426_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-ul-hf_20250425-repoint01010_v001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-ul-hf_20250426-repoint01011_v001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-ul-hf-raw_20250425_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-ul-hf-raw_20250426_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20250425-repoint01010_v001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20250426-repoint01011_v001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20250425_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20250426_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20250425-repoint01010_v001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20250426-repoint01011_v001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20250425_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20250426_v001.dat'),

                ScienceFilePath('imap_glows_l3e_survival-probability-lo_20250425-repoint01010_v001.cdf'),
                ScienceFilePath('imap_glows_l3e_survival-probability-lo_20250426-repoint01011_v001.cdf'),
                AncillaryFilePath('imap_glows_survival-probability-lo-raw_20250425_v001.dat'),
                AncillaryFilePath('imap_glows_survival-probability-lo-raw_20250426_v001.dat'),
            ]

            for file_path in expected_files:
                self.assertTrue(file_path.construct_path().exists(), msg=str(file_path.construct_path()))

    @skip("takes an hour to run")
    @run_test_in_docker
    def test_local_integration(self):
        original_datetime2et = spiceypy.datetime2et
        original_get = requests.get
        offset = (datetime(2025, 4, 15) - datetime(2010, 1, 1)).total_seconds()
        jan_through_apr_offset = (datetime(2026, 1, 1) - datetime(2010, 1, 1)).total_seconds()
        apr_through_dec_offset = (datetime(2025, 4, 15) - datetime(2010, 4, 15)).total_seconds()

        def hijack_metakernel_params(url: str, *, params: dict = {}):
            if 'start_time' in params and 'end_time' in params:
                params['start_time'] = str(int(int(params['start_time']) + offset))
                params['end_time'] = str(int(int(params['end_time']) + offset))
            return original_get(url, params=params)

        def determine_spice_offset(date: datetime):
            if date.month < 4 or (date.month == 4 and date.day < 15):
                return jan_through_apr_offset
            else:
                return apr_through_dec_offset

        with (patch('spiceypy.datetime2et', side_effect=lambda x: original_datetime2et(x) + determine_spice_offset(x)),
              patch('requests.get', side_effect=hijack_metakernel_params)):

            ancillary_file_paths = [
                GLOWS_TEST_DATA / "imap_glows_calibration-data_20000101_v003.dat",
                GLOWS_TEST_DATA / "imap_glows_map-of-extra-helio-bckgrd_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_pipeline-settings_20100101_v003.json",
                GLOWS_TEST_DATA / "imap_glows_time-dep-bckgrd_20100101_v001.dat"
            ]
            l2_paths = list(get_run_local_data_path("glows_l2_cdfs").iterdir())
            input_files = l2_paths + ancillary_file_paths

            l3a_integration_data_dir = get_run_local_data_path("glows_local_integration_l3a")

            with (mock_imap_data_access(l3a_integration_data_dir, input_files)):
                for i, l2_path in enumerate(l2_paths):
                    l2_science_file_path = ScienceFilePath(l2_path)
                    l2_science_input = ScienceInput(l2_path.name)
                    input_files.extend(ancillary_file_paths)

                    input_metadata = InputMetadata(
                        instrument="glows",
                        data_level="l3a",
                        start_date=l2_science_input.get_time_range()[0],
                        end_date=l2_science_input.get_time_range()[1],
                        version="v001",
                        descriptor="hist",
                        repointing=l2_science_file_path.repointing,
                    )

                    processingInputs = [AncillaryInput(ancillary.name) for ancillary in ancillary_file_paths] + \
                                       [l2_science_input]
                    processor = GlowsProcessor(ProcessingInputCollection(*processingInputs), input_metadata)
                    _ = processor.process()

            l3bcde_input_files = [
                GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-1CR_20100101_v001.json",
                GLOWS_TEST_DATA / "imap_glows_WawHelioIonMP_20100101_v001.json",
                GLOWS_TEST_DATA / "imap_glows_bad-days-list_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_pipeline-settings-l3bcde_20100101_v003.json",
                GLOWS_TEST_DATA / "imap_glows_plasma-speed-2010a_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_proton-density-2010a_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_uv-anisotropy-2010a_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_photoion-2010a_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_lya-2010a_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_electron-density-2010a_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_ionization-files_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_energy-grid-lo_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_tess-xyz-8_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_lo_elongation-data_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_energy-grid-hi_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_energy-grid-ultra_20100101_v001.dat",
                GLOWS_TEST_DATA / "imap_glows_tess-ang-16_20100101_v001.dat",
                INTEGRATION_TEST_DATA / "spice" / "imap_2026_269_15.repoint",
                INTEGRATION_TEST_DATA / "spice" / "imap_2025_105_2026_105_01.ah.bc",
                INTEGRATION_TEST_DATA / "spice" / "imap_dps_2025_105_2026_105_009.ah.bc",
                INTEGRATION_TEST_DATA / "spice" / "imap_science_108.tf",
                INTEGRATION_TEST_DATA / "spice" / "naif020.tls",
                INTEGRATION_TEST_DATA / "spice" / "imap_sclk_008.tsc",
                INTEGRATION_TEST_DATA / "spice" / "de440.bsp",
                INTEGRATION_TEST_DATA / "spice" / "imap_recon_20250415_20260415_v01.bsp",
            ]
            l3a_inputs = list((l3a_integration_data_dir / "imap/glows/l3a").rglob("*.cdf"))

            logging.basicConfig(force=True, level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            with mock_imap_data_access(get_run_local_data_path("glows_local_integration_l3bcde"),
                                       l3bcde_input_files + l3a_inputs):
                processing_input = ProcessingInputCollection(RepointInput("imap_2026_269_15.repoint"))
                input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                               version="v001", start_date=datetime(2000, 1, 1),
                                               end_date=datetime(2000, 1, 1))

                processor = GlowsProcessor(processing_input, input_metadata)
                processor.process()

    @staticmethod
    def _fill_official_l2_cdf_with_json_values(output_folder: Path) -> Path:
        official_l2_path = get_test_data_path("glows/imap_glows_l2_hist_20130908-repoint01000_v001.cdf")
        json_file_path = get_test_instrument_team_data_path("glows/imap_glows_l2_20130908085214_orbX_modX_p_v00.json")

        new_file_path = output_folder / official_l2_path.name
        new_file_path.unlink(missing_ok=True)

        with CDF(str(new_file_path), masterpath=str(official_l2_path)) as cdf:
            with open(json_file_path) as f:
                instrument_data = json.load(f)

                start_of_epoch_window = datetime.fromisoformat(instrument_data["start_time"])
                end_of_epoch_window = datetime.fromisoformat(instrument_data["end_time"])
                epoch_window = end_of_epoch_window - start_of_epoch_window
                epoch = start_of_epoch_window + epoch_window / 2

                cdf["epoch"][0] = epoch
                cdf['start_time'][0] = (start_of_epoch_window - TTJ2000_EPOCH).total_seconds() * 1e9
                cdf['end_time'][0] = (end_of_epoch_window - TTJ2000_EPOCH).total_seconds() * 1e9

                lightcurve_vars = [
                    "spin_angle",
                    "photon_flux",
                    "exposure_times",
                    "flux_uncertainties",
                    "ecliptic_lon",
                    "ecliptic_lat",
                ]
                for var in lightcurve_vars:
                    cdf[var] = np.array(instrument_data["daily_lightcurve"][var])[np.newaxis, :]

                cdf["raw_histograms"] = np.array(instrument_data["daily_lightcurve"]["raw_histogram"])[np.newaxis, :]
                cdf["histogram_flag_array"] = np.array(
                    [int(f, 16) for f in instrument_data["daily_lightcurve"]["histogram_flag_array"]])[np.newaxis, :]

                single_value_vars = [
                    "filter_temperature_average",
                    "filter_temperature_std_dev",
                    "hv_voltage_average",
                    "hv_voltage_std_dev",
                    "spin_period_average",
                    "spin_period_std_dev",
                    "pulse_length_average",
                    "pulse_length_std_dev",
                    "spin_period_ground_average",
                    "spin_period_ground_std_dev",
                    "position_angle_offset_average",
                    "position_angle_offset_std_dev",
                    "identifier"
                ]
                for var in single_value_vars:
                    cdf[var][0] = instrument_data[var]

                vector_vars = [
                    "spacecraft_location_average",
                    "spacecraft_location_std_dev",
                    "spacecraft_velocity_average",
                    "spacecraft_velocity_std_dev",
                    "spin_axis_orientation_average",
                    "spin_axis_orientation_std_dev"
                ]
                for var in vector_vars:
                    cdf[var][0] = np.array(list(instrument_data[var].values()))

                cdf["bad_time_flag_occurrences"][0] = list(instrument_data["bad_time_flag_occurences"].values())
                cdf["number_of_good_l1b_inputs"][0] = instrument_data["header"]["number_of_l1b_files_used"]
                cdf["total_l1b_inputs"][0] = instrument_data["header"]["number_of_all_l1b_files"]

        return new_file_path


if __name__ == '__main__':
    unittest.main()
