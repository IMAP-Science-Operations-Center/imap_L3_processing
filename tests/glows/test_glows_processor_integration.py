import logging
import os
import shutil
import subprocess
import unittest
from datetime import timedelta, datetime
from pathlib import Path
from platform import platform
from unittest import skipIf
from unittest.mock import patch

import imap_data_access
import spiceypy
from imap_data_access import ProcessingInputCollection, RepointInput
from imap_data_access.file_validation import generate_imap_file_path, SPICEFilePath, ScienceFilePath, AncillaryFilePath

import tests
from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import create_glows_mock_query_results, run_periodically, get_spice_data_path, \
    get_run_local_data_path, get_test_data_path

GLOWS_L3E_INTEGRATION_DATA_DIR = get_run_local_data_path("glows_l3bcde_integration_data_dir")

@patch.dict(imap_data_access.config, {"DATA_DIR": GLOWS_L3E_INTEGRATION_DATA_DIR})
class TestGlowsProcessorIntegration(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if GLOWS_L3E_INTEGRATION_DATA_DIR.exists():
            shutil.rmtree(GLOWS_L3E_INTEGRATION_DATA_DIR)
        GLOWS_L3E_INTEGRATION_DATA_DIR.mkdir(exist_ok=True, parents=True)

        for folder in ["data_l3b", "data_l3c", "data_l3d", "data_l3d_txt"]:
            path = PATH_TO_L3D_TOOLKIT / folder
            if path.exists():
                shutil.rmtree(path)

    @skipIf(os.getenv("IN_GLOWS_INTEGRATION_DOCKER"), "Not needed on linux")
    @run_periodically(timedelta(days=7))
    def test_glows_integration_running_docker(self):
        l3_processing_dir = Path(tests.__file__).parent.parent

        docker_build = subprocess.run(["docker", "build", "-q", "-f", "Dockerfile_glows_integration", "."],
                       cwd=l3_processing_dir, capture_output=True)
        image_hash = docker_build.stdout.strip().decode('utf-8')

        print(f"Built docker container: {image_hash}")
        subprocess.run(["docker", "run", "--rm",
                        f"--mount", f'type=bind,src={l3_processing_dir}/temp_cdf_data,dst=/temp_cdf_data',
                        "--mount", f'type=bind,src={l3_processing_dir}/run_local_input_data,dst=/run_local_input_data',
                        image_hash], cwd=l3_processing_dir)

        # @formatter:off
        expected_files = [
            ScienceFilePath('imap_glows_l3b_ion-rate-profile_20250425-cr02297_v001.cdf').construct_path(),
            ScienceFilePath('imap_glows_l3b_ion-rate-profile_20250523-cr02298_v001.cdf').construct_path(),

            ScienceFilePath('imap_glows_l3c_sw-profile_20250425-cr02297_v001.cdf').construct_path(),
            ScienceFilePath('imap_glows_l3c_sw-profile_20250523-cr02298_v001.cdf').construct_path(),

            ScienceFilePath('imap_glows_l3d_solar-hist_19470303-cr02297_v001.cdf').construct_path(),
            AncillaryFilePath('imap_glows_uv-anis_19470303_20250509_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_lya_19470303_20250509_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_e-dens_19470303_20250509_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_p-dens_19470303_20250509_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_speed_19470303_20250509_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_phion_19470303_20250509_v001.dat').construct_path(),

            ScienceFilePath('imap_glows_l3e_survival-probability-ul_20250425-repoint01010_v001.cdf').construct_path(),
            ScienceFilePath('imap_glows_l3e_survival-probability-ul_20250426-repoint01011_v001.cdf').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-ul-raw_20250425_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-ul-raw_20250426_v001.dat').construct_path(),

            ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20250425-repoint01010_v001.cdf').construct_path(),
            ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20250426-repoint01011_v001.cdf').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20250425_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20250426_v001.dat').construct_path(),

            ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20250425-repoint01010_v001.cdf').construct_path(),
            ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20250426-repoint01011_v001.cdf').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20250425_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20250426_v001.dat').construct_path(),

            ScienceFilePath('imap_glows_l3e_survival-probability-lo_20250425-repoint01010_v001.cdf').construct_path(),
            ScienceFilePath('imap_glows_l3e_survival-probability-lo_20250426-repoint01011_v001.cdf').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-lo-raw_20250425_v001.dat').construct_path(),
            AncillaryFilePath('imap_glows_survival-probability-lo-raw_20250426_v001.dat').construct_path(),
        ]
        # @formatter:on

        for file_path in expected_files:
            self.assertTrue(file_path.exists(), msg=str(file_path))

    @skipIf(os.getenv("IN_GLOWS_INTEGRATION_DOCKER") is None, "Only runs in a docker container!")
    def test_l3bcde_integration(self):
        expected_queries = {
            "hist": create_glows_mock_query_results([
                "imap_glows_l3a_hist_20250428-repoint01013_v001.cdf",
                "imap_glows_l3a_hist_20250429-repoint01014_v001.cdf",
                "imap_glows_l3a_hist_20250510-repoint01025_v008.cdf",
                "imap_glows_l3a_hist_20250511-repoint01026_v008.cdf",
                "imap_glows_l3a_hist_20250525-repoint01040_v001.cdf",
                "imap_glows_l3a_hist_20250526-repoint01041_v001.cdf",
                "imap_glows_l3a_hist_20250607-repoint01053_v012.cdf",
                "imap_glows_l3a_hist_20250607-repoint01054_v012.cdf",
            ]),
            "ion-rate-profile": create_glows_mock_query_results([]),
            "sw-profile": create_glows_mock_query_results([]),
            "uv-anisotropy-1CR": create_glows_mock_query_results(["imap_glows_uv-anisotropy-1CR_20100101_v004.json"]),
            "WawHelioIonMP": create_glows_mock_query_results(["imap_glows_WawHelioIonMP_20100101_v002.json"]),
            "bad-days-list": create_glows_mock_query_results(["imap_glows_bad-days-list_20100101_v001.dat"]),
            "pipeline-settings-l3bcde": create_glows_mock_query_results(
                ["imap_glows_pipeline-settings-l3bcde_20100101_v006.json"]),
            'solar-hist': create_glows_mock_query_results([]),
            'plasma-speed-2010a': create_glows_mock_query_results(['imap_glows_plasma-speed-2010a_20100101_v003.dat']),
            'proton-density-2010a': create_glows_mock_query_results(
                ['imap_glows_proton-density-2010a_20100101_v003.dat']),
            'uv-anisotropy-2010a': create_glows_mock_query_results(
                ['imap_glows_uv-anisotropy-2010a_20100101_v003.dat']),
            'photoion-2010a': create_glows_mock_query_results(['imap_glows_photoion-2010a_20100101_v003.dat']),
            'lya-2010a': create_glows_mock_query_results(['imap_glows_lya-2010a_20100101_v003.dat']),
            'electron-density-2010a': create_glows_mock_query_results(
                ['imap_glows_electron-density-2010a_20100101_v003.dat']),
            'ionization-files': create_glows_mock_query_results(['imap_glows_ionization-files_20100101_v001.dat']),
            'energy-grid-lo': create_glows_mock_query_results(['imap_glows_energy-grid-lo_20100101_v001.dat']),
            'tess-xyz-8': create_glows_mock_query_results(['imap_glows_tess-xyz-8_20100101_v001.dat']),
            'elongation-data': create_glows_mock_query_results(['imap_lo_elongation-data_20100101_v001.dat']),
            'energy-grid-hi': create_glows_mock_query_results(['imap_glows_energy-grid-hi_20100101_v001.dat']),
            'energy-grid-ultra': create_glows_mock_query_results(['imap_glows_energy-grid-ultra_20100101_v001.dat']),
            'tess-ang-16': create_glows_mock_query_results(['imap_glows_tess-ang-16_20100101_v001.dat']),
            'survival-probability-hi-90': create_glows_mock_query_results([]),
            'survival-probability-hi-45': create_glows_mock_query_results([]),
            'survival-probability-lo': create_glows_mock_query_results([]),
            'survival-probability-ultra': create_glows_mock_query_results([]),
        }

        input_files = [
            "imap_glows_l3a_hist_20250428-repoint01013_v001.cdf",
            "imap_glows_l3a_hist_20250429-repoint01014_v001.cdf",
            "imap_glows_l3a_hist_20250510-repoint01025_v008.cdf",
            "imap_glows_l3a_hist_20250511-repoint01026_v008.cdf",
            "imap_glows_l3a_hist_20250525-repoint01040_v001.cdf",
            "imap_glows_l3a_hist_20250526-repoint01041_v001.cdf",
            "imap_glows_l3a_hist_20250607-repoint01053_v012.cdf",
            "imap_glows_l3a_hist_20250607-repoint01054_v012.cdf",
            "imap_glows_uv-anisotropy-1CR_20100101_v004.json",
            "imap_glows_WawHelioIonMP_20100101_v002.json",
            "imap_glows_bad-days-list_20100101_v001.dat",
            "imap_glows_pipeline-settings-l3bcde_20100101_v006.json",
            'imap_glows_plasma-speed-2010a_20100101_v003.dat',
            'imap_glows_proton-density-2010a_20100101_v003.dat',
            'imap_glows_uv-anisotropy-2010a_20100101_v003.dat',
            'imap_glows_photoion-2010a_20100101_v003.dat',
            'imap_glows_lya-2010a_20100101_v003.dat',
            'imap_glows_electron-density-2010a_20100101_v003.dat',
            'imap_glows_ionization-files_20100101_v001.dat',
            'imap_glows_energy-grid-lo_20100101_v001.dat',
            'imap_glows_tess-xyz-8_20100101_v001.dat',
            'imap_lo_elongation-data_20100101_v001.dat',
            'imap_glows_energy-grid-hi_20100101_v001.dat',
            'imap_glows_energy-grid-ultra_20100101_v001.dat',
            'imap_glows_tess-ang-16_20100101_v001.dat',
            "imap_2026_269_05.repoint.csv",
            "imap_2025_105_2026_105_01.ah.bc",
            "imap_dps_2025_105_2026_105_01.ah.bc",
            "imap_science_100.tf",
            "naif0012.tls",
            "imap_sclk_0000.tsc",
            "de440.bsp",
            "imap_recon_20250415_20260415_v01.bsp",
        ]

        def fake_download(file: Path | str):
            filename = Path(file).name
            imap_file_path = generate_imap_file_path(filename)
            full_path = imap_file_path.construct_path()

            self.assertTrue(full_path.exists())
            return full_path

        def fake_query(**kwargs):
            return expected_queries[kwargs["descriptor"]]

        with (
            patch.object(imap_data_access, "download", new=fake_download),
            patch.object(imap_data_access, "query", new=fake_query)
        ):
            for filename in input_files:
                paths_to_generate = generate_imap_file_path(filename).construct_path()
                paths_to_generate.parent.mkdir(exist_ok=True, parents=True)

                input_files_path = get_test_data_path("glows/l3bcde_integration_test_data")
                shutil.copy(src=input_files_path / filename, dst=paths_to_generate)

            processing_input = ProcessingInputCollection(RepointInput("imap_2026_269_05.repoint.csv"))

            input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                           version="v001", start_date=datetime(2000, 1, 1),
                                           end_date=datetime(2000, 1, 1))
            processor = GlowsProcessor(processing_input, input_metadata)
            processor.process()


if __name__ == '__main__':
    unittest.main()