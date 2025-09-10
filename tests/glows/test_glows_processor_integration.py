import logging
import shutil
import unittest
from datetime import timedelta, datetime
from pathlib import Path
from unittest.mock import patch

import imap_data_access
import spiceypy
from imap_data_access import ProcessingInputCollection, RepointInput
from imap_data_access.file_validation import generate_imap_file_path, SPICEFilePath, ScienceFilePath, AncillaryFilePath

from imap_l3_processing.glows.glows_processor import GlowsProcessor
from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import create_glows_mock_query_results, run_periodically, get_spice_data_path, \
    get_run_local_data_path, get_test_data_path


class TestGlowsProcessorIntegration(unittest.TestCase):
    # @skipIf(not platform == "linux", "Only runs in a docker container!")
    @run_periodically(timedelta(seconds=7))
    @patch("imap_data_access.query")
    def test_l3bcde_integration(self, mock_query):
        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        for folder in ["data_l3b", "data_l3c", "data_l3d", "data_l3d_txt"]:
            path = PATH_TO_L3D_TOOLKIT / folder
            if path.exists():
                shutil.rmtree(path)

        expected_queries = {
            "hist": create_glows_mock_query_results([
                "imap_glows_l3a_hist_20100105-repoint00153_v001.cdf",
                "imap_glows_l3a_hist_20100106-repoint00154_v001.cdf",
                "imap_glows_l3a_hist_20100107-repoint00155_v008.cdf",
                "imap_glows_l3a_hist_20100120-repoint00168_v008.cdf",
                "imap_glows_l3a_hist_20100131-repoint00180_v001.cdf",
                "imap_glows_l3a_hist_20100201-repoint00181_v001.cdf",
                "imap_glows_l3a_hist_20100220-repoint00200_v012.cdf",
                "imap_glows_l3a_hist_20100221-repoint00201_v012.cdf",
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
            "imap_glows_l3a_hist_20100105-repoint00153_v001.cdf",
            "imap_glows_l3a_hist_20100106-repoint00154_v001.cdf",
            "imap_glows_l3a_hist_20100107-repoint00155_v008.cdf",
            "imap_glows_l3a_hist_20100120-repoint00168_v008.cdf",
            "imap_glows_l3a_hist_20100120-repoint00168_v008.cdf",
            "imap_glows_l3a_hist_20100131-repoint00180_v001.cdf",
            "imap_glows_l3a_hist_20100201-repoint00181_v001.cdf",
            "imap_glows_l3a_hist_20100220-repoint00200_v012.cdf",
            "imap_glows_l3a_hist_20100221-repoint00201_v012.cdf",
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
            "imap_2026_269_05.repoint.csv",
            'imap_glows_ionization-files_20100101_v001.dat',
            'imap_glows_energy-grid-lo_20100101_v001.dat',
            'imap_glows_tess-xyz-8_20100101_v001.dat',
            'imap_lo_elongation-data_20100101_v001.dat',
            'imap_glows_energy-grid-hi_20100101_v001.dat',
            'imap_glows_energy-grid-ultra_20100101_v001.dat',
            'imap_glows_tess-ang-16_20100101_v001.dat',
        ]

        original_download = imap_data_access.download
        def fake_download(file: Path | str):
            filename = Path(file).name
            imap_file_path = generate_imap_file_path(filename)
            full_path = imap_file_path.construct_path()

            if isinstance(imap_file_path, SPICEFilePath) and "repoint" not in filename:
                original_download(full_path)

            self.assertTrue(full_path.exists())
            return full_path

        for kernel in get_spice_data_path("some_file").parent.iterdir():
            spiceypy.furnsh(str(kernel))

        original_datetime2et = spiceypy.datetime2et
        mock_query.side_effect = lambda **kwargs: expected_queries[kwargs["descriptor"]]
        timeshift_get_et_time = lambda date: original_datetime2et(date + timedelta(days=365 * 16 + 4, hours=2))

        input_files_path = get_test_data_path("glows/l3bcde_integration_test_data")

        fake_data_dir = get_run_local_data_path("glows_l3bcde_integration_data_dir")
        if fake_data_dir.exists():
            shutil.rmtree(fake_data_dir)
        fake_data_dir.mkdir(exist_ok=True, parents=True)

        with (
            patch.dict(imap_data_access.config, {"DATA_DIR": fake_data_dir}),
            patch.object(imap_data_access, "download", new=fake_download),
            patch.object(spiceypy, "datetime2et", new=timeshift_get_et_time)
        ):

            for filename in input_files:
                paths_to_generate = generate_imap_file_path(filename).construct_path()
                paths_to_generate.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(src=input_files_path / filename, dst=paths_to_generate)

            processing_input = ProcessingInputCollection(RepointInput("imap_2026_269_05.repoint.csv"))

            input_metadata = InputMetadata(instrument="glows", data_level="l3b", descriptor="ion-rate-profile",
                                           version="v001", start_date=datetime(2000, 1, 1),
                                           end_date=datetime(2000, 1, 1))
            processor = GlowsProcessor(processing_input, input_metadata)
            products = processor.process()

            print(products)

            # @formatter:off
            expected_files = [
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20100103-cr02092_v001.cdf').construct_path(),
                ScienceFilePath('imap_glows_l3b_ion-rate-profile_20100130-cr02093_v001.cdf').construct_path(),

                ScienceFilePath('imap_glows_l3c_sw-profile_20100103-cr02092_v001.cdf').construct_path(),
                ScienceFilePath('imap_glows_l3c_sw-profile_20100130-cr02093_v001.cdf').construct_path(),

                ScienceFilePath('imap_glows_l3d_solar-hist_19470303-cr02092_v001.cdf').construct_path(),
                AncillaryFilePath('imap_glows_uv-anis_19470303_20100117_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_lya_19470303_20100117_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_e-dens_19470303_20100117_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_p-dens_19470303_20100117_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_speed_19470303_20100117_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_phion_19470303_20100117_v001.dat').construct_path(),

                ScienceFilePath('imap_glows_l3e_survival-probability-ul_20100103-repoint00151_v001.cdf').construct_path(),
                ScienceFilePath('imap_glows_l3e_survival-probability-ul_20100104-repoint00152_v001.cdf').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-ul-raw_20100103_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-ul-raw_20100104_v001.dat').construct_path(),

                ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20100103-repoint00151_v001.cdf').construct_path(),
                ScienceFilePath('imap_glows_l3e_survival-probability-hi-45_20100104-repoint00152_v001.cdf').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20100103_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-hi-45-raw_20100104_v001.dat').construct_path(),

                ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20100103-repoint00151_v001.cdf').construct_path(),
                ScienceFilePath('imap_glows_l3e_survival-probability-hi-90_20100104-repoint00152_v001.cdf').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20100103_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-hi-90-raw_20100104_v001.dat').construct_path(),

                ScienceFilePath('imap_glows_l3e_survival-probability-lo_20100103-repoint00151_v001.cdf').construct_path(),
                ScienceFilePath('imap_glows_l3e_survival-probability-lo_20100104-repoint00152_v001.cdf').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-lo-raw_20100103_v001.dat').construct_path(),
                AncillaryFilePath('imap_glows_survival-probability-lo-raw_20100104_v001.dat').construct_path(),
            ]
            # @formatter:on

            for file_path in expected_files:
                self.assertTrue(file_path.exists(), msg=str(file_path))

if __name__ == '__main__':
    unittest.main()