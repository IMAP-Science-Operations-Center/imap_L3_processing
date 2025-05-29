import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, call

import imap_data_access
from astropy.time import TimeDelta
from imap_data_access.processing_input import AncillaryInput, ProcessingInputCollection

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import \
    GlowsInitializerAncillaryDependencies, \
    F107_FLUX_TABLE_URL, \
    LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL, _comment_headers
from tests.test_helpers import get_test_data_path


class TestGlowsInitializerAncillaryDependencies(unittest.TestCase):

    @patch("imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies._comment_headers")
    @patch("imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.download")
    @patch("imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies.download_external_dependency")
    def test_fetch_dependencies(self, mock_download_external_dependency, mock_download, mock_comment_headers):
        mission = 'imap'
        instrument = 'glows'
        version = 'v001'
        start_date = '20250423'

        uv_anisotropy_factor_path = f'{mission}_{instrument}_uv-anisotropy-1CR_{start_date}_{version}.dat'
        waw_helioion_mp_path = f'{mission}_{instrument}_WawHelioIonMP_{start_date}_{version}.dat'
        bad_days_list_path = f'{mission}_{instrument}_bad-days-list_{start_date}_{version}.dat'
        pipeline_settings_path = f'{mission}_{instrument}_pipeline-settings-l3bcde_{start_date}_{version}.json'

        uv_anisotropy_factor_input = AncillaryInput(uv_anisotropy_factor_path)
        waw_helioion_mp_input = AncillaryInput(waw_helioion_mp_path)
        bad_days_list_input = AncillaryInput(bad_days_list_path)
        pipeline_settings_input = AncillaryInput(pipeline_settings_path)

        processing_input_collection = ProcessingInputCollection(
            uv_anisotropy_factor_input,
            waw_helioion_mp_input,
            bad_days_list_input,
            pipeline_settings_input
        )

        mock_download.return_value = get_test_data_path("glows/imap_glows_pipeline-settings-l3bcde_20250423_v001.json")

        f107_index_path = TEMP_CDF_FOLDER_PATH / "f107_fluxtable.txt"
        lyman_alpha_path = TEMP_CDF_FOLDER_PATH / "lyman_alpha_composite.nc"
        omni_2_path = TEMP_CDF_FOLDER_PATH / "omni2_all_years.dat"

        mock_download_external_dependency.side_effect = [
            f107_index_path,
            lyman_alpha_path,
            omni_2_path
        ]

        actual_dependencies = GlowsInitializerAncillaryDependencies.fetch_dependencies(processing_input_collection)
        self.assertIsInstance(actual_dependencies, GlowsInitializerAncillaryDependencies)
        mock_comment_headers.assert_called_once_with(f107_index_path, num_lines=2)

        data_dir = imap_data_access.config["DATA_DIR"] / "imap" / "ancillary" / "glows"

        mock_download.assert_called_once_with(data_dir / pipeline_settings_path)

        self.assertEqual([
            call(F107_FLUX_TABLE_URL, f107_index_path),
            call(LYMAN_ALPHA_COMPOSITE_INDEX_URL, lyman_alpha_path),
            call(OMNI2_URL, omni_2_path)
        ], mock_download_external_dependency.call_args_list)

        self.assertEqual(str(data_dir / uv_anisotropy_factor_path), actual_dependencies.uv_anisotropy_path)
        self.assertEqual(str(data_dir / waw_helioion_mp_path), actual_dependencies.waw_helioion_mp_path)
        self.assertEqual(str(data_dir / bad_days_list_path), actual_dependencies.bad_days_list)
        self.assertEqual(str(data_dir / pipeline_settings_path), actual_dependencies.pipeline_settings)
        self.assertEqual(f107_index_path, actual_dependencies.f107_index_file_path)
        self.assertEqual(lyman_alpha_path, actual_dependencies.lyman_alpha_path)
        self.assertEqual(omni_2_path, actual_dependencies.omni2_data_path)
        self.assertEqual(TimeDelta(56, format="jd").value, actual_dependencies.initializer_time_buffer.value)

    def test_comment_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = Path(tmpdir) / "flux_table.txt"
            with open(file_name, "w") as fp:
                fp.writelines([
                    "fluxdate    fluxtime    fluxjulian    fluxcarrington  fluxobsflux  fluxadjflux  fluxursi\n",
                    "----------  ----------  ------------  --------------  -----------  -----------  ----------\n",
                    "20041028    170000      02453307.229  002022.605      000132.7     000130.9     000117.8\n",
                    "20041028    200000      02453307.354  002022.610      000135.8     000134.0     000120.6\n"])

            _comment_headers(file_name)
            with open(file_name, "r") as fp:
                lines = fp.readlines()
                self.assertEqual("#", lines[0][0])
                self.assertEqual("#", lines[1][0])
