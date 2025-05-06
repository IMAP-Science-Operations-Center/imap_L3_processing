import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies, SW_SPECIES_DESCRIPTOR, MASS_PER_CHARGE_DESCRIPTOR


class TestCodiceLoL3aPartialDensitiesDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies.imap_data_access.download")
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies.CodiceLoL3aPartialDensitiesDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_imap_data_access_download):
        expected_codice_science_file_download_path = f"imap/codice/l2/2010/01/imap_codice_l2_{SW_SPECIES_DESCRIPTOR}_20100105_v010.cdf"
        expected_mass_per_charge_file_download_path = f"imap/ancillary/codice/imap_codice_{MASS_PER_CHARGE_DESCRIPTOR}_20100105_v001.csv"

        sectored_science_input = ScienceInput(f"imap_codice_l2_{SW_SPECIES_DESCRIPTOR}_20100105_v010.cdf")
        ancillary_input = AncillaryInput(f"imap_codice_{MASS_PER_CHARGE_DESCRIPTOR}_20100105_v001.csv")

        non_codice_ancillary_input = AncillaryInput("imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")

        processing_input_collection = ProcessingInputCollection(
            sectored_science_input, ancillary_input,
            non_codice_ancillary_input, non_codice_science_input,
            non_l2_codice_science_input)

        expected_local_codice_science_file_path = Path("some_local_folder") / expected_codice_science_file_download_path
        expected_local_mass_per_charge_file_download_path = Path(
            "some_local_folder") / expected_mass_per_charge_file_download_path

        mock_imap_data_access_download.side_effect = [
            expected_local_codice_science_file_path,
            expected_local_mass_per_charge_file_download_path
        ]

        codice_l3_dependencies = CodiceLoL3aPartialDensitiesDependencies.fetch_dependencies(processing_input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]
        expected_download_sectored_science_path = data_dir / expected_codice_science_file_download_path
        expected_download_ancillary_path = data_dir / expected_mass_per_charge_file_download_path

        mock_imap_data_access_download.assert_has_calls([
            call(expected_download_sectored_science_path.name),
            call(expected_download_ancillary_path.name),

        ], any_order=True)

        mock_from_file_paths.assert_called_with(expected_local_codice_science_file_path,
                                                expected_local_mass_per_charge_file_download_path)
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies.CodiceLoL2SWSpeciesData.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies.MassPerChargeLookup.read_from_file')
    def test_from_file_paths(self, mock_mpc_read_from_file, mock_read_from_cdf):
        actual_dependencies = CodiceLoL3aPartialDensitiesDependencies.from_file_paths(
            sentinel.cdf_path, sentinel.mpc_lookup_path)

        mock_mpc_read_from_file.assert_called_once_with(sentinel.mpc_lookup_path)
        mock_read_from_cdf.assert_called_once_with(sentinel.cdf_path)

        self.assertEqual(mock_read_from_cdf.return_value, actual_dependencies.codice_l2_lo_data)
        self.assertEqual(mock_mpc_read_from_file.return_value, actual_dependencies.mass_per_charge_lookup)


if __name__ == '__main__':
    unittest.main()
