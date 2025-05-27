import unittest
from unittest.mock import patch, sentinel, call

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies, MASS_SPECIES_BIN_LOOKUP_DESCRIPTOR, GEOMETRIC_FACTOR_LOOKUP_DESCRIPTOR, \
    ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import SW_PRIORITY_DESCRIPTOR, \
    NSW_PRIORITY_DESCRIPTOR


class TestCodiceLoL3a3dDistributions(unittest.TestCase):
    def test_fetch_dependencies_multiple_species(self):
        for species in ["heplus", "oplus6", "hplus", "heplus2"]:
            with self.subTest(species):
                self._test_fetch_dependencies(species)

    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.imap_data_access.download")
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoL3a3dDistributionsDependencies.from_file_paths")
    def _test_fetch_dependencies(self, species: str, mock_3d_distribution_deps_from_file_paths,
                                 mock_data_access_download):
        l3a_direct_event_name = "imap_codice_l3a_lo-direct-events_20100105_v010.cdf"
        l1a_sw_priority_name = f"imap_codice_l1a_{SW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        l1a_nsw_priority_name = f"imap_codice_l1a_{NSW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        mass_species_lut_name = f"imap_codice_{MASS_SPECIES_BIN_LOOKUP_DESCRIPTOR}_20100105_v010.csv"
        geometric_factors_lut_name = f"imap_codice_{GEOMETRIC_FACTOR_LOOKUP_DESCRIPTOR}_20100105_v010.csv"
        efficiency_factors_lut_name = f"imap_codice_lo-{species}-efficiency_20100105_v010.csv"
        energy_per_charge_lut_name = f"imap_codice_lo-{ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR}_20100105_v010.csv"

        unused_science_input = ScienceInput(f"imap_codice_l2_lo-direct-events_20100105_v010.cdf")

        processing_input_collection = ProcessingInputCollection(AncillaryInput(mass_species_lut_name),
                                                                AncillaryInput(geometric_factors_lut_name),
                                                                AncillaryInput(efficiency_factors_lut_name),
                                                                AncillaryInput(energy_per_charge_lut_name),
                                                                ScienceInput(l3a_direct_event_name),
                                                                ScienceInput(l1a_sw_priority_name),
                                                                ScienceInput(l1a_nsw_priority_name),
                                                                unused_science_input)

        mock_data_access_download.side_effect = [
            sentinel.l3a_direct_event_downloaded_path,
            sentinel.l1a_sw_priority_downloaded_path,
            sentinel.l1a_nsw_priority_downloaded_path,
            sentinel.mass_species_lut_downloaded_path,
            sentinel.geometric_factors_lut_downloaded_path,
            sentinel.efficiency_factors_lut_downloaded_path,
            sentinel.energy_per_charge_downloaded_path,
        ]

        CodiceLoL3a3dDistributionsDependencies.fetch_dependencies(processing_input_collection, species)

        mock_data_access_download.assert_has_calls([
            call(l3a_direct_event_name),
            call(l1a_sw_priority_name),
            call(l1a_nsw_priority_name),
            call(mass_species_lut_name),
            call(geometric_factors_lut_name),
            call(efficiency_factors_lut_name),
            call(energy_per_charge_lut_name)
        ])

        mock_3d_distribution_deps_from_file_paths.assert_called_once_with(
            l3a_file_path=sentinel.l3a_direct_event_downloaded_path,
            l1a_sw_file_path=sentinel.l1a_sw_priority_downloaded_path,
            l1a_nsw_file_path=sentinel.l1a_nsw_priority_downloaded_path,
            mass_species_bin_lut=sentinel.mass_species_lut_downloaded_path,
            geometric_factors_lut=sentinel.geometric_factors_lut_downloaded_path,
            efficiency_factors_lut=sentinel.efficiency_factors_lut_downloaded_path,
            energy_per_charge_lut=sentinel.energy_per_charge_downloaded_path,
            species=species,
        )

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.EfficiencyLookup.read_from_csv')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.GeometricFactorLookup.read_from_csv')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.MassSpeciesBinLookup.read_from_csv')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoL1aSWPriorityRates.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoL1aNSWPriorityRates.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoDirectEventData.read_from_cdf')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.EnergyLookup.read_from_csv')
    def test_from_file_paths(self, mock_energy_lookup_from_csv, mock_l3a_direct_event_read_from_cdf,
                             mock_l1a_nsw_read_from_cdf,
                             mock_l1a_sw_read_from_cdf, mock_mass_species_from_csv, mock_geometric_factor_from_csv,
                             mock_efficiency_lut_read_from_csv):
        actual = CodiceLoL3a3dDistributionsDependencies.from_file_paths(
            l3a_file_path=sentinel.l3a_path,
            l1a_sw_file_path=sentinel.l1a_sw_path,
            l1a_nsw_file_path=sentinel.l1a_nsw_path,
            mass_species_bin_lut=sentinel.mass_species_path,
            geometric_factors_lut=sentinel.geometric_factor_path,
            efficiency_factors_lut=sentinel.efficiency_factor_path,
            energy_per_charge_lut=sentinel.energy_per_charge_path,
            species="some-species"
        )
        mock_l3a_direct_event_read_from_cdf.assert_called_once_with(sentinel.l3a_path)
        mock_l1a_nsw_read_from_cdf.assert_called_once_with(sentinel.l1a_nsw_path)
        mock_l1a_sw_read_from_cdf.assert_called_once_with(sentinel.l1a_sw_path)
        mock_mass_species_from_csv.assert_called_once_with(sentinel.mass_species_path)
        mock_geometric_factor_from_csv.assert_called_once_with(sentinel.geometric_factor_path)
        mock_efficiency_lut_read_from_csv.assert_called_once_with(sentinel.efficiency_factor_path)
        mock_energy_lookup_from_csv.assert_called_once_with(sentinel.energy_per_charge_path)

        expected_dependencies = CodiceLoL3a3dDistributionsDependencies(
            l3a_direct_event_data=mock_l3a_direct_event_read_from_cdf.return_value,
            l1a_sw_data=mock_l1a_sw_read_from_cdf.return_value,
            l1a_nsw_data=mock_l1a_nsw_read_from_cdf.return_value,
            mass_species_bin_lookup=mock_mass_species_from_csv.return_value,
            geometric_factors_lookup=mock_geometric_factor_from_csv.return_value,
            efficiency_factors_lut=mock_efficiency_lut_read_from_csv.return_value,
            energy_per_charge_lut=mock_energy_lookup_from_csv.return_value,
            species="some-species")
        self.assertEqual(expected_dependencies, actual)
