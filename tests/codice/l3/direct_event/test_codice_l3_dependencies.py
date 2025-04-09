import unittest
from pathlib import Path
from unittest.mock import patch

from imap_l3_processing.codice.l3.direct_event.codice_l3_dependencies import CodiceL3Dependencies


class TestCodiceL2Dependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.models.CodiceL2HiData.read_from_cdf")
    @patch("imap_l3_processing.codice.l3.direct_event.science.tof_lookup.TOFLookup.read_from_file")
    def test_can_load_from_files(self, mock_tof_lookup_from_file, mock_read_cdf):
        tof_lookup_file = Path("energy_per_nuc")
        codice_l2_cdf_file = Path("CodiceL2CDF")

        codice_l3_dependencies = CodiceL3Dependencies.from_file_paths(codice_l2_cdf_file, tof_lookup_file)

        mock_tof_lookup_from_file.assert_called_with(tof_lookup_file)
        mock_read_cdf.assert_called_with(codice_l2_cdf_file)

        self.assertEqual(mock_tof_lookup_from_file.return_value, codice_l3_dependencies.tof_lookup)
        self.assertEqual(mock_read_cdf.return_value, codice_l3_dependencies.codice_l2_hi_data)
