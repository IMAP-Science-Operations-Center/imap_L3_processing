import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection

from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData


class TestHiL3SpectralFitDependencies(unittest.TestCase):

    @patch(
        "imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies.RectangularIntensityMapData.read_from_path")
    def test_from_file_paths(self, mock_read_from_path):
        hi_l3 = Path("test_hi_l3_cdf.cdf")

        result = HiL3SpectralFitDependencies.from_file_paths(hi_l3)

        mock_read_from_path.assert_called_with(hi_l3)

        self.assertEqual(mock_read_from_path.call_count, 1)
        self.assertEqual(mock_read_from_path.return_value, result.hi_l3_data)

    @patch("imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies.imap_data_access.download")
    @patch(
        "imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies.RectangularIntensityMapData.read_from_path")
    def test_fetch_dependencies(self, mock_read_from_path, mock_download):
        file_name = "imap_hi_l3_h90-spx-h-hf-sp-full-hae-4deg-6mo_20250422_v001.cdf"
        processing_input = ProcessingInputCollection(
            ScienceInput(file_name))

        dependency = HiL3SpectralFitDependencies.fetch_dependencies(processing_input)

        mock_download.assert_called_with(file_name)
        mock_read_from_path.assert_called_with(mock_download.return_value)
        self.assertEqual(mock_read_from_path.return_value, dependency.hi_l3_data)

    def test_raises_value_error_if_instrument_doesnt_match(self):
        file_name = "imap_lo_l3_h90-spx-h-hf-sp-full-hae-4deg-6mo_20250422_v001.cdf"
        processing_input = ProcessingInputCollection(
            ScienceInput(file_name))

        with self.assertRaises(ValueError) as error:
            _ = HiL3SpectralFitDependencies.fetch_dependencies(processing_input)

        self.assertEqual(str(error.exception), "Missing Hi dependency.")

    def test_fit_energy_ranges_returns_full_energy_range(self):
        energy = np.arange(1, 4)
        energy_delta_minus = np.full((1,), 0.1)
        energy_delta_plus = np.full((1,), 0.2)

        map_data = _create_intensity_map_data(energy=energy, energy_delta_plus=energy_delta_plus,
                                              energy_delta_minus=energy_delta_minus)
        data = RectangularIntensityMapData(map_data, Mock())
        deps = HiL3SpectralFitDependencies(data)

        np.testing.assert_array_equal(deps.get_fit_energy_ranges(), [[0.9, 3.2]])


def _create_intensity_map_data(ena_intensity=None, energy=None, energy_delta_plus=None, energy_delta_minus=None):
    return IntensityMapData(
        ena_intensity=ena_intensity if ena_intensity is not None else np.full((1, 1, 8, 15), 1),
        ena_intensity_stat_unc=None,
        ena_intensity_sys_err=None,
        epoch=None,
        epoch_delta=None,
        energy=energy if energy is not None else np.full((1,), 1),
        energy_delta_plus=energy_delta_plus if energy_delta_plus is not None else np.full((1,), 0.1),
        energy_delta_minus=energy_delta_minus if energy_delta_minus is not None else np.full((1,), 0.1),
        energy_label=None,
        latitude=None,
        longitude=None,
        exposure_factor=None,
        obs_date=None,
        obs_date_range=None,
        solid_angle=None,
    )
