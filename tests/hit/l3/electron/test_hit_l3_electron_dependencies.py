import unittest
from unittest.mock import call, patch, sentinel

from imap_data_access import AncillaryFilePath, ScienceFilePath
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
)

from imap_l3_processing.hit.l3.electron.hit_l3_electron_dependencies import (
    HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR,
    HIT_L1B_STANDARD_RATES_DESCRIPTOR,
    HitL3ElectronDependencies,
)

MODULE = "imap_l3_processing.hit.l3.electron.hit_l3_electron_dependencies"


class TestHitL3ElectronDependencies(unittest.TestCase):
    @patch(f"{MODULE}.imap_data_access.download")
    @patch(f"{MODULE}.HitL1bIalirtRates.read_from_cdf")
    @patch(f"{MODULE}.ElectronIntensityFactors.from_file")
    def test_fetch_dependencies_downloads_reads_and_exposes_inputs(
        self,
        mock_from_file,
        mock_read_from_cdf,
        mock_download,
    ) -> None:
        mock_download.side_effect = [sentinel.l1b_path, sentinel.factors_path]
        l2_standard_rates = ScienceFilePath(
            f"imap_hit_l2_{HIT_L1B_STANDARD_RATES_DESCRIPTOR}_20260601_v001.cdf"
        )
        l1b_standard_rates = ScienceFilePath(
            f"imap_hit_l1b_{HIT_L1B_STANDARD_RATES_DESCRIPTOR}_20260601_v001.cdf"
        )
        intensity_factors = AncillaryFilePath(
            f"imap_hit_{HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR}_20260513_v001.csv"
        )
        input_collection = ProcessingInputCollection()
        input_collection.add(
            [
                ScienceInput(l2_standard_rates.filename.name),
                ScienceInput(l1b_standard_rates.filename.name),
                AncillaryInput(intensity_factors.filename.name),
            ]
        )

        result = HitL3ElectronDependencies.fetch_dependencies(input_collection)

        self.assertEqual(
            [
                call(l1b_standard_rates.construct_path()),
                call(intensity_factors.construct_path()),
            ],
            mock_download.call_args_list,
        )
        mock_read_from_cdf.assert_called_once_with(sentinel.l1b_path)
        mock_from_file.assert_called_once_with(sentinel.factors_path)
        self.assertEqual(
            HitL3ElectronDependencies(
                ialirt_rates=mock_read_from_cdf.return_value,
                intensity_factors=mock_from_file.return_value,
            ),
            result,
        )

    @patch(f"{MODULE}.imap_data_access.download")
    @patch(f"{MODULE}.HitL1bIalirtRates.read_from_cdf")
    @patch(f"{MODULE}.ElectronIntensityFactors.from_file")
    def test_fetch_dependencies_raises_when_required_input_missing(
        self,
        _mock_from_file,
        _mock_read_from_cdf,
        _mock_download,
    ) -> None:
        cases = [
            (
                HIT_L1B_STANDARD_RATES_DESCRIPTOR,
                AncillaryInput(
                    f"imap_hit_{HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR}_20260513_v001.csv"
                ),
            ),
            (
                HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR,
                ScienceInput(
                    f"imap_hit_l1b_{HIT_L1B_STANDARD_RATES_DESCRIPTOR}_20260601_v001.cdf"
                ),
            ),
        ]
        for descriptor, present_input in cases:
            with self.subTest(missing=descriptor):
                input_collection = ProcessingInputCollection()
                input_collection.add([present_input])
                with self.assertRaises(ValueError) as error:
                    HitL3ElectronDependencies.fetch_dependencies(input_collection)
                self.assertEqual(f"Missing {descriptor} dependency.", str(error.exception))
