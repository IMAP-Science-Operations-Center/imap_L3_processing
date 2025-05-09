import unittest
import warnings
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, call, sentinel, MagicMock

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import CodiceLoL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_partial_densities_dependencies import \
    CodiceLoL3aPartialDensitiesDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct, PriorityEvent, CodiceLoL2SWSpeciesData, \
    CodiceLoL1aSWPriorityRates, CodiceLoL1aNSWPriorityRates, CodiceLoPartialDensityData, CodiceLoL3aRatiosDataProduct, \
    CODICE_LO_L2_NUM_PRIORITIES, CodiceLoL3ChargeStateDistributionsDataProduct
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from tests.test_helpers import NumpyArrayMatcher


class TestCodiceLoProcessor(unittest.TestCase):
    def test_implements_processor(self):
        processor = CodiceLoProcessor(Mock(), Mock())
        self.assertIsInstance(processor, Processor)

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.upload')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoL3aPartialDensitiesDependencies.fetch_dependencies')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor.process_l3a_partial_densities')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_partial_densities(self, mock_spiceypy, mock_save_data, mock_process_l3a_partial_densities,
                                       mock_fetch_dependencies, mock_upload):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1'), Path('path/to/parent_file_2')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-partial-densities')
        mock_spiceypy.ktotal.return_value = 0

        mock_save_data.return_value = "file1"
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a_partial_densities.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_partial_densities.return_value)

        self.assertEqual(['parent_file_1', 'parent_file_2'],
                         mock_process_l3a_partial_densities.return_value.parent_file_names)
        mock_upload.assert_called_once_with("file1")

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.upload')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoL3aRatiosDependencies.fetch_dependencies')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor.process_l3a_ratios')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_ratios(self, mock_spiceypy, mock_save_data, mock_process_l3a_ratios,
                            mock_fetch_dependencies, mock_upload):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-sw-ratios')
        mock_spiceypy.ktotal.return_value = 0

        mock_save_data.return_value = "file1"
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a_ratios.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_ratios.return_value)

        self.assertEqual(['parent_file_1'],
                         mock_process_l3a_ratios.return_value.parent_file_names)
        mock_upload.assert_called_once_with("file1")

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.upload')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoL3aRatiosDependencies.fetch_dependencies')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor.process_l3a_abundances')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_abundances(self, mock_spiceypy, mock_save_data, mock_process_l3a_abundances,
                                mock_fetch_dependencies, mock_upload):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-sw-abundances')
        mock_spiceypy.ktotal.return_value = 0

        mock_save_data.return_value = "file1"
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a_abundances.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_l3a_abundances.return_value)

        self.assertEqual(['parent_file_1'],
                         mock_process_l3a_abundances.return_value.parent_file_names)
        mock_upload.assert_called_once_with("file1")

    def test_process_ratios_calculate_abundance_ratios(self):
        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now, now + timedelta(minutes=4), now + timedelta(minutes=8),
                 now + timedelta(minutes=12), now + timedelta(minutes=16)]),
            epoch_delta=np.array([120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000]),
            fe_hiq_partial_density=np.array([1, 2, 3, 10, 12]),
            fe_loq_partial_density=np.array([4, 5, 6, 11, 13]),
            oplus5_partial_density=np.array([1, 2, 4, 12, 14]),
            oplus6_partial_density=np.array([5, 6, 7, 13, 15]),
            oplus7_partial_density=np.array([8, 9, 10, 14, 16]),
            oplus8_partial_density=np.array([11, 12, 13, 15, 17]),
            mg_partial_density=np.array([14, 15, 16, 16, 18]),
            cplus4_partial_density=np.array([17, 18, 19, 17, 19]),
            cplus5_partial_density=np.array([20, 21, 22, 18, 20]),
            cplus6_partial_density=np.array([23, 24, 25, 19, 21]),
            ne_partial_density=np.arange(5),
            si_partial_density=np.arange(5),
            heplusplus_partial_density=np.arange(5),
            hplus_partial_density=np.arange(5),
        )

        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        ratios_data_product: CodiceLoL3aRatiosDataProduct = processor.process_l3a_ratios(dependency)
        self.assertEqual(input_metadata, ratios_data_product.input_metadata)
        np.testing.assert_array_equal(ratios_data_product.epoch,
                                      [now + timedelta(minutes=4), now + timedelta(minutes=14)])
        np.testing.assert_array_equal(ratios_data_product.epoch_delta, [360_000_000_000, 240_000_000_000])
        np.testing.assert_array_almost_equal(ratios_data_product.fe_to_o_ratio,
                                             np.array(
                                                 [(2 + 5) / (7 / 3 + 6 + 9 + 12), (11 + 12) / (13 + 14 + 15 + 16)]))
        np.testing.assert_array_almost_equal(ratios_data_product.mg_to_o_ratio,
                                             np.array([15 / (7 / 3 + 6 + 9 + 12), 17 / (13 + 14 + 15 + 16)]))
        np.testing.assert_array_almost_equal(ratios_data_product.c_to_o_ratio, np.array(
            [(18 + 21 + 24) / (7 / 3 + 6 + 9 + 12), (18 + 19 + 20) / (13 + 14 + 15 + 16)]))

        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c4_ratio, np.array([24 / 18, 20 / 18]))
        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c5_ratio, np.array([24 / 21, 20 / 19]))

        np.testing.assert_array_almost_equal(ratios_data_product.o7_to_o6_ratio, np.array([9 / 6, 15 / 14]))
        np.testing.assert_array_almost_equal(ratios_data_product.felo_to_fehi_ratio, np.array([5 / 2, 12 / 11]))

    def test_process_ratios_calculate_abundance_ratios_using_safe_divide(self):
        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now, now + timedelta(minutes=4), now + timedelta(minutes=8)]),
            epoch_delta=np.array([120_000_000_000, 120_000_000_000, 120_000_000_000]),
            fe_hiq_partial_density=np.array([0, 0, 0]),
            fe_loq_partial_density=np.array([0, 0, 0]),
            oplus5_partial_density=np.array([0, 0, 0]),
            oplus6_partial_density=np.array([0, 0, 0]),
            oplus7_partial_density=np.array([0, 0, 0]),
            oplus8_partial_density=np.array([0, 0, 0]),
            mg_partial_density=np.array([0, 0, 0]),
            cplus4_partial_density=np.array([0, 0, 0]),
            cplus5_partial_density=np.array([0, 0, 0]),
            cplus6_partial_density=np.array([0, 0, 0]),
            ne_partial_density=np.arange(3),
            si_partial_density=np.arange(3),
            heplusplus_partial_density=np.arange(3),
            hplus_partial_density=np.arange(3),
        )

        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        with warnings.catch_warnings(record=True) as w:
            ratios_data_product = processor.process_l3a_ratios(dependency)
        self.assertEqual(0, len(w))

        self.assertEqual(input_metadata, ratios_data_product.input_metadata)
        np.testing.assert_array_equal(ratios_data_product.epoch,
                                      [now + timedelta(minutes=4)])
        np.testing.assert_array_equal(ratios_data_product.epoch_delta, [360_000_000_000])
        np.testing.assert_array_almost_equal(ratios_data_product.fe_to_o_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.mg_to_o_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.c_to_o_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c4_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.c6_to_c5_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.o7_to_o6_ratio, np.array([np.nan]))
        np.testing.assert_array_almost_equal(ratios_data_product.felo_to_fehi_ratio, np.array([np.nan]))

    def test_process_abundances_calculates_abundance_ratios(self):
        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now, now + timedelta(minutes=4), now + timedelta(minutes=8),
                 now + timedelta(minutes=12), now + timedelta(minutes=16)]),
            epoch_delta=np.array([120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000, 120_000_000_000]),
            fe_hiq_partial_density=np.arange(5),
            fe_loq_partial_density=np.arange(5),
            oplus5_partial_density=np.array([0, 1, 2, 2, 2]),
            oplus6_partial_density=np.array([5, 5, 5, 6, 6]),
            oplus7_partial_density=np.array([8, 8, 8, 9, 9]),
            oplus8_partial_density=np.array([11, 11, 11, 12, 12, ]),
            mg_partial_density=np.arange(5),
            cplus4_partial_density=np.array([16, 17, 18, 18, 18]),
            cplus5_partial_density=np.array([20, 20, 20, 21, 21]),
            cplus6_partial_density=np.array([23, 23, 23, 24, 24]),
            ne_partial_density=np.arange(5),
            si_partial_density=np.arange(5),
            heplusplus_partial_density=np.arange(5),
            hplus_partial_density=np.arange(5),
        )

        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        abundances_data_product = processor.process_l3a_abundances(dependency)
        self.assertIsInstance(abundances_data_product, CodiceLoL3ChargeStateDistributionsDataProduct)
        self.assertEqual(input_metadata, abundances_data_product.input_metadata)
        np.testing.assert_array_equal(abundances_data_product.epoch,
                                      [now + timedelta(minutes=4), now + timedelta(minutes=14)])
        np.testing.assert_array_equal(abundances_data_product.epoch_delta, [360_000_000_000, 240_000_000_000])

        np.testing.assert_array_almost_equal(abundances_data_product.oxygen_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [1 / (1 + 5 + 8 + 11), 5 / (1 + 5 + 8 + 11), 8 / (1 + 5 + 8 + 11),
                                                      11 / (1 + 5 + 8 + 11), ],
                                                     [2 / (2 + 6 + 9 + 12), 6 / (2 + 6 + 9 + 12), 9 / (2 + 6 + 9 + 12),
                                                      12 / (2 + 6 + 9 + 12), ],
                                                 ])
                                             )
        np.testing.assert_array_almost_equal(abundances_data_product.carbon_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [17 / (17 + 20 + 23), 20 / (17 + 20 + 23), 23 / (17 + 20 + 23)],
                                                     [18 / (18 + 21 + 24), 21 / (18 + 21 + 24), 24 / (18 + 21 + 24)],
                                                 ]
                                             ))

    def test_process_abundances_uses_safe_divide(self):

        now = datetime.now()
        data = CodiceLoPartialDensityData(
            epoch=np.array(
                [now]),
            epoch_delta=np.array([120_000_000_000]),
            fe_hiq_partial_density=np.array([1]),
            fe_loq_partial_density=np.array([1]),
            oplus5_partial_density=np.array([0]),
            oplus6_partial_density=np.array([0]),
            oplus7_partial_density=np.array([0]),
            oplus8_partial_density=np.array([0]),
            mg_partial_density=np.array([1]),
            cplus4_partial_density=np.array([0]),
            cplus5_partial_density=np.array([0]),
            cplus6_partial_density=np.array([0]),
            ne_partial_density=np.array([1]),
            si_partial_density=np.array([1]),
            heplusplus_partial_density=np.array([1]),
            hplus_partial_density=np.array([1]),
        )
        dependency = CodiceLoL3aRatiosDependencies(data)
        input_metadata = Mock()
        processor = CodiceLoProcessor(dependencies=Mock(), input_metadata=input_metadata)

        with warnings.catch_warnings(record=True) as w:
            abundances_data_product = processor.process_l3a_abundances(dependency)
        self.assertEqual(0, len(w))

        self.assertIsInstance(abundances_data_product, CodiceLoL3ChargeStateDistributionsDataProduct)
        self.assertEqual(input_metadata, abundances_data_product.input_metadata)
        np.testing.assert_array_equal(abundances_data_product.epoch,
                                      [now])
        np.testing.assert_array_equal(abundances_data_product.epoch_delta,
                                      [120_000_000_000])

        np.testing.assert_array_almost_equal(abundances_data_product.oxygen_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [np.nan, np.nan, np.nan, np.nan]
                                                 ])
                                             )
        np.testing.assert_array_almost_equal(abundances_data_product.carbon_charge_state_distribution,
                                             np.array(
                                                 [
                                                     [np.nan, np.nan, np.nan]
                                                 ]
                                             ))

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.upload')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoL3aDirectEventsDependencies.fetch_dependencies')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor.process_l3a_direct_event_data_product')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.save_data')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_direct_events(self, mock_spiceypy, mock_save_data, mock_process_direct_event,
                                   mock_fetch_dependencies, mock_upload):
        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1'), Path('path/to/parent_file_2')]
        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='lo-direct-events')
        mock_spiceypy.ktotal.return_value = 0

        mock_save_data.return_value = "file1"
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_direct_event.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_called_once_with(mock_process_direct_event.return_value)

        mock_upload.assert_called_once_with("file1")

    def test_raises_exception_on_non_l3_input_metadata(self):
        input_metadata = InputMetadata('codice', "L2a", Mock(), Mock(), 'v02', "bad-descriptor")

        processor = CodiceLoProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as context:
            processor.process()
        self.assertEqual("Unknown data level and descriptor for CoDICE: L2a, bad-descriptor", str(context.exception))

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_partial_densities')
    def test_process_l3a_partial_densities(self, mock_calculate_partial_densities):
        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02',
                                       descriptor='lo-partial-densities')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)

        epochs = np.array([datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)])
        num_species = 14

        codice_lo_l2_data = CodiceLoL2SWSpeciesData(
            *[Mock() for _ in range(len(fields(CodiceLoL2SWSpeciesData)))],
        )

        mass_per_charge_lookup = MassPerChargeLookup(*[i for i in range(len(fields(MassPerChargeLookup)))])

        codice_lo_l2_data.epoch = epochs

        cplus4_partial_density = np.array([1])
        cplus5_partial_density = np.array([2])
        cplus6_partial_density = np.array([3])
        oplus5_partial_density = np.array([4])
        oplus6_partial_density = np.array([5])
        oplus7_partial_density = np.array([6])
        oplus8_partial_density = np.array([7])
        mg_partial_density = np.array([8])
        fe_loq_partial_density = np.array([9])
        fe_hiq_partial_density = np.array([10])

        mock_calculate_partial_densities.side_effect = [
            sentinel.hplus_partial_density,
            sentinel.heplusplus_partial_density,
            cplus4_partial_density,
            cplus5_partial_density,
            cplus6_partial_density,
            oplus5_partial_density,
            oplus6_partial_density,
            oplus7_partial_density,
            oplus8_partial_density,
            sentinel.ne_partial_density,
            mg_partial_density,
            sentinel.si_partial_density,
            fe_loq_partial_density,
            fe_hiq_partial_density,
        ]

        codice_lo_dependencies = CodiceLoL3aPartialDensitiesDependencies(codice_lo_l2_data, mass_per_charge_lookup)
        result_data_product = processor.process_l3a_partial_densities(codice_lo_dependencies)

        self.assertEqual(num_species, mock_calculate_partial_densities.call_count)

        mock_calculate_partial_densities.assert_has_calls(
            [call(codice_lo_l2_data.hplus, codice_lo_l2_data.energy_table, mass_per_charge_lookup.hplus),
             call(codice_lo_l2_data.heplusplus, codice_lo_l2_data.energy_table, mass_per_charge_lookup.heplusplus),
             call(codice_lo_l2_data.cplus4, codice_lo_l2_data.energy_table, mass_per_charge_lookup.cplus4),
             call(codice_lo_l2_data.cplus5, codice_lo_l2_data.energy_table, mass_per_charge_lookup.cplus5),
             call(codice_lo_l2_data.cplus6, codice_lo_l2_data.energy_table, mass_per_charge_lookup.cplus6),
             call(codice_lo_l2_data.oplus5, codice_lo_l2_data.energy_table, mass_per_charge_lookup.oplus5),
             call(codice_lo_l2_data.oplus6, codice_lo_l2_data.energy_table, mass_per_charge_lookup.oplus6),
             call(codice_lo_l2_data.oplus7, codice_lo_l2_data.energy_table, mass_per_charge_lookup.oplus7),
             call(codice_lo_l2_data.oplus8, codice_lo_l2_data.energy_table, mass_per_charge_lookup.oplus8),
             call(codice_lo_l2_data.ne, codice_lo_l2_data.energy_table, mass_per_charge_lookup.ne),
             call(codice_lo_l2_data.mg, codice_lo_l2_data.energy_table, mass_per_charge_lookup.mg),
             call(codice_lo_l2_data.si, codice_lo_l2_data.energy_table, mass_per_charge_lookup.si),
             call(codice_lo_l2_data.fe_loq, codice_lo_l2_data.energy_table, mass_per_charge_lookup.fe_loq),
             call(codice_lo_l2_data.fe_hiq, codice_lo_l2_data.energy_table, mass_per_charge_lookup.fe_hiq)])

        self.assertIsInstance(result_data_product, CodiceLoL3aPartialDensityDataProduct)
        self.assertEqual(input_metadata, result_data_product.input_metadata)
        result_data = result_data_product.data
        self.assertIsInstance(result_data, CodiceLoPartialDensityData)

        np.testing.assert_array_equal(result_data.epoch, codice_lo_l2_data.epoch)
        np.testing.assert_array_equal(result_data.epoch_delta, codice_lo_l2_data.epoch_delta_plus)
        self.assertEqual(sentinel.hplus_partial_density, result_data.hplus_partial_density),
        self.assertEqual(sentinel.heplusplus_partial_density, result_data.heplusplus_partial_density),
        self.assertEqual(cplus4_partial_density, result_data.cplus4_partial_density),
        self.assertEqual(cplus5_partial_density, result_data.cplus5_partial_density),
        self.assertEqual(cplus6_partial_density, result_data.cplus6_partial_density),
        self.assertEqual(oplus5_partial_density, result_data.oplus5_partial_density),
        self.assertEqual(oplus6_partial_density, result_data.oplus6_partial_density),
        self.assertEqual(oplus7_partial_density, result_data.oplus7_partial_density),
        self.assertEqual(oplus8_partial_density, result_data.oplus8_partial_density),
        self.assertEqual(sentinel.ne_partial_density, result_data.ne_partial_density),
        self.assertEqual(mg_partial_density, result_data.mg_partial_density),
        self.assertEqual(sentinel.si_partial_density, result_data.si_partial_density),
        self.assertEqual(fe_loq_partial_density, result_data.fe_loq_partial_density),
        self.assertEqual(fe_hiq_partial_density, result_data.fe_hiq_partial_density),

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_total_number_of_events')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_normalization_ratio')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_mass_per_charge')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_mass')
    def test_process_l3a_direct_events(self, mock_calculate_mass, mock_calculate_mass_per_charge,
                                       mock_calculate_normalization_ratio, mock_total_number_of_events):
        rng = np.random.default_rng()
        sw_priority_rate_parameters = {f.name: None for f in fields(CodiceLoL1aSWPriorityRates)}
        nsw_priority_rate_parameters = {f.name: None for f in fields(CodiceLoL1aNSWPriorityRates)}

        sw_priority_rates = CodiceLoL1aSWPriorityRates(
            **sw_priority_rate_parameters,
        )

        nsw_priority_rates = CodiceLoL1aNSWPriorityRates(
            **nsw_priority_rate_parameters,
        )

        epochs = np.array([datetime.now(), datetime.now() + timedelta(hours=1)])
        event_num = np.arange(10)

        sw_priority_rates.epoch = epochs
        nsw_priority_rates.epoch = epochs
        (sw_priority_rates.acquisition_time_per_step,
         sw_priority_rates.p0_tcrs,
         sw_priority_rates.p1_hplus,
         sw_priority_rates.p2_heplusplus,
         sw_priority_rates.p3_heavies,
         sw_priority_rates.p4_dcrs,
         nsw_priority_rates.p5_heavies,
         nsw_priority_rates.p6_hplus_heplusplus) = [rng.random((len(epochs), 2, 2)) for _ in range(8)]

        expected_total_events = np.arange(1, 17).reshape(8, 2)
        expected_total_by_energy_and_spin_angle = [rng.random((128, 12)) for _ in range(16)]

        mass_per_charge_side_effect = [rng.random((len(epochs), len(event_num))) for _ in range(7)]
        mass_side_effect = [rng.random((len(epochs), len(event_num))) for _ in range(7)]

        (expected_mass,
         expected_mass_per_charge,
         expected_apd_energy,
         expected_apd_gain,
         expected_apd_id,
         expected_multi_flag,
         expected_pha_type,
         expected_tof) = [np.full((len(epochs), 7, len(event_num)), np.nan) for _ in range(8)]

        (expected_data_quality,
         expected_num_events) = [np.full((len(epochs), 7), np.nan) for _ in range(2)]

        mock_calculate_mass_per_charge.side_effect = mass_per_charge_side_effect
        mock_calculate_mass.side_effect = mass_side_effect
        mock_total_number_of_events.side_effect = expected_total_events
        mock_calculate_normalization_ratio.side_effect = expected_total_by_energy_and_spin_angle

        priority_events = []
        for i, (mass_per_charge, mass) in enumerate(
                zip(mass_per_charge_side_effect, mass_side_effect)):
            priority_event_kwargs = {f.name: rng.random((len(epochs), len(event_num))) for f in fields(PriorityEvent)}
            priority_event = PriorityEvent(**priority_event_kwargs)
            priority_event.data_quality = rng.random((len(epochs)))
            priority_event.num_events = rng.random((len(epochs)))
            priority_event.total_events_binned_by_energy_step_and_spin_angle = Mock()

            priority_event.total_events_binned_by_energy_step_and_spin_angle.return_value = [{f"mock_p{i % 1}": i % 1}
                                                                                             for i in
                                                                                             range(len(epochs))]
            priority_events.append(priority_event)

            expected_apd_energy[:, i, :] = np.copy(priority_event.apd_energy)
            expected_apd_gain[:, i, :] = np.copy(priority_event.apd_gain)
            expected_apd_id[:, i, :] = np.copy(priority_event.apd_id)
            expected_multi_flag[:, i, :] = np.copy(priority_event.multi_flag)
            expected_tof[:, i, :] = np.copy(priority_event.tof)
            expected_mass[:, i, :] = np.copy(mass)
            expected_mass_per_charge[:, i, :] = np.copy(mass_per_charge)

            expected_data_quality[:, i] = np.copy(priority_event.data_quality)
            expected_num_events[:, i] = np.copy(priority_event.num_events)

        empty_priority_7 = PriorityEvent(
            **{f.name: rng.random((len(epochs), len(event_num))) for f in fields(PriorityEvent)})
        priority_events.append(empty_priority_7)
        direct_events = CodiceLoL2DirectEventData(epochs, np.array([]), np.array([]), priority_events)

        dependencies = CodiceLoL3aDirectEventsDependencies(sw_priority_rates, nsw_priority_rates, direct_events, Mock())

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        l3a_direct_event_data_product = processor.process_l3a_direct_event_data_product(dependencies)

        priority_index = np.arange(CODICE_LO_L2_NUM_PRIORITIES)

        expected_calculate_normalization_calls = []
        expected_calculate_mass_calls = []
        expected_calculate_mass_per_charge_calls = []
        for index in range(0, len(direct_events.priority_events) - 1):
            priority_event = direct_events.priority_events[index]
            epoch_call_1 = call(priority_event.total_events_binned_by_energy_step_and_spin_angle.return_value[0],
                                expected_total_events[index][0])
            epoch_call_2 = call(priority_event.total_events_binned_by_energy_step_and_spin_angle.return_value[0],
                                expected_total_events[index][1])
            expected_calculate_normalization_calls.extend([epoch_call_1, epoch_call_2])
            expected_calculate_mass_calls.append(call(priority_event, dependencies.mass_coefficient_lookup))
            expected_calculate_mass_per_charge_calls.append(call(priority_event))

            expected_epoch_1 = index * 2
            expected_epoch_2 = index * 2 + 1
            np.testing.assert_array_equal(l3a_direct_event_data_product.normalization[0][index],
                                          expected_total_by_energy_and_spin_angle[expected_epoch_1])
            np.testing.assert_array_equal(l3a_direct_event_data_product.normalization[1][index],
                                          expected_total_by_energy_and_spin_angle[expected_epoch_2])

        self.assertIsInstance(l3a_direct_event_data_product, CodiceLoL3aDirectEventDataProduct)
        self.assertEqual(input_metadata, l3a_direct_event_data_product.input_metadata)

        np.testing.assert_array_equal(epochs, l3a_direct_event_data_product.epoch)
        np.testing.assert_array_equal(direct_events.epoch_delta_plus, l3a_direct_event_data_product.epoch_delta)

        np.testing.assert_array_equal(priority_index, l3a_direct_event_data_product.priority_index)
        np.testing.assert_array_equal(expected_mass_per_charge, l3a_direct_event_data_product.mass_per_charge)
        np.testing.assert_array_equal(expected_mass, l3a_direct_event_data_product.mass)

        mock_calculate_normalization_ratio.assert_has_calls(expected_calculate_normalization_calls)
        mock_calculate_mass.assert_has_calls(expected_calculate_mass_calls)

        mock_total_number_of_events.assert_has_calls([
            call(NumpyArrayMatcher(sw_priority_rates.p0_tcrs),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step)),

            call(NumpyArrayMatcher(sw_priority_rates.p1_hplus),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step)),

            call(NumpyArrayMatcher(sw_priority_rates.p2_heplusplus),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step)),

            call(NumpyArrayMatcher(sw_priority_rates.p3_heavies),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step)),

            call(NumpyArrayMatcher(sw_priority_rates.p4_dcrs),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step)),

            call(NumpyArrayMatcher(nsw_priority_rates.p5_heavies),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step)),

            call(NumpyArrayMatcher(nsw_priority_rates.p6_hplus_heplusplus),
                 NumpyArrayMatcher(sw_priority_rates.acquisition_time_per_step))

        ], any_order=False)

        np.testing.assert_array_equal(expected_apd_energy, l3a_direct_event_data_product.event_energy)
        np.testing.assert_array_equal(expected_apd_gain, l3a_direct_event_data_product.gain)
        np.testing.assert_array_equal(expected_apd_id, l3a_direct_event_data_product.apd_id)
        np.testing.assert_array_equal(expected_multi_flag, l3a_direct_event_data_product.multi_flag)
        np.testing.assert_array_equal(expected_num_events, l3a_direct_event_data_product.num_events)
        np.testing.assert_array_equal(expected_data_quality, l3a_direct_event_data_product.data_quality)
        np.testing.assert_array_equal(expected_tof, l3a_direct_event_data_product.tof)


if __name__ == '__main__':
    unittest.main()
