import dataclasses
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, sentinel, Mock, call, MagicMock

import numpy as np

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.models import CodiceL2HiDirectEventData, CodiceHiL2SectoredIntensitiesData, \
    CodiceL3HiDirectEvents
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.codice.l3.lo.constants import CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import NumpyArrayMatcher, get_test_instrument_team_data_path


class TestCodiceHiProcessor(unittest.TestCase):
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiL3aDirectEventsDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3a_direct_event")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    def test_process_l3a(self, mock_save_data, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3a", start_date, end_date, 'v02')
        mock_processed_direct_events = Mock()
        mock_process_l3a.return_value = mock_processed_direct_events

        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1')]

        processor = CodiceHiProcessor(input_collection, input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_with(input_collection)
        mock_process_l3a.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(mock_processed_direct_events)
        self.assertEqual([mock_save_data.return_value], product)
        self.assertEqual(['parent_file_1'], mock_processed_direct_events.parent_file_names)

    def test_process_l3a_with_small_dataset(self):

        codice_hi_processor = CodiceHiProcessor(sentinel.processing_input_collection,
                                                input_metadata=sentinel.input_metadata)

        expected_number_of_events = np.array([
            [0, 2], [4, 3]
        ])

        expected_ssd_energy = np.array([
            [
                [np.nan, np.nan, np.nan, np.nan],
                [5, 6, np.nan, np.nan]
            ],
            [
                [1, 2, 3, 4],
                [5, 6, 2, np.nan]
            ]
        ], dtype=float)

        expected_energy_per_nuc = np.array([
            [
                [np.nan, np.nan, np.nan, np.nan],
                [15, 16, np.nan, np.nan]
            ],
            [
                [11, 12, 13, 14],
                [15, 16, 12, np.nan]
            ]
        ], dtype=float)

        l2_data = CodiceL2HiDirectEventData(
            epoch=sentinel.expected_epoch,
            epoch_delta_plus=sentinel.expected_epoch_delta_plus,
            data_quality=sentinel.expected_data_quality,
            multi_flag=sentinel.expected_multi_flag,
            number_of_events=expected_number_of_events,
            ssd_energy=expected_ssd_energy,
            ssd_id=np.ones((2, 2, 4)),
            spin_angle=sentinel.expected_spin_angle,
            spin_number=sentinel.expected_spin_number,
            time_of_flight=sentinel.expected_time_of_flight,
            type=sentinel.expected_type,
            energy_per_nuc=expected_energy_per_nuc,
        )
        dependencies = CodiceHiL3aDirectEventsDependencies(
            codice_l2_hi_data=l2_data
        )
        expected_output = CodiceL3HiDirectEvents(
            input_metadata=sentinel.input_metadata,
            epoch=sentinel.expected_epoch,
            epoch_delta=sentinel.expected_epoch_delta_plus,
            data_quality=sentinel.expected_data_quality,
            multi_flag=sentinel.expected_multi_flag,
            num_events=expected_number_of_events,
            ssd_energy=expected_ssd_energy,
            ssd_id=np.ones((2, 2, 4)),
            spin_angle=sentinel.expected_spin_angle,
            spin_number=sentinel.expected_spin_number,
            tof=sentinel.expected_time_of_flight,
            type=sentinel.expected_type,
            energy_per_nuc=expected_energy_per_nuc,
            estimated_mass=np.array([
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [5 / 15, 6 / 16, np.nan, np.nan]
                ],
                [
                    [1 / 11, 2 / 12, 3 / 13, 4 / 14],
                    [5 / 15, 6 / 16, 2 / 12, np.nan]
                ]
            ])
        )

        actual_output = codice_hi_processor.process_l3a_direct_event(dependencies=dependencies)
        for field in dataclasses.fields(CodiceL3HiDirectEvents):
            np.testing.assert_equal(getattr(actual_output, field.name), getattr(expected_output, field.name))

    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodicePitchAngleDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3b")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    def test_process_l3b_saves(self, mock_save_data, mock_process_l3b,
                               mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3b", start_date, end_date, 'v02')

        mock_processed_pitch_angles = Mock()
        mock_process_l3b.return_value = mock_processed_pitch_angles

        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_2')]

        processor = CodiceHiProcessor(input_collection, input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_with(input_collection)
        mock_process_l3b.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(mock_processed_pitch_angles)
        self.assertEqual([mock_save_data.return_value], product)
        self.assertEqual(['parent_file_2'], mock_processed_pitch_angles.parent_file_names)

    def test_raises_exception_on_non_l3_input_metadata(self):
        input_metadata = InputMetadata('codice', "L2a", Mock(), Mock(), 'v02')

        processor = CodiceHiProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as context:
            processor.process()
        self.assertEqual("Unknown data level for CoDICE: L2a", str(context.exception))


    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_unit_vector")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.get_sector_unit_vectors")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_pitch_angle")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_gyrophase")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.rebin_by_pitch_angle_and_gyrophase")
    def test_process_l3b_with_mocks(self, mock_rebin_by_pitch_angle_and_gyrophase, mock_calculate_gyrophases,
                                    mock_calculate_pitch_angles, mock_get_sector_unit_vectors,
                                    mock_calculate_unit_vector):
        rng = np.random.default_rng()
        epoch_1 = datetime(2025, 2, 5)
        epoch_2 = datetime(2025, 2, 6)
        epoch = np.array([epoch_1, epoch_2])
        epoch_delta = np.full(epoch.shape, timedelta(days=.5))

        rebinned_mag_data = rng.random((len(epoch), 3))
        mag_l1d_data = Mock()
        mag_l1d_data.rebin_to.return_value = rebinned_mag_data

        h_intensity = rng.random((len(epoch), len(np.array([1.11, 1.17]) * 1.4), len(np.array([270, 15])), len(
            np.array([15, 45, 75, 105, 135, 165]))))
        he3he4_intensity = rng.random((len(epoch), len(np.array([1.11, 1.17]) * 1.4), len(np.array([270, 15])), len(
            np.array([15, 45, 75, 105, 135, 165]))))
        cno_intensity = rng.random((len(epoch), len(np.array([1.11, 1.17, 1.25])), len(np.array([270, 15])), len(
            np.array([15, 45, 75, 105, 135, 165]))))
        fe_intensity = rng.random(
            (len(epoch), len(np.array([1.11, 1.17]) * 1.3), len(np.array([270, 15])),
             len(np.array([15, 45, 75, 105, 135, 165]))))

        codice_l2_data = CodiceHiL2SectoredIntensitiesData(
            epoch=epoch,
            epoch_delta_plus=epoch_delta,
            data_quality=sentinel.data_quality,
            elevation_angle=(np.array([270, 15])),
            spin_angles=(np.array([15, 45, 75, 105, 135, 165])),
            h_intensities=h_intensity,
            energy_h=(np.array([1.11, 1.17])),
            energy_h_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_h_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            cno_intensities=cno_intensity,
            energy_cno=(np.array([1.11, 1.17, 1.25])),
            energy_cno_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_cno_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            fe_intensities=fe_intensity,
            energy_fe=(np.array([1.11, 1.17]) * 1.3),
            energy_fe_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_fe_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            he3he4_intensities=he3he4_intensity,
            energy_he3he4=(np.array([1.11, 1.17]) * 1.4),
            energy_he3he4_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_he3he4_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
        )

        dependencies = CodicePitchAngleDependencies(mag_l1d_data=mag_l1d_data,
                                                    codice_sectored_intensities_data=codice_l2_data)

        expected_pitch_angles = np.linspace(15, 165, 6)
        expected_gyrophase = np.linspace(15, 345, 12)

        expected_pitch_angle_delta = np.repeat(15, 6)
        expected_gyrophase_delta = np.repeat(15, 12)

        sector_unit_vectors = rng.random((2, 6, 3))
        mag_unit_vectors = rng.random((2, 3))
        mock_calculate_unit_vector.side_effect = [mag_unit_vectors, sector_unit_vectors]

        mock_calculate_gyrophases.side_effect = [
            sentinel.epoch1_gyrophase,
            sentinel.epoch2_gyrophase,
        ]

        mock_calculate_pitch_angles.side_effect = [
            sentinel.epoch1_pitch_angle,
            sentinel.epoch2_pitch_angle,
        ]

        expected_h_intensity_binned_by_pa = rng.random((2, 2, 6))
        expected_he4_intensity_binned_by_pa = rng.random((2, 2, 6))
        expected_cno_intensity_binned_by_pa = rng.random((2, 3, 6))
        expected_fe_intensity_binned_by_pa = rng.random((2, 2, 6))

        expected_h_intensity_binned_by_pa_and_gyro = rng.random((2, 2, 6, 12))
        expected_he4_intensity_binned_by_pa_and_gyro = rng.random((2, 2, 6, 12))
        expected_cno_intensity_binned_by_pa_and_gyro = rng.random((2, 3, 6, 12))
        expected_fe_intensity_binned_by_pa_and_gyro = rng.random((2, 2, 6, 12))

        expected_parents = []

        mock_rebin_by_pitch_angle_and_gyrophase.side_effect = [
            (expected_h_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_h_intensity_binned_by_pa[0], 0, 0),
            (expected_he4_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_he4_intensity_binned_by_pa[0], 0, 0),
            (expected_cno_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_cno_intensity_binned_by_pa[0], 0, 0),
            (expected_fe_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_fe_intensity_binned_by_pa[0], 0, 0),
            (expected_h_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_h_intensity_binned_by_pa[1], 0, 0),
            (expected_he4_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_he4_intensity_binned_by_pa[1], 0, 0),
            (expected_cno_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_cno_intensity_binned_by_pa[1], 0, 0),
            (expected_fe_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_fe_intensity_binned_by_pa[1], 0, 0),
        ]

        codice_processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=sentinel.input_metadata)
        codice_hi_data_product = codice_processor.process_l3b(dependencies=dependencies)

        mock_get_sector_unit_vectors.assert_called_once_with(
            codice_l2_data.elevation_angle,
            NumpyArrayMatcher((codice_l2_data.spin_angles + CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM) % 360))
        mock_calculate_unit_vector.assert_has_calls(
            [call(NumpyArrayMatcher(rebinned_mag_data)), call(mock_get_sector_unit_vectors.return_value)])

        mock_calculate_pitch_angles.assert_has_calls([
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[0])),
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[1])),
        ])

        mock_calculate_gyrophases.assert_has_calls([
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[0])),
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[1])),
        ])

        species_intensities = [h_intensity, he3he4_intensity, cno_intensity, fe_intensity]

        species_uncertainties = [
            np.zeros_like(h_intensity),
            np.zeros_like(he3he4_intensity),
            np.zeros_like(cno_intensity),
            np.zeros_like(fe_intensity),
        ]

        expected_rebin_calls = []
        for species_intensity, species_uncertainty in zip(species_intensities, species_uncertainties):
            expected_rebin_calls.append(
                call(intensity_data=NumpyArrayMatcher(species_intensity[0]),
                     intensity_delta_plus=NumpyArrayMatcher(species_uncertainty[0]),
                     intensity_delta_minus=NumpyArrayMatcher(species_uncertainty[0]),
                     pitch_angles=sentinel.epoch1_pitch_angle,
                     gyrophases=sentinel.epoch1_gyrophase,
                     number_of_pitch_angle_bins=6, number_of_gyrophase_bins=12
                     )

            )

        for species_intensity, species_uncertainty in zip(species_intensities, species_uncertainties):
            expected_rebin_calls.append(
                call(intensity_data=NumpyArrayMatcher(species_intensity[1]),
                     intensity_delta_plus=NumpyArrayMatcher(species_uncertainty[1]),
                     intensity_delta_minus=NumpyArrayMatcher(species_uncertainty[1]),
                     pitch_angles=sentinel.epoch2_pitch_angle,
                     gyrophases=sentinel.epoch2_gyrophase,
                     number_of_pitch_angle_bins=6, number_of_gyrophase_bins=12
                     )
            )

        mock_rebin_by_pitch_angle_and_gyrophase.assert_has_calls(expected_rebin_calls)

        self.assertEqual(sentinel.input_metadata, codice_hi_data_product.input_metadata)
        np.testing.assert_array_equal(epoch, codice_hi_data_product.epoch)
        np.testing.assert_array_equal(epoch_delta, codice_hi_data_product.epoch_delta)

        np.testing.assert_array_equal(codice_hi_data_product.energy_h, codice_l2_data.energy_h)
        np.testing.assert_array_equal(codice_hi_data_product.energy_h_plus, codice_l2_data.energy_h_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_h_minus, codice_l2_data.energy_h_minus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno, codice_l2_data.energy_cno)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno_plus, codice_l2_data.energy_cno_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno_minus, codice_l2_data.energy_cno_minus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe, codice_l2_data.energy_fe)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe_plus, codice_l2_data.energy_fe_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe_minus, codice_l2_data.energy_fe_minus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4, codice_l2_data.energy_he3he4)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4_plus, codice_l2_data.energy_he3he4_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4_minus, codice_l2_data.energy_he3he4_minus)

        np.testing.assert_array_almost_equal(expected_pitch_angles, codice_hi_data_product.pitch_angle)
        np.testing.assert_array_equal(expected_pitch_angle_delta, codice_hi_data_product.pitch_angle_delta)
        np.testing.assert_array_almost_equal(expected_gyrophase, codice_hi_data_product.gyrophase)
        np.testing.assert_array_equal(expected_gyrophase_delta, codice_hi_data_product.gyrophase_delta)

        np.testing.assert_allclose(codice_hi_data_product.h_intensity_by_pitch_angle, expected_h_intensity_binned_by_pa)

        np.testing.assert_array_equal(codice_hi_data_product.he3he4_intensity_by_pitch_angle,
                                      expected_he4_intensity_binned_by_pa)
        np.testing.assert_array_equal(codice_hi_data_product.cno_intensity_by_pitch_angle,
                                      expected_cno_intensity_binned_by_pa)
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle,
                                      expected_fe_intensity_binned_by_pa)

        np.testing.assert_array_equal(codice_hi_data_product.h_intensity_by_pitch_angle_and_gyrophase,
                                      expected_h_intensity_binned_by_pa_and_gyro)
        np.testing.assert_array_equal(codice_hi_data_product.he3he4_intensity_by_pitch_angle_and_gyrophase,
                                      expected_he4_intensity_binned_by_pa_and_gyro)
        np.testing.assert_array_equal(codice_hi_data_product.cno_intensity_by_pitch_angle_and_gyrophase,
                                      expected_cno_intensity_binned_by_pa_and_gyro)
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle_and_gyrophase,
                                      expected_fe_intensity_binned_by_pa_and_gyro)

        np.testing.assert_array_equal(codice_hi_data_product.parent_file_names, expected_parents)

    def test_integration_test(self):
        l2_direct_event_sci_path = get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-direct-events_20250814_v001.cdf")

        codice_hi_dependencies = CodiceHiL3aDirectEventsDependencies.from_file_paths(l2_direct_event_sci_path)

        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='hi-direct-events')
        processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=input_metadata)

        try:
            processor.process_l3a_direct_event(codice_hi_dependencies)
        except Exception as e:
            self.fail(e)

