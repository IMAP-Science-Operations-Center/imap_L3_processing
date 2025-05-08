import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, sentinel, Mock, call

import numpy as np

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.direct_event.science.tof_lookup import TOFLookup, EnergyPerNuc
from imap_l3_processing.codice.l3.hi.models import PriorityEventL2, CodiceL2HiData, CodiceHiL2SectoredIntensitiesData, \
    CODICE_HI_NUM_L2_PRIORITIES
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.models import InputMetadata, MagL1dData
from tests.test_helpers import NumpyArrayMatcher


class TestCodiceHiProcessor(unittest.TestCase):
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiL3aDirectEventsDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3a_direct_event")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.upload")
    def test_process_l3a(self, mock_upload, mock_save_data, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3a", start_date, end_date, 'v02')
        mock_processed_direct_events = Mock()
        mock_process_l3a.return_value = mock_processed_direct_events
        mock_expected_cdf = Mock()
        mock_save_data.return_value = mock_expected_cdf

        processor = CodiceHiProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(sentinel.processing_input_collection)
        mock_process_l3a.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(mock_processed_direct_events)
        mock_upload.assert_called_with(mock_expected_cdf)

    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodicePitchAngleDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3b")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.upload")
    def test_process_l3b_saves_and_uploads(self, mock_upload, mock_save_data, mock_process_l3b,
                                           mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3b", start_date, end_date, 'v02')
        mock_process_l3b.return_value = sentinel.mock_processed_pitch_angles
        mock_expected_cdf = Mock()
        mock_save_data.return_value = mock_expected_cdf

        processor = CodiceHiProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(sentinel.processing_input_collection)
        mock_process_l3b.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(sentinel.mock_processed_pitch_angles)
        mock_upload.assert_called_with(mock_expected_cdf)

    def test_raises_exception_on_non_l3_input_metadata(self):
        input_metadata = InputMetadata('codice', "L2a", Mock(), Mock(), 'v02')

        processor = CodiceHiProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as context:
            processor.process()
        self.assertEqual("Unknown data level for CoDICE: L2a", str(context.exception))

    def test_process_l3a_returns_data_product(self):
        epoch = np.array([datetime(2025, 1, 1), datetime(2025, 1, 1)])
        epoch_delta_plus = np.full(epoch.shape, 1_000_000)

        l2_priority_events, (reshaped_l2_data_quality,
                             reshaped_l2_multi_flag,
                             reshaped_l2_num_events,
                             reshaped_l2_ssd_energy,
                             reshaped_l2_ssd_energy_plus,
                             reshaped_l2_ssd_energy_minus,
                             reshaped_l2_ssd_id,
                             reshaped_l2_spin_angle,
                             reshaped_l2_spin_number,
                             reshaped_l2_time_of_flight,
                             reshaped_l2_type) = self._create_priority_events()

        l2_data = CodiceL2HiData(epoch, epoch_delta_plus, l2_priority_events)
        multiply_by_100_energy_per_nuc_lookup = TOFLookup(
            {i: EnergyPerNuc(i * 10, i * 100, i * 1000) for i in np.arange(1, 25)})
        dependencies = CodiceHiL3aDirectEventsDependencies(tof_lookup=multiply_by_100_energy_per_nuc_lookup,
                                                           codice_l2_hi_data=l2_data)

        expected_energy_per_nuc = reshaped_l2_time_of_flight * 100

        processor = CodiceHiProcessor(Mock(), Mock())
        codice_direct_event_product = processor.process_l3a_direct_event(dependencies)

        np.testing.assert_array_equal(codice_direct_event_product.epoch, l2_data.epoch)
        np.testing.assert_array_equal(codice_direct_event_product.epoch_delta, l2_data.epoch_delta_plus)

        np.testing.assert_array_equal(codice_direct_event_product.data_quality, reshaped_l2_data_quality)
        np.testing.assert_array_equal(codice_direct_event_product.multi_flag, reshaped_l2_multi_flag)
        np.testing.assert_array_equal(codice_direct_event_product.num_events, reshaped_l2_num_events)

        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy, reshaped_l2_ssd_energy)
        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy_plus, reshaped_l2_ssd_energy_plus)
        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy_minus, reshaped_l2_ssd_energy_minus)

        np.testing.assert_array_equal(codice_direct_event_product.ssd_id, reshaped_l2_ssd_id)
        np.testing.assert_array_equal(codice_direct_event_product.spin_angle, reshaped_l2_spin_angle)
        np.testing.assert_array_equal(codice_direct_event_product.spin_number, reshaped_l2_spin_number)
        np.testing.assert_array_equal(codice_direct_event_product.tof, reshaped_l2_time_of_flight)
        np.testing.assert_array_equal(codice_direct_event_product.type, reshaped_l2_type)

        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy / expected_energy_per_nuc,
                                      codice_direct_event_product.estimated_mass)

        np.testing.assert_array_equal(expected_energy_per_nuc, codice_direct_event_product.energy_per_nuc)

    def test_process_l3b_drift_test(self):
        rng = np.random.default_rng()
        epoch_1 = datetime(2025, 2, 5)
        epoch_2 = datetime(2025, 2, 6)
        epoch_3 = datetime(2025, 2, 7)

        mag_l1d_data = MagL1dData(
            epoch=np.array([epoch_1, epoch_1 + timedelta(days=0.49), epoch_2, epoch_2 + timedelta(days=0.49), epoch_3,
                            epoch_3 + timedelta(days=0.49)]),
            mag_data=np.array(
                [
                    [.5, .25, .25], [.5, .25, .25],
                    [.5, .5, 0], [.5, .5, 0],
                    [.5, .25, .25], [.5, .25, .25],
                ]),
        )
        spin_sector = np.array([15, 45, 75, 105, 135, 165])
        ssd_id = np.array([270, 15])

        epoch = np.array([epoch_1, epoch_2, epoch_3])
        epoch_delta = np.full(epoch.shape, timedelta(days=.5))
        energy = np.array([1.11, 1.17])
        energy_h = energy
        energy_cno = np.array([1.11, 1.17, 1.25])
        energy_fe = energy * 1.3
        energy_he3he4 = energy * 1.4
        energy_delta_plus = np.repeat(1.6, len(energy))

        h_intensity = np.array([
            [np.arange(1, 13).reshape(2, 6),
             np.arange(1, 13).reshape(2, 6) * 2, ],
            [np.arange(6, 18).reshape(2, 6),
             np.arange(6, 18).reshape(2, 6) * 2, ],
            [np.arange(11, 23).reshape(2, 6),
             np.arange(11, 23).reshape(2, 6) * 2, ],
        ])
        he3he4_intensity = rng.random((len(epoch), len(energy_he3he4), len(ssd_id), len(spin_sector)))
        cno_intensity = rng.random((len(epoch), len(energy_cno), len(ssd_id), len(spin_sector)))
        fe_intensity = rng.random((len(epoch), len(energy_fe), len(ssd_id), len(spin_sector)))

        codice_l2_data = CodiceHiL2SectoredIntensitiesData(
            epoch=epoch,
            epoch_delta_plus=epoch_delta,
            data_quality=sentinel.data_quality,
            ssd_index=ssd_id,
            spin_sector_index=spin_sector,
            h_intensities=h_intensity,
            energy_h=energy_h,
            energy_h_delta=energy_delta_plus,
            cno_intensities=cno_intensity,
            energy_cno=energy_cno,
            energy_cno_delta=energy_delta_plus,
            fe_intensities=fe_intensity,
            energy_fe=energy_fe,
            energy_fe_delta=energy_delta_plus,
            he3he4_intensities=he3he4_intensity,
            energy_he3he4=energy_he3he4,
            energy_he3he4_delta=energy_delta_plus,
        )

        dependencies = CodicePitchAngleDependencies(mag_l1d_data=mag_l1d_data,
                                                    codice_sectored_intensities_data=codice_l2_data)
        codice_processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=sentinel.input_metadata)

        expected_pitch_angles = np.linspace(15, 165, 6)
        expected_gyrophase = np.linspace(15, 345, 12)

        expected_pitch_angle_delta = np.repeat(15, 6)
        expected_gyrophase_delta = np.repeat(15, 12)

        codice_hi_data_product = codice_processor.process_l3b(dependencies=dependencies)
        self.assertEqual(sentinel.input_metadata, codice_hi_data_product.input_metadata)
        np.testing.assert_array_equal(epoch, codice_hi_data_product.epoch)
        np.testing.assert_array_equal(epoch_delta, codice_hi_data_product.epoch_delta)

        np.testing.assert_array_equal(codice_hi_data_product.energy_h, codice_l2_data.energy_h)
        np.testing.assert_array_equal(codice_hi_data_product.energy_h_delta, codice_l2_data.energy_h_delta)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno, codice_l2_data.energy_cno)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno_delta, codice_l2_data.energy_cno_delta)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe, codice_l2_data.energy_fe)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe_delta, codice_l2_data.energy_fe_delta)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4, codice_l2_data.energy_he3he4)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4_delta, codice_l2_data.energy_he3he4_delta)

        np.testing.assert_array_almost_equal(expected_pitch_angles, codice_hi_data_product.pitch_angle)
        np.testing.assert_array_equal(expected_pitch_angle_delta, codice_hi_data_product.pitch_angle_delta)
        np.testing.assert_array_almost_equal(expected_gyrophase, codice_hi_data_product.gyrophase)
        np.testing.assert_array_equal(expected_gyrophase_delta, codice_hi_data_product.gyrophase_delta)

        np.testing.assert_allclose(codice_hi_data_product.h_intensity_by_pitch_angle, np.array([
            [[np.nan, 9.00000, 8.50000, 4.66667, 5.00000, 5.00000],
             [np.nan, 18.00000, 17.00000, 9.33333, 10.00000, 10.00000]],
            [[np.nan, 11.00000, 11.00000, 12.00000, 12.00000, np.nan],
             [np.nan, 22.00000, 22.00000, 24.00000, 24.00000, np.nan]
             ],
            [[np.nan, 19.00000, 18.50000, 14.66667, 15.00000, 15.00000],
             [np.nan, 38.00000, 37.00000, 29.33333, 30.00000, 30.00000]
             ]
        ]), rtol=1e-6)

        np.testing.assert_array_equal(codice_hi_data_product.he4_intensity_by_pitch_angle,
                                      np.array([[[np.nan, 0.36652964, 0.3874943, 0.50275126, 0.44873382, 0.42571702],
                                                 [np.nan, 0.64013699, 0.87564106, 0.32029546, 0.5962911, 0.68864472]],
                                                [[np.nan, 0.56471798, 0.68419308, 0.14134164, 0.69236274, np.nan],
                                                 [np.nan, 0.27865903, 0.5541061, 0.13706179, 0.4226635, np.nan]],
                                                [[np.nan, 0.34978144, 0.50717228, 0.71996217, 0.54886183, 0.31439651],
                                                 [np.nan, 0.20783655, 0.94472064, 0.5002713, 0.39946124, 0.22996017]]]))
        np.testing.assert_array_equal(codice_hi_data_product.o_intensity_by_pitch_angle,
                                      np.array([[[np.nan, 0.5995281, 0.36612431, 0.50505372, 0.47808497, 0.92685677],
                                                 [np.nan, 0.45173388, 0.41625021, 0.4255396, 0.67226001, 0.39181205],
                                                 [np.nan, 0.61371052, 0.61126329, 0.29613694, 0.28298212, 0.1670665]],
                                                [[np.nan, 0.24808064, 0.65803097, 0.51308025, 0.43438923, np.nan],
                                                 [np.nan, 0.39214352, 0.2695424, 0.92799146, 0.37685974, np.nan],
                                                 [np.nan, 0.68397339, 0.21505729, 0.58290104, 0.57941079, np.nan]],
                                                [[np.nan, 0.58692423, 0.67538211, 0.49404422, 0.21969893, 0.44848684],
                                                 [np.nan, 0.25943448, 0.52853397, 0.37064543, 0.56508607, 0.54677672],
                                                 [np.nan, 0.57938967, 0.24983683, 0.56396387, 0.40954981,
                                                  0.54766378]]]))
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle,
                                      np.array([[[np.nan, 0.4331313135726192, 0.6840577028722109,
                                                  0.6512587535769027, 0.46103783752657196,
                                                  0.4458780299268002],
                                                 [np.nan, 0.3472622657241471, 0.4967412743122601,
                                                  0.3511254390743444, 0.6158668723817633,
                                                  0.6530306371341883]],
                                                [[np.nan, 0.28961434759647886, 0.4605315640327775,
                                                  0.1696400491684556, 0.5536033722169851, np.nan],
                                                 [np.nan, 0.36893257326036033, 0.5503067315578603,
                                                  0.4180400469436511, 0.34500665266917885, np.nan]],
                                                [[np.nan, 0.3235344172044824, 0.2723923266567329,
                                                  0.47371839168027874, 0.24877332851572392, 0.26530669335734053],
                                                 [np.nan, 0.7168536710108676, 0.7201942929609031,
                                                  0.8626536919242743, 0.195765390947093, 0.831855394214509]]]))

        expected_h_rebinned_pitch_angles_and_gyrophase = np.transpose(h_intensity, (0, 1, 3, 2))
        expected_he4_rebinned_pitch_angles_and_gyrophase = np.transpose(he3he4_intensity, (0, 1, 3, 2))
        expected_o_rebinned_pitch_angles_and_gyrophase = np.transpose(cno_intensity, (0, 1, 3, 2))
        expected_fe_rebinned_pitch_angles_and_gyrophase = np.transpose(fe_intensity, (0, 1, 3, 2))

        np.testing.assert_array_equal(codice_hi_data_product.h_intensity_by_pitch_angle_and_gyrophase,
                                      expected_h_rebinned_pitch_angles_and_gyrophase)

        np.testing.assert_array_equal(codice_hi_data_product.he4_intensity_by_pitch_angle_and_gyrophase,
                                      expected_he4_rebinned_pitch_angles_and_gyrophase)
        np.testing.assert_array_equal(codice_hi_data_product.o_intensity_by_pitch_angle_and_gyrophase,
                                      expected_o_rebinned_pitch_angles_and_gyrophase)
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle_and_gyrophase,
                                      expected_fe_rebinned_pitch_angles_and_gyrophase)

    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_unit_vector")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.get_sector_unit_vectors")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_pitch_angle")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_gyrophase")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.hit_rebin_by_pitch_angle_and_gyrophase")
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
            ssd_index=(np.array([270, 15])),
            spin_sector_index=(np.array([15, 45, 75, 105, 135, 165])),
            h_intensities=h_intensity,
            energy_h=(np.array([1.11, 1.17])),
            energy_h_delta=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            cno_intensities=cno_intensity,
            energy_cno=(np.array([1.11, 1.17, 1.25])),
            energy_cno_delta=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            fe_intensities=fe_intensity,
            energy_fe=(np.array([1.11, 1.17]) * 1.3),
            energy_fe_delta=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            he3he4_intensities=he3he4_intensity,
            energy_he3he4=(np.array([1.11, 1.17]) * 1.4),
            energy_he3he4_delta=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
        )

        dependencies = CodicePitchAngleDependencies(mag_l1d_data=mag_l1d_data,
                                                    codice_sectored_intensities_data=codice_l2_data)

        expected_pitch_angles = np.linspace(15, 165, 6)
        expected_gyrophase = np.linspace(15, 345, 12)

        expected_pitch_angle_delta = np.repeat(15, 6)
        expected_gyrophase_delta = np.repeat(15, 12)

        sector_unit_vectors = rng.random((2, 6, 3))
        mag_vectors = rng.random((2, 3))
        mock_calculate_unit_vector.side_effect = [mag_vectors, sector_unit_vectors]

        mock_calculate_gyrophases.side_effect = [
            sentinel.epoch1_gyrophase,
            sentinel.epoch2_gyrophase,
        ]

        mock_calculate_pitch_angles.side_effect = [
            sentinel.epoch1_pitch_angle,
            sentinel.epoch2_pitch_angle,
        ]

        mock_rebin_by_pitch_angle_and_gyrophase.side_effect = [
            (10, 0, 0, 1, 0, 0),
            (20, 0, 0, 2, 0, 0),
            (30, 0, 0, 3, 0, 0),
            (40, 0, 0, 4, 0, 0),
            (50, 0, 0, 5, 0, 0),
            (60, 0, 0, 6, 0, 0),
            (70, 0, 0, 7, 0, 0),
            (80, 0, 0, 8, 0, 0),
        ]

        codice_processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=sentinel.input_metadata)
        codice_hi_data_product = codice_processor.process_l3b(dependencies=dependencies)

        mock_get_sector_unit_vectors.assert_called_once_with(codice_l2_data.spin_sector_index,
                                                             codice_l2_data.ssd_index)
        mock_calculate_unit_vector.assert_has_calls(
            [NumpyArrayMatcher(rebinned_mag_data), mock_get_sector_unit_vectors.return_value])

        mock_calculate_pitch_angles.assert_has_calls([
            call(-1 * sector_unit_vectors, rebinned_mag_data[0]),
            call(-1 * sector_unit_vectors, rebinned_mag_data[1]),
        ])

        mock_calculate_gyrophases.assert_has_calls([
            call(-1 * sector_unit_vectors, rebinned_mag_data[0]),
            call(-1 * sector_unit_vectors, rebinned_mag_data[1]),
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
            expected_rebin_calls.extend([
                call(intensity_data=NumpyArrayMatcher(species_intensity),
                     intensity_delta_plus=NumpyArrayMatcher(species_uncertainty),
                     intensity_delta_minus=NumpyArrayMatcher(species_uncertainty),
                     pitch_angles=sentinel.epoch1_pitch_angle,
                     gyrophases=sentinel.epoch1_gyrophase,
                     number_of_pitch_angle_bins=6, number_of_gyrophase_bins=12
                     ),
                call(intensity_data=NumpyArrayMatcher(species_intensity),
                     intensity_delta_plus=NumpyArrayMatcher(species_uncertainty),
                     intensity_delta_minus=NumpyArrayMatcher(species_uncertainty),
                     pitch_angles=sentinel.epoch2_pitch_angle,
                     gyrophases=sentinel.epoch2_gyrophase,
                     number_of_pitch_angle_bins=6, number_of_gyrophase_bins=12
                     )
            ])

        mock_rebin_by_pitch_angle_and_gyrophase.assert_has_calls(expected_rebin_calls)

        self.assertEqual(sentinel.input_metadata, codice_hi_data_product.input_metadata)
        np.testing.assert_array_equal(epoch, codice_hi_data_product.epoch)
        np.testing.assert_array_equal(epoch_delta, codice_hi_data_product.epoch_delta)

        np.testing.assert_array_equal(codice_hi_data_product.energy_h, codice_l2_data.energy_h)
        np.testing.assert_array_equal(codice_hi_data_product.energy_h_delta, codice_l2_data.energy_h_delta)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno, codice_l2_data.energy_cno)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno_delta, codice_l2_data.energy_cno_delta)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe, codice_l2_data.energy_fe)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe_delta, codice_l2_data.energy_fe_delta)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4, codice_l2_data.energy_he3he4)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4_delta, codice_l2_data.energy_he3he4_delta)

        np.testing.assert_array_almost_equal(expected_pitch_angles, codice_hi_data_product.pitch_angle)
        np.testing.assert_array_equal(expected_pitch_angle_delta, codice_hi_data_product.pitch_angle_delta)
        np.testing.assert_array_almost_equal(expected_gyrophase, codice_hi_data_product.gyrophase)
        np.testing.assert_array_equal(expected_gyrophase_delta, codice_hi_data_product.gyrophase_delta)

        np.testing.assert_allclose(codice_hi_data_product.h_intensity_by_pitch_angle, np.array([1, 2]))

        np.testing.assert_array_equal(codice_hi_data_product.he4_intensity_by_pitch_angle, np.array([3, 4]))
        np.testing.assert_array_equal(codice_hi_data_product.o_intensity_by_pitch_angle, np.array([5, 6]))
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle, np.array([7, 8]))

        np.testing.assert_array_equal(codice_hi_data_product.h_intensity_by_pitch_angle_and_gyrophase,
                                      np.array([10, 20]))
        np.testing.assert_array_equal(codice_hi_data_product.he4_intensity_by_pitch_angle_and_gyrophase,
                                      np.array([30, 40]))
        np.testing.assert_array_equal(codice_hi_data_product.o_intensity_by_pitch_angle_and_gyrophase,
                                      np.array([50, 60]))
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle_and_gyrophase,
                                      np.array([70, 80]))

    def _create_priority_events(self):
        num_epochs = 2
        event_buffer_size = 102

        number_of_event_data_points = CODICE_HI_NUM_L2_PRIORITIES * num_epochs * event_buffer_size

        numbers = (np.arange(1, number_of_event_data_points + 1).reshape(CODICE_HI_NUM_L2_PRIORITIES, num_epochs,
                                                                         event_buffer_size))
        tof_all_events = (numbers % 24) + 1
        ssd_id_all_events = numbers * 1000

        (reshaped_l2_data_quality,
         reshaped_l2_number_of_events) = [np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES), np.nan) for _ in range(2)]

        (reshaped_l2_multi_flag,
         reshaped_l2_ssd_energy,
         reshaped_l2_ssd_energy_plus,
         reshaped_l2_ssd_energy_minus,
         reshaped_l2_ssd_id,
         reshaped_l2_spin_angle,
         reshaped_l2_spin_number,
         reshaped_l2_time_of_flight,
         reshaped_l2_type) = [np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES, event_buffer_size), np.nan) for _ in
                              range(9)]

        events = []
        for i in range(CODICE_HI_NUM_L2_PRIORITIES):
            data_quality = np.arange(num_epochs) * 1
            number_of_events = np.arange(num_epochs) + i

            ssd_energy = ssd_id_all_events[i]
            ssd_energy_plus = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i
            ssd_energy_minus = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i

            ssd_id = ssd_id_all_events[i]
            time_of_flight = tof_all_events[i]

            multi_flag = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i
            spin_angle = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i
            spin_number = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i
            type = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i

            events.append(PriorityEventL2(data_quality, multi_flag, number_of_events, ssd_energy, ssd_energy_plus,
                                          ssd_energy_minus, ssd_id,
                                          spin_angle, spin_number, time_of_flight, type))

            reshaped_l2_data_quality[:, i] = data_quality
            reshaped_l2_number_of_events[:, i] = number_of_events

            reshaped_l2_multi_flag[:, i, :] = multi_flag
            reshaped_l2_ssd_energy[:, i, :] = ssd_energy
            reshaped_l2_ssd_id[:, i, :] = ssd_id
            reshaped_l2_spin_angle[:, i, :] = spin_angle
            reshaped_l2_spin_number[:, i, :] = spin_number
            reshaped_l2_time_of_flight[:, i, :] = time_of_flight
            reshaped_l2_type[:, i, :] = type
            reshaped_l2_ssd_energy_plus[:, i, :] = ssd_energy_plus
            reshaped_l2_ssd_energy_minus[:, i, :] = ssd_energy_minus

        return events, (
            reshaped_l2_data_quality, reshaped_l2_multi_flag, reshaped_l2_number_of_events,
            reshaped_l2_ssd_energy, reshaped_l2_ssd_energy_plus, reshaped_l2_ssd_energy_minus, reshaped_l2_ssd_id,
            reshaped_l2_spin_angle, reshaped_l2_spin_number,
            reshaped_l2_time_of_flight, reshaped_l2_type)

    def _assert_estimated_mass(self, l2_priority_event, actual_calculated_mass, actual_energy_per_nuc,
                               expected_energy_per_nuc):
        np.testing.assert_array_equal(l2_priority_event.ssd_energy / expected_energy_per_nuc, actual_calculated_mass)
        np.testing.assert_array_equal(expected_energy_per_nuc, actual_energy_per_nuc)
