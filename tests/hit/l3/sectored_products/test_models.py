from datetime import timedelta
from unittest import TestCase
from unittest.mock import sentinel, Mock

import numpy as np

from imap_l3_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
from imap_l3_processing.models import DataProductVariable


class TestHitPitchAngleDataProduct(TestCase):
    def test_to_data_product_variables(self):
        epoch_deltas = np.array([timedelta(seconds=5)])

        pitch_angles = np.array([15, 30, 45, 60, 75, 90])
        gyrophases = np.array([1, 2, 3, 4])
        h_energy = np.array([5, 6, 7, 8])
        he_energy = np.array([9, 10, 11, 12])
        cno_energy = np.array([13, 14, 15, 16])
        nemgsi_energy = np.array([17, 18, 19, 20])
        fe_energy = np.array([21, 22, 23, 24])

        mock_measurement_pitch_angle = Mock()
        mock_measurement_pitch_angle.shape = (1, 8, 15)

        data = HitPitchAngleDataProduct(sentinel.input_meta_data,
                                        sentinel.epoch,
                                        epoch_deltas,
                                        pitch_angles,
                                        sentinel.pitch_angles_deltas,
                                        gyrophases,
                                        sentinel.gyrophase_deltas,
                                        sentinel.h_intensity,
                                        sentinel.h_intensity_delta_plus,
                                        sentinel.h_intensity_delta_minus,
                                        sentinel.h_pa_intensity,
                                        sentinel.h_pa_intensity_delta_plus,
                                        sentinel.h_pa_intensity_delta_minus,
                                        h_energy,
                                        sentinel.h_energy_delta_plus,
                                        sentinel.h_energy_delta_minus,
                                        sentinel.he4_intensity,
                                        sentinel.he4_intensity_delta_plus,
                                        sentinel.he4_intensity_delta_minus,
                                        sentinel.he4_pa_intensity,
                                        sentinel.he4_pa_intensity_delta_plus,
                                        sentinel.he4_pa_intensity_delta_minus,
                                        he_energy,
                                        sentinel.he4_energy_delta_plus,
                                        sentinel.he4_energy_delta_minus,
                                        sentinel.cno_intensity,
                                        sentinel.cno_intensity_delta_plus,
                                        sentinel.cno_intensity_delta_minus,
                                        sentinel.cno_pa_intensity,
                                        sentinel.cno_pa_intensity_delta_plus,
                                        sentinel.cno_pa_intensity_delta_minus,
                                        cno_energy,
                                        sentinel.cno_energy_delta_plus,
                                        sentinel.cno_energy_delta_minus,
                                        sentinel.ne_mg_si_intensity,
                                        sentinel.ne_mg_si_intensity_delta_plus,
                                        sentinel.ne_mg_si_intensity_delta_minus,
                                        sentinel.ne_mg_si_pa_intensity,
                                        sentinel.ne_mg_si_pa_intensity_delta_plus,
                                        sentinel.ne_mg_si_pa_intensity_delta_minus,
                                        nemgsi_energy,
                                        sentinel.ne_mg_si_energy_delta_plus,
                                        sentinel.ne_mg_si_energy_delta_minus,
                                        sentinel.iron_intensity,
                                        sentinel.iron_intensity_delta_plus,
                                        sentinel.iron_intensity_delta_minus,
                                        sentinel.iron_pa_intensity,
                                        sentinel.iron_pa_intensity_delta_plus,
                                        sentinel.iron_pa_intensity_delta_minus,
                                        fe_energy,
                                        sentinel.iron_energy_delta_plus,
                                        sentinel.iron_energy_delta_minus,
                                        mock_measurement_pitch_angle,
                                        sentinel.measurement_gyrophase
                                        )

        data_product_variables = data.to_data_product_variables()
        expected_epoch_deltas = np.array([5e9])
        expected_data_product_variables = [
            DataProductVariable("epoch", sentinel.epoch),
            DataProductVariable("epoch_delta", expected_epoch_deltas),
            DataProductVariable("pitch_angle", pitch_angles),
            DataProductVariable("pitch_angle_delta", sentinel.pitch_angles_deltas),
            DataProductVariable("gyrophase", gyrophases),
            DataProductVariable("gyrophase_delta", sentinel.gyrophase_deltas),
            DataProductVariable("h_intensity", sentinel.h_intensity),
            DataProductVariable("h_intensity_delta_plus", sentinel.h_intensity_delta_plus),
            DataProductVariable("h_intensity_delta_minus", sentinel.h_intensity_delta_minus),
            DataProductVariable("h_intensity_pa", sentinel.h_pa_intensity),
            DataProductVariable("h_intensity_pa_delta_plus", sentinel.h_pa_intensity_delta_plus),
            DataProductVariable("h_intensity_pa_delta_minus", sentinel.h_pa_intensity_delta_minus),
            DataProductVariable("h_energy", h_energy),
            DataProductVariable("h_energy_delta_plus", sentinel.h_energy_delta_plus),
            DataProductVariable("h_energy_delta_minus", sentinel.h_energy_delta_minus),
            DataProductVariable("he4_intensity", sentinel.he4_intensity),
            DataProductVariable("he4_intensity_delta_plus", sentinel.he4_intensity_delta_plus),
            DataProductVariable("he4_intensity_delta_minus", sentinel.he4_intensity_delta_minus),
            DataProductVariable("he4_intensity_pa", sentinel.he4_pa_intensity),
            DataProductVariable("he4_intensity_pa_delta_plus", sentinel.he4_pa_intensity_delta_plus),
            DataProductVariable("he4_intensity_pa_delta_minus", sentinel.he4_pa_intensity_delta_minus),
            DataProductVariable("he4_energy", he_energy),
            DataProductVariable("he4_energy_delta_plus", sentinel.he4_energy_delta_plus),
            DataProductVariable("he4_energy_delta_minus", sentinel.he4_energy_delta_minus),
            DataProductVariable("cno_intensity", sentinel.cno_intensity),
            DataProductVariable("cno_intensity_delta_plus", sentinel.cno_intensity_delta_plus),
            DataProductVariable("cno_intensity_delta_minus", sentinel.cno_intensity_delta_minus),
            DataProductVariable("cno_intensity_pa", sentinel.cno_pa_intensity),
            DataProductVariable("cno_intensity_pa_delta_plus", sentinel.cno_pa_intensity_delta_plus),
            DataProductVariable("cno_intensity_pa_delta_minus", sentinel.cno_pa_intensity_delta_minus),
            DataProductVariable("cno_energy", cno_energy),
            DataProductVariable("cno_energy_delta_plus", sentinel.cno_energy_delta_plus),
            DataProductVariable("cno_energy_delta_minus", sentinel.cno_energy_delta_minus),
            DataProductVariable("nemgsi_intensity", sentinel.ne_mg_si_intensity),
            DataProductVariable("nemgsi_intensity_delta_plus", sentinel.ne_mg_si_intensity_delta_plus),
            DataProductVariable("nemgsi_intensity_delta_minus", sentinel.ne_mg_si_intensity_delta_minus),
            DataProductVariable("nemgsi_intensity_pa", sentinel.ne_mg_si_pa_intensity),
            DataProductVariable("nemgsi_intensity_pa_delta_plus", sentinel.ne_mg_si_pa_intensity_delta_plus),
            DataProductVariable("nemgsi_intensity_pa_delta_minus", sentinel.ne_mg_si_pa_intensity_delta_minus),
            DataProductVariable("nemgsi_energy", nemgsi_energy),
            DataProductVariable("nemgsi_energy_delta_plus", sentinel.ne_mg_si_energy_delta_plus),
            DataProductVariable("nemgsi_energy_delta_minus", sentinel.ne_mg_si_energy_delta_minus),
            DataProductVariable("fe_intensity", sentinel.iron_intensity),
            DataProductVariable("fe_intensity_delta_plus", sentinel.iron_intensity_delta_plus),
            DataProductVariable("fe_intensity_delta_minus", sentinel.iron_intensity_delta_minus),
            DataProductVariable("fe_intensity_pa", sentinel.iron_pa_intensity),
            DataProductVariable("fe_intensity_pa_delta_plus", sentinel.iron_pa_intensity_delta_plus),
            DataProductVariable("fe_intensity_pa_delta_minus", sentinel.iron_pa_intensity_delta_minus),
            DataProductVariable("fe_energy", fe_energy),
            DataProductVariable("fe_energy_delta_plus", sentinel.iron_energy_delta_plus),
            DataProductVariable("fe_energy_delta_minus", sentinel.iron_energy_delta_minus),
            DataProductVariable("measurement_pitch_angle", mock_measurement_pitch_angle),
            DataProductVariable("measurement_gyrophase", sentinel.measurement_gyrophase),
            DataProductVariable("pitch_angle_label",
                                ["Pitch Angle Label 1", "Pitch Angle Label 2", "Pitch Angle Label 3",
                                 "Pitch Angle Label 4", "Pitch Angle Label 5", "Pitch Angle Label 6"]),
            DataProductVariable("gyrophase_label",
                                ["Gyrophase Label 1", "Gyrophase Label 2", "Gyrophase Label 3", "Gyrophase Label 4"]),
            DataProductVariable("h_energy_label",
                                ["H Energy Label 1", "H Energy Label 2", "H Energy Label 3", "H Energy Label 4"]),
            DataProductVariable("he4_energy_label", ["He4 Energy Label 1", "He4 Energy Label 2", "He4 Energy Label 3",
                                                     "He4 Energy Label 4"]),
            DataProductVariable("cno_energy_label", ["CNO Energy Label 1", "CNO Energy Label 2", "CNO Energy Label 3",
                                                     "CNO Energy Label 4"]),
            DataProductVariable("nemgsi_energy_label",
                                ["NeMgSi Energy Label 1", "NeMgSi Energy Label 2", "NeMgSi Energy Label 3",
                                 "NeMgSi Energy Label 4"]),
            DataProductVariable("fe_energy_label",
                                ["Fe Energy Label 1", "Fe Energy Label 2", "Fe Energy Label 3", "Fe Energy Label 4"]),
            DataProductVariable("declination_idx", [0, 1, 2, 3, 4, 5, 6, 7]),
            DataProductVariable("azimuth_idx", [0, 1, 2, 3, 4, 5, 6, 7,
                                                8, 9, 10, 11, 12, 13, 14]),
            DataProductVariable("declination_idx_label",
                                ["Declination Index Label 0", "Declination Index Label 1", "Declination Index Label 2",
                                 "Declination Index Label 3", "Declination Index Label 4", "Declination Index Label 5",
                                 "Declination Index Label 6", "Declination Index Label 7"]),
            DataProductVariable("azimuth_idx_label",
                                ['Azimuth Index Label 0', 'Azimuth Index Label 1', 'Azimuth Index Label 2',
                                 'Azimuth Index Label 3', 'Azimuth Index Label 4', 'Azimuth Index Label 5',
                                 'Azimuth Index Label 6', 'Azimuth Index Label 7', 'Azimuth Index Label 8',
                                 'Azimuth Index Label 9', 'Azimuth Index Label 10', 'Azimuth Index Label 11',
                                 'Azimuth Index Label 12', 'Azimuth Index Label 13', 'Azimuth Index Label 14'])
        ]

        self.assertEqual(expected_data_product_variables, data_product_variables)
