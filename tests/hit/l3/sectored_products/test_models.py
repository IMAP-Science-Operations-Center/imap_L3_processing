from unittest import TestCase
from unittest.mock import sentinel

from spacepy import pycdf

from imap_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct, EPOCH_CDF_VAR_NAME, \
    EPOCH_DELTA_CDF_VAR_NAME, PITCH_ANGLE_CDF_VAR_NAME, GYROPHASE_CDF_VAR_NAME, H_FLUX_CDF_VAR_NAME, \
    H_ENERGY_CDF_VAR_NAME, H_ENERGY_DELTA_CDF_VAR_NAME, HE4_FLUX_CDF_VAR_NAME, HE4_ENERGY_CDF_VAR_NAME, \
    HE4_ENERGY_DELTA_CDF_VAR_NAME, CNO_FLUX_CDF_VAR_NAME, CNO_ENERGY_CDF_VAR_NAME, CNO_ENERGY_DELTA_CDF_VAR_NAME, \
    NE_MG_SI_FLUX_CDF_VAR_NAME, NE_MG_SI_ENERGY_CDF_VAR_NAME, NE_MG_SI_ENERGY_DELTA_CDF_VAR_NAME, \
    IRON_FLUX_CDF_VAR_NAME, IRON_ENERGY_CDF_VAR_NAME, IRON_ENERGY_DELTA_CDF_VAR_NAME, PITCH_ANGLE_DELTA_CDF_VAR_NAME, \
    GYROPHASE_DELTA_CDF_VAR_NAME
from imap_processing.models import DataProductVariable


class TestHitPitchAngleDataProduct(TestCase):
    def test_to_data_product_variables(self):
        h_fluxes = sentinel.h_fluxes
        h_energies = sentinel.h_energies
        h_energy_deltas = sentinel.h_energy_deltas

        he4_fluxes = sentinel.he4_fluxes
        he4_energies = sentinel.he4_energies
        he4_energy_deltas = sentinel.he4_energy_deltas

        cno_fluxes = sentinel.cno_fluxes
        cno_energies = sentinel.cno_energies
        cno_energy_deltas = sentinel.cno_energy_deltas

        ne_mg_si_fluxes = sentinel.ne_mg_si_fluxes
        ne_mg_si_energies = sentinel.ne_mg_si_energies
        ne_mg_si_energy_deltas = sentinel.ne_mg_si_energy_deltas

        iron_fluxes = sentinel.iron_fluxes
        iron_energies = sentinel.iron_energies
        iron_energy_deltas = sentinel.iron_energy_deltas

        pitch_angles = sentinel.pitch_angles
        pitch_angle_deltas = sentinel.pitch_angles_deltas
        gyrophases = sentinel.gyrophases
        gyrophase_deltas = sentinel.gyrophase_deltas
        epochs = sentinel.epochs
        epoch_deltas = sentinel.epoch_delta

        data = HitPitchAngleDataProduct(epochs,
                                        epoch_deltas,
                                        pitch_angles,
                                        pitch_angle_deltas,
                                        gyrophases,
                                        gyrophase_deltas,
                                        h_fluxes,
                                        h_energies,
                                        h_energy_deltas,
                                        he4_fluxes,
                                        he4_energies,
                                        he4_energy_deltas,
                                        cno_fluxes,
                                        cno_energies,
                                        cno_energy_deltas,
                                        ne_mg_si_fluxes,
                                        ne_mg_si_energies,
                                        ne_mg_si_energy_deltas,
                                        iron_fluxes,
                                        iron_energies,
                                        iron_energy_deltas
                                        )

        data_product_variables = data.to_data_product_variables()

        expected_data_product_variables = [
            DataProductVariable(EPOCH_CDF_VAR_NAME, epochs, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, epoch_deltas),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, pitch_angles, record_varying=False),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, pitch_angle_deltas, record_varying=False),
            DataProductVariable(GYROPHASE_CDF_VAR_NAME, gyrophases, record_varying=False),
            DataProductVariable(GYROPHASE_DELTA_CDF_VAR_NAME, gyrophase_deltas, record_varying=False),
            DataProductVariable(H_FLUX_CDF_VAR_NAME, h_fluxes),
            DataProductVariable(H_ENERGY_CDF_VAR_NAME, h_energies, record_varying=False),
            DataProductVariable(H_ENERGY_DELTA_CDF_VAR_NAME, h_energy_deltas, record_varying=False),
            DataProductVariable(HE4_FLUX_CDF_VAR_NAME, he4_fluxes),
            DataProductVariable(HE4_ENERGY_CDF_VAR_NAME, he4_energies, record_varying=False),
            DataProductVariable(HE4_ENERGY_DELTA_CDF_VAR_NAME, he4_energy_deltas, record_varying=False),
            DataProductVariable(CNO_FLUX_CDF_VAR_NAME, cno_fluxes),
            DataProductVariable(CNO_ENERGY_CDF_VAR_NAME, cno_energies, record_varying=False),
            DataProductVariable(CNO_ENERGY_DELTA_CDF_VAR_NAME, cno_energy_deltas, record_varying=False),
            DataProductVariable(NE_MG_SI_FLUX_CDF_VAR_NAME, ne_mg_si_fluxes),
            DataProductVariable(NE_MG_SI_ENERGY_CDF_VAR_NAME, ne_mg_si_energies, record_varying=False),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_CDF_VAR_NAME, ne_mg_si_energy_deltas, record_varying=False),
            DataProductVariable(IRON_FLUX_CDF_VAR_NAME, iron_fluxes),
            DataProductVariable(IRON_ENERGY_CDF_VAR_NAME, iron_energies, record_varying=False),
            DataProductVariable(IRON_ENERGY_DELTA_CDF_VAR_NAME, iron_energy_deltas, record_varying=False),
        ]

        self.assertEqual(expected_data_product_variables, data_product_variables)
