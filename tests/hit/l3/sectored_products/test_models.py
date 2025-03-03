from datetime import timedelta
from unittest import TestCase
from unittest.mock import sentinel

import numpy as np
from spacepy import pycdf

from imap_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
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
        epoch_deltas = np.array([timedelta(seconds=5)])

        input_meta_data = sentinel.input_meta_data

        data = HitPitchAngleDataProduct(input_meta_data,
                                        epochs,
                                        epoch_deltas,
                                        pitch_angles,
                                        pitch_angle_deltas,
                                        gyrophases,
                                        gyrophase_deltas,
                                        h_fluxes,
                                        sentinel.h_pa_fluxes,
                                        h_energies,
                                        h_energy_deltas,
                                        he4_fluxes,
                                        sentinel.he4_pa_fluxes,
                                        he4_energies,
                                        he4_energy_deltas,
                                        cno_fluxes,
                                        sentinel.cno_pa_fluxes,
                                        cno_energies,
                                        cno_energy_deltas,
                                        ne_mg_si_fluxes,
                                        sentinel.ne_mg_si_pa_fluxes,
                                        ne_mg_si_energies,
                                        ne_mg_si_energy_deltas,
                                        iron_fluxes,
                                        sentinel.iron_pa_fluxes,
                                        iron_energies,
                                        iron_energy_deltas
                                        )

        data_product_variables = data.to_data_product_variables()
        expected_epoch_deltas = np.array([5e9])
        expected_data_product_variables = [
            DataProductVariable("epoch", epochs, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable("epoch_delta", expected_epoch_deltas, cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable("pitch_angle", pitch_angles, record_varying=False),
            DataProductVariable("pitch_angle_delta", pitch_angle_deltas, record_varying=False),
            DataProductVariable("gyrophase", gyrophases, record_varying=False),
            DataProductVariable("gyrophase_delta", gyrophase_deltas, record_varying=False),
            DataProductVariable("h_flux", h_fluxes),
            DataProductVariable("h_flux_pa", sentinel.h_pa_fluxes),
            DataProductVariable("h_energy", h_energies, record_varying=False),
            DataProductVariable("h_energy_delta", h_energy_deltas, record_varying=False),
            DataProductVariable("he4_flux", he4_fluxes),
            DataProductVariable("he4_flux_pa", sentinel.he4_pa_fluxes),
            DataProductVariable("he4_energy", he4_energies, record_varying=False),
            DataProductVariable("he4_energy_delta", he4_energy_deltas, record_varying=False),
            DataProductVariable("cno_flux", cno_fluxes),
            DataProductVariable("cno_flux_pa", sentinel.cno_pa_fluxes),
            DataProductVariable("cno_energy", cno_energies, record_varying=False),
            DataProductVariable("cno_energy_delta", cno_energy_deltas, record_varying=False),
            DataProductVariable("nemgsi_flux", ne_mg_si_fluxes),
            DataProductVariable("nemgsi_flux_pa", sentinel.ne_mg_si_pa_fluxes),
            DataProductVariable("nemgsi_energy", ne_mg_si_energies, record_varying=False),
            DataProductVariable("nemgsi_energy_delta", ne_mg_si_energy_deltas, record_varying=False),
            DataProductVariable("fe_flux", iron_fluxes),
            DataProductVariable("fe_flux_pa", sentinel.iron_pa_fluxes),
            DataProductVariable("fe_energy", iron_energies, record_varying=False),
            DataProductVariable("fe_energy_delta", iron_energy_deltas, record_varying=False),
        ]

        self.assertEqual(expected_data_product_variables, data_product_variables)
