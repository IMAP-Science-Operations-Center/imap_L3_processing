from dataclasses import dataclass

import numpy as np
from spacepy import pycdf

from imap_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
GYROPHASE_CDF_VAR_NAME = "gyrophase"
GYROPHASE_DELTA_CDF_VAR_NAME = "gyrophase_delta"
PITCH_ANGLE_CDF_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_CDF_VAR_NAME = "pitch_angle_delta"
H_FLUX_CDF_VAR_NAME = "h_flux"
H_PA_FLUX_CDF_VAR_NAME = "h_flux_pa"
H_ENERGY_CDF_VAR_NAME = "h_energy"
H_ENERGY_DELTA_CDF_VAR_NAME = "h_energy_delta"
HE4_FLUX_CDF_VAR_NAME = "he4_flux"
HE4_PA_FLUX_CDF_VAR_NAME = "he4_flux_pa"
HE4_ENERGY_CDF_VAR_NAME = "he4_energy"
HE4_ENERGY_DELTA_CDF_VAR_NAME = "he4_energy_delta"
CNO_FLUX_CDF_VAR_NAME = "cno_flux"
CNO_PA_FLUX_CDF_VAR_NAME = "cno_flux_pa"
CNO_ENERGY_CDF_VAR_NAME = "cno_energy"
CNO_ENERGY_DELTA_CDF_VAR_NAME = "cno_energy_delta"
NE_MG_SI_FLUX_CDF_VAR_NAME = "nemgsi_flux"
NE_MG_SI_PA_FLUX_CDF_VAR_NAME = "nemgsi_flux_pa"
NE_MG_SI_ENERGY_CDF_VAR_NAME = "nemgsi_energy"
NE_MG_SI_ENERGY_DELTA_CDF_VAR_NAME = "nemgsi_energy_delta"
IRON_FLUX_CDF_VAR_NAME = "fe_flux"
IRON_PA_FLUX_CDF_VAR_NAME = "fe_flux_pa"
IRON_ENERGY_CDF_VAR_NAME = "fe_energy"
IRON_ENERGY_DELTA_CDF_VAR_NAME = "fe_energy_delta"

@dataclass
class HitPitchAngleDataProduct(DataProduct):
    epochs: np.ndarray
    epoch_deltas: np.ndarray
    pitch_angles: np.ndarray
    pitch_angle_deltas: np.ndarray
    gyrophases: np.ndarray
    gyrophase_deltas: np.ndarray
    h_fluxes: np.ndarray
    h_pa_fluxes: np.ndarray
    h_energies: np.ndarray
    h_energy_deltas: np.ndarray
    he4_fluxes: np.ndarray
    he4_pa_fluxes: np.ndarray
    he4_energies: np.ndarray
    he4_energy_deltas: np.ndarray
    cno_fluxes: np.ndarray
    cno_pa_fluxes: np.ndarray
    cno_energies: np.ndarray
    cno_energy_deltas: np.ndarray
    ne_mg_si_fluxes: np.ndarray
    ne_mg_si_pa_fluxes: np.ndarray
    ne_mg_si_energies: np.ndarray
    ne_mg_si_energy_deltas: np.ndarray
    iron_fluxes: np.ndarray
    iron_pa_fluxes: np.ndarray
    iron_energies: np.ndarray
    iron_energy_deltas: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epochs, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.array([t.total_seconds() for t in self.epoch_deltas]) * 1e9, cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, self.pitch_angles, record_varying=False),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, self.pitch_angle_deltas, record_varying=False),
            DataProductVariable(GYROPHASE_CDF_VAR_NAME, self.gyrophases, record_varying=False),
            DataProductVariable(GYROPHASE_DELTA_CDF_VAR_NAME, self.gyrophase_deltas, record_varying=False),
            DataProductVariable(H_FLUX_CDF_VAR_NAME, self.h_fluxes),
            DataProductVariable(H_PA_FLUX_CDF_VAR_NAME, self.h_pa_fluxes),
            DataProductVariable(H_ENERGY_CDF_VAR_NAME, self.h_energies, record_varying=False),
            DataProductVariable(H_ENERGY_DELTA_CDF_VAR_NAME, self.h_energy_deltas, record_varying=False),
            DataProductVariable(HE4_FLUX_CDF_VAR_NAME, self.he4_fluxes),
            DataProductVariable(HE4_PA_FLUX_CDF_VAR_NAME, self.he4_pa_fluxes),
            DataProductVariable(HE4_ENERGY_CDF_VAR_NAME, self.he4_energies, record_varying=False),
            DataProductVariable(HE4_ENERGY_DELTA_CDF_VAR_NAME, self.he4_energy_deltas, record_varying=False),
            DataProductVariable(CNO_FLUX_CDF_VAR_NAME, self.cno_fluxes),
            DataProductVariable(CNO_PA_FLUX_CDF_VAR_NAME, self.cno_pa_fluxes),
            DataProductVariable(CNO_ENERGY_CDF_VAR_NAME, self.cno_energies, record_varying=False),
            DataProductVariable(CNO_ENERGY_DELTA_CDF_VAR_NAME, self.cno_energy_deltas, record_varying=False),
            DataProductVariable(NE_MG_SI_FLUX_CDF_VAR_NAME, self.ne_mg_si_fluxes),
            DataProductVariable(NE_MG_SI_PA_FLUX_CDF_VAR_NAME, self.ne_mg_si_pa_fluxes),
            DataProductVariable(NE_MG_SI_ENERGY_CDF_VAR_NAME, self.ne_mg_si_energies, record_varying=False),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_CDF_VAR_NAME, self.ne_mg_si_energy_deltas, record_varying=False),
            DataProductVariable(IRON_FLUX_CDF_VAR_NAME, self.iron_fluxes),
            DataProductVariable(IRON_PA_FLUX_CDF_VAR_NAME, self.iron_pa_fluxes),
            DataProductVariable(IRON_ENERGY_CDF_VAR_NAME, self.iron_energies, record_varying=False),
            DataProductVariable(IRON_ENERGY_DELTA_CDF_VAR_NAME, self.iron_energy_deltas, record_varying=False),
        ]
