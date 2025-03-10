from dataclasses import dataclass

import numpy as np
from spacepy import pycdf

from imap_l3_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
GYROPHASE_CDF_VAR_NAME = "gyrophase"
GYROPHASE_DELTA_CDF_VAR_NAME = "gyrophase_delta"
PITCH_ANGLE_CDF_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_CDF_VAR_NAME = "pitch_angle_delta"
H_INTENSITY_CDF_VAR_NAME = "h_intensity"
H_INTENSITY_DELTA_PLUS_CDF_VAR_NAME = "h_intensity_delta_plus"
H_INTENSITY_DELTA_MINUS_CDF_VAR_NAME = "h_intensity_delta_minus"
H_INTENSITY_PA_CDF_VAR_NAME = "h_intensity_pa"
H_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME = "h_intensity_pa_delta_plus"
H_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME = "h_intensity_pa_delta_minus"
H_ENERGY_CDF_VAR_NAME = "h_energy"
H_ENERGY_DELTA_PLUS_CDF_VAR_NAME = "h_energy_delta_plus"
H_ENERGY_DELTA_MINUS_CDF_VAR_NAME = "h_energy_delta_minus"
HE4_INTENSITY_CDF_VAR_NAME = "he4_intensity"
HE4_INTENSITY_DELTA_PLUS_CDF_VAR_NAME = "he4_intensity_delta_plus"
HE4_INTENSITY_DELTA_MINUS_CDF_VAR_NAME = "he4_intensity_delta_minus"
HE4_INTENSITY_PA_CDF_VAR_NAME = "he4_intensity_pa"
HE4_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME = "he4_intensity_pa_delta_plus"
HE4_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME = "he4_intensity_pa_delta_minus"
HE4_ENERGY_CDF_VAR_NAME = "he4_energy"
HE4_ENERGY_DELTA_PLUS_CDF_VAR_NAME = "he4_energy_delta_plus"
HE4_ENERGY_DELTA_MINUS_CDF_VAR_NAME = "he4_energy_delta_minus"
CNO_INTENSITY_CDF_VAR_NAME = "cno_intensity"
CNO_INTENSITY_DELTA_PLUS_CDF_VAR_NAME = "cno_intensity_delta_plus"
CNO_INTENSITY_DELTA_MINUS_CDF_VAR_NAME = "cno_intensity_delta_minus"
CNO_INTENSITY_PA_CDF_VAR_NAME = "cno_intensity_pa"
CNO_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME = "cno_intensity_pa_delta_plus"
CNO_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME = "cno_intensity_pa_delta_minus"
CNO_ENERGY_CDF_VAR_NAME = "cno_energy"
CNO_ENERGY_DELTA_PLUS_CDF_VAR_NAME = "cno_energy_delta_plus"
CNO_ENERGY_DELTA_MINUS_CDF_VAR_NAME = "cno_energy_delta_minus"
NE_MG_SI_INTENSITY_CDF_VAR_NAME = "nemgsi_intensity"
NE_MG_SI_INTENSITY_DELTA_PLUS_CDF_VAR_NAME = "nemgsi_intensity_delta_plus"
NE_MG_SI_INTENSITY_DELTA_MINUS_CDF_VAR_NAME = "nemgsi_intensity_delta_minus"
NE_MG_SI_INTENSITY_PA_CDF_VAR_NAME = "nemgsi_intensity_pa"
NE_MG_SI_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME = "nemgsi_intensity_pa_delta_plus"
NE_MG_SI_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME = "nemgsi_intensity_pa_delta_minus"
NE_MG_SI_ENERGY_CDF_VAR_NAME = "nemgsi_energy"
NE_MG_SI_ENERGY_DELTA_PLUS_CDF_VAR_NAME = "nemgsi_energy_delta_plus"
NE_MG_SI_ENERGY_DELTA_MINUS_CDF_VAR_NAME = "nemgsi_energy_delta_minus"
IRON_INTENSITY_CDF_VAR_NAME = "fe_intensity"
IRON_INTENSITY_DELTA_PLUS_CDF_VAR_NAME = "fe_intensity_delta_plus"
IRON_INTENSITY_DELTA_MINUS_CDF_VAR_NAME = "fe_intensity_delta_minus"
IRON_INTENSITY_PA_CDF_VAR_NAME = "fe_intensity_pa"
IRON_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME = "fe_intensity_pa_delta_plus"
IRON_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME = "fe_intensity_pa_delta_minus"
IRON_ENERGY_CDF_VAR_NAME = "fe_energy"
IRON_ENERGY_DELTA_PLUS_CDF_VAR_NAME = "fe_energy_delta_plus"
IRON_ENERGY_DELTA_MINUS_CDF_VAR_NAME = "fe_energy_delta_minus"


@dataclass
class HitPitchAngleDataProduct(DataProduct):
    epochs: np.ndarray
    epoch_deltas: np.ndarray
    pitch_angles: np.ndarray
    pitch_angle_deltas: np.ndarray
    gyrophases: np.ndarray
    gyrophase_deltas: np.ndarray
    h_fluxes: np.ndarray
    h_flux_delta_plus: np.ndarray
    h_flux_delta_minus: np.ndarray
    h_flux_pa: np.ndarray
    h_flux_pa_delta_plus: np.ndarray
    h_flux_pa_delta_minus: np.ndarray
    h_energies: np.ndarray
    h_energy_delta_plus: np.ndarray
    h_energy_delta_minus: np.ndarray
    he4_fluxes: np.ndarray
    he4_flux_delta_plus: np.ndarray
    he4_flux_delta_minus: np.ndarray
    he4_flux_pa: np.ndarray
    he4_flux_pa_delta_plus: np.ndarray
    he4_flux_pa_delta_minus: np.ndarray
    he4_energies: np.ndarray
    he4_energy_delta_plus: np.ndarray
    he4_energy_delta_minus: np.ndarray
    cno_fluxes: np.ndarray
    cno_flux_delta_plus: np.ndarray
    cno_flux_delta_minus: np.ndarray
    cno_flux_pa: np.ndarray
    cno_flux_pa_delta_plus: np.ndarray
    cno_flux_pa_delta_minus: np.ndarray
    cno_energies: np.ndarray
    cno_energy_delta_plus: np.ndarray
    cno_energy_delta_minus: np.ndarray
    ne_mg_si_fluxes: np.ndarray
    ne_mg_si_flux_delta_plus: np.ndarray
    ne_mg_si_flux_delta_minus: np.ndarray
    ne_mg_si_flux_pa: np.ndarray
    ne_mg_si_flux_pa_delta_plus: np.ndarray
    ne_mg_si_flux_pa_delta_minus: np.ndarray
    ne_mg_si_energies: np.ndarray
    ne_mg_si_energy_delta_plus: np.ndarray
    ne_mg_si_energy_delta_minus: np.ndarray
    iron_fluxes: np.ndarray
    iron_flux_delta_plus: np.ndarray
    iron_flux_delta_minus: np.ndarray
    iron_flux_pa: np.ndarray
    iron_flux_pa_delta_plus: np.ndarray
    iron_flux_pa_delta_minus: np.ndarray
    iron_energies: np.ndarray
    iron_energy_delta_plus: np.ndarray
    iron_energy_delta_minus: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epochs, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME,
                                np.array([t.total_seconds() for t in self.epoch_deltas]) * 1e9,
                                cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, self.pitch_angles, record_varying=False),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, self.pitch_angle_deltas, record_varying=False),
            DataProductVariable(GYROPHASE_CDF_VAR_NAME, self.gyrophases, record_varying=False),
            DataProductVariable(GYROPHASE_DELTA_CDF_VAR_NAME, self.gyrophase_deltas, record_varying=False),
            DataProductVariable(H_INTENSITY_CDF_VAR_NAME, self.h_fluxes),
            DataProductVariable(H_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.h_flux_delta_plus),
            DataProductVariable(H_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.h_flux_delta_minus),
            DataProductVariable(H_INTENSITY_PA_CDF_VAR_NAME, self.h_flux_pa),
            DataProductVariable(H_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.h_flux_pa_delta_plus),
            DataProductVariable(H_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.h_flux_pa_delta_minus),
            DataProductVariable(H_ENERGY_CDF_VAR_NAME, self.h_energies, record_varying=False),
            DataProductVariable(H_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.h_energy_delta_plus, record_varying=False),
            DataProductVariable(H_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.h_energy_delta_minus, record_varying=False),
            DataProductVariable(HE4_INTENSITY_CDF_VAR_NAME, self.he4_fluxes),
            DataProductVariable(HE4_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.he4_flux_delta_plus),
            DataProductVariable(HE4_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.he4_flux_delta_minus),
            DataProductVariable(HE4_INTENSITY_PA_CDF_VAR_NAME, self.he4_flux_pa),
            DataProductVariable(HE4_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.he4_flux_pa_delta_plus),
            DataProductVariable(HE4_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.he4_flux_pa_delta_minus),
            DataProductVariable(HE4_ENERGY_CDF_VAR_NAME, self.he4_energies, record_varying=False),
            DataProductVariable(HE4_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.he4_energy_delta_plus, record_varying=False),
            DataProductVariable(HE4_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.he4_energy_delta_minus, record_varying=False),
            DataProductVariable(CNO_INTENSITY_CDF_VAR_NAME, self.cno_fluxes),
            DataProductVariable(CNO_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.cno_flux_delta_plus),
            DataProductVariable(CNO_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.cno_flux_delta_minus),
            DataProductVariable(CNO_INTENSITY_PA_CDF_VAR_NAME, self.cno_flux_pa),
            DataProductVariable(CNO_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.cno_flux_pa_delta_plus),
            DataProductVariable(CNO_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.cno_flux_pa_delta_minus),
            DataProductVariable(CNO_ENERGY_CDF_VAR_NAME, self.cno_energies, record_varying=False),
            DataProductVariable(CNO_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.cno_energy_delta_plus, record_varying=False),
            DataProductVariable(CNO_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.cno_energy_delta_minus, record_varying=False),
            DataProductVariable(NE_MG_SI_INTENSITY_CDF_VAR_NAME, self.ne_mg_si_fluxes),
            DataProductVariable(NE_MG_SI_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.ne_mg_si_flux_delta_plus),
            DataProductVariable(NE_MG_SI_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.ne_mg_si_flux_delta_minus),
            DataProductVariable(NE_MG_SI_INTENSITY_PA_CDF_VAR_NAME, self.ne_mg_si_flux_pa),
            DataProductVariable(NE_MG_SI_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.ne_mg_si_flux_pa_delta_plus),
            DataProductVariable(NE_MG_SI_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.ne_mg_si_flux_pa_delta_minus),
            DataProductVariable(NE_MG_SI_ENERGY_CDF_VAR_NAME, self.ne_mg_si_energies, record_varying=False),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.ne_mg_si_energy_delta_plus,
                                record_varying=False),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.ne_mg_si_energy_delta_minus,
                                record_varying=False),
            DataProductVariable(IRON_INTENSITY_CDF_VAR_NAME, self.iron_fluxes),
            DataProductVariable(IRON_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.iron_flux_delta_plus),
            DataProductVariable(IRON_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.iron_flux_delta_minus),
            DataProductVariable(IRON_INTENSITY_PA_CDF_VAR_NAME, self.iron_flux_pa),
            DataProductVariable(IRON_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.iron_flux_pa_delta_plus),
            DataProductVariable(IRON_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.iron_flux_pa_delta_minus),
            DataProductVariable(IRON_ENERGY_CDF_VAR_NAME, self.iron_energies, record_varying=False),
            DataProductVariable(IRON_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.iron_energy_delta_plus, record_varying=False),
            DataProductVariable(IRON_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.iron_energy_delta_minus,
                                record_varying=False),
        ]
