from dataclasses import dataclass

import numpy as np

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
MEASUREMENT_PITCH_ANGLE_VAR_NAME = "measurement_pitch_angle"
MEASUREMENT_GYROPHASE_VAR_NAME = "measurement_gyrophase"
PITCH_ANGLE_LABEL_VAR_NAME = "pitch_angle_label"
GYROPHASE_LABEL_VAR_NAME = "gyrophase_label"
H_ENERGY_LABEL_VAR_NAME = "h_energy_label"
HE4_ENERGY_LABEL_VAR_NAME = "he4_energy_label"
CNO_ENERGY_LABEL_VAR_NAME = "cno_energy_label"
NE_MG_SI_ENERGY_LABEL_VAR_NAME = "nemgsi_energy_label"
IRON_ENERGY_LABEL_VAR_NAME = "fe_energy_label"
AZIMUTH_VAR_NAME = "azimuth"
ZENITH_VAR_NAME = "zenith"
AZIMUTH_LABEL_VAR_NAME = "azimuth_label"
ZENITH_LABEL_VAR_NAME = "zenith_label"


@dataclass
class HitPitchAngleDataProduct(DataProduct):
    epochs: np.ndarray
    epoch_deltas: np.ndarray
    pitch_angles: np.ndarray
    pitch_angle_deltas: np.ndarray
    gyrophases: np.ndarray
    gyrophase_deltas: np.ndarray
    h_intensity: np.ndarray
    h_intensity_delta_plus: np.ndarray
    h_intensity_delta_minus: np.ndarray
    h_intensity_pa: np.ndarray
    h_intensity_pa_delta_plus: np.ndarray
    h_intensity_pa_delta_minus: np.ndarray
    h_energies: np.ndarray
    h_energy_delta_plus: np.ndarray
    h_energy_delta_minus: np.ndarray
    he4_intensity: np.ndarray
    he4_intensity_delta_plus: np.ndarray
    he4_intensity_delta_minus: np.ndarray
    he4_intensity_pa: np.ndarray
    he4_intensity_pa_delta_plus: np.ndarray
    he4_intensity_pa_delta_minus: np.ndarray
    he4_energies: np.ndarray
    he4_energy_delta_plus: np.ndarray
    he4_energy_delta_minus: np.ndarray
    cno_intensity: np.ndarray
    cno_intensity_delta_plus: np.ndarray
    cno_intensity_delta_minus: np.ndarray
    cno_intensity_pa: np.ndarray
    cno_intensity_pa_delta_plus: np.ndarray
    cno_intensity_pa_delta_minus: np.ndarray
    cno_energies: np.ndarray
    cno_energy_delta_plus: np.ndarray
    cno_energy_delta_minus: np.ndarray
    ne_mg_si_intensity: np.ndarray
    ne_mg_si_intensity_delta_plus: np.ndarray
    ne_mg_si_intensity_delta_minus: np.ndarray
    ne_mg_si_intensity_pa: np.ndarray
    ne_mg_si_intensity_pa_delta_plus: np.ndarray
    ne_mg_si_intensity_pa_delta_minus: np.ndarray
    ne_mg_si_energies: np.ndarray
    ne_mg_si_energy_delta_plus: np.ndarray
    ne_mg_si_energy_delta_minus: np.ndarray
    iron_intensity: np.ndarray
    iron_intensity_delta_plus: np.ndarray
    iron_intensity_delta_minus: np.ndarray
    iron_intensity_pa: np.ndarray
    iron_intensity_pa_delta_plus: np.ndarray
    iron_intensity_pa_delta_minus: np.ndarray
    iron_energies: np.ndarray
    iron_energy_delta_plus: np.ndarray
    iron_energy_delta_minus: np.ndarray
    measurement_pitch_angle: np.ndarray
    measurement_gyrophase: np.ndarray
    azimuth: np.ndarray
    zenith: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epochs),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME,
                                np.array([t.total_seconds() for t in self.epoch_deltas]) * 1e9),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, self.pitch_angles),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, self.pitch_angle_deltas),
            DataProductVariable(GYROPHASE_CDF_VAR_NAME, self.gyrophases),
            DataProductVariable(GYROPHASE_DELTA_CDF_VAR_NAME, self.gyrophase_deltas),
            DataProductVariable(H_INTENSITY_CDF_VAR_NAME, self.h_intensity),
            DataProductVariable(H_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.h_intensity_delta_plus),
            DataProductVariable(H_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.h_intensity_delta_minus),
            DataProductVariable(H_INTENSITY_PA_CDF_VAR_NAME, self.h_intensity_pa),
            DataProductVariable(H_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.h_intensity_pa_delta_plus),
            DataProductVariable(H_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.h_intensity_pa_delta_minus),
            DataProductVariable(H_ENERGY_CDF_VAR_NAME, self.h_energies),
            DataProductVariable(H_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.h_energy_delta_plus),
            DataProductVariable(H_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.h_energy_delta_minus),
            DataProductVariable(HE4_INTENSITY_CDF_VAR_NAME, self.he4_intensity),
            DataProductVariable(HE4_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.he4_intensity_delta_plus),
            DataProductVariable(HE4_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.he4_intensity_delta_minus),
            DataProductVariable(HE4_INTENSITY_PA_CDF_VAR_NAME, self.he4_intensity_pa),
            DataProductVariable(HE4_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.he4_intensity_pa_delta_plus),
            DataProductVariable(HE4_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.he4_intensity_pa_delta_minus),
            DataProductVariable(HE4_ENERGY_CDF_VAR_NAME, self.he4_energies),
            DataProductVariable(HE4_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.he4_energy_delta_plus),
            DataProductVariable(HE4_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.he4_energy_delta_minus),
            DataProductVariable(CNO_INTENSITY_CDF_VAR_NAME, self.cno_intensity),
            DataProductVariable(CNO_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.cno_intensity_delta_plus),
            DataProductVariable(CNO_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.cno_intensity_delta_minus),
            DataProductVariable(CNO_INTENSITY_PA_CDF_VAR_NAME, self.cno_intensity_pa),
            DataProductVariable(CNO_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.cno_intensity_pa_delta_plus),
            DataProductVariable(CNO_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.cno_intensity_pa_delta_minus),
            DataProductVariable(CNO_ENERGY_CDF_VAR_NAME, self.cno_energies),
            DataProductVariable(CNO_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.cno_energy_delta_plus),
            DataProductVariable(CNO_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.cno_energy_delta_minus),
            DataProductVariable(NE_MG_SI_INTENSITY_CDF_VAR_NAME, self.ne_mg_si_intensity),
            DataProductVariable(NE_MG_SI_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.ne_mg_si_intensity_delta_plus),
            DataProductVariable(NE_MG_SI_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.ne_mg_si_intensity_delta_minus),
            DataProductVariable(NE_MG_SI_INTENSITY_PA_CDF_VAR_NAME, self.ne_mg_si_intensity_pa),
            DataProductVariable(NE_MG_SI_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.ne_mg_si_intensity_pa_delta_plus),
            DataProductVariable(NE_MG_SI_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.ne_mg_si_intensity_pa_delta_minus),
            DataProductVariable(NE_MG_SI_ENERGY_CDF_VAR_NAME, self.ne_mg_si_energies),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.ne_mg_si_energy_delta_plus),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.ne_mg_si_energy_delta_minus),
            DataProductVariable(IRON_INTENSITY_CDF_VAR_NAME, self.iron_intensity),
            DataProductVariable(IRON_INTENSITY_DELTA_PLUS_CDF_VAR_NAME, self.iron_intensity_delta_plus),
            DataProductVariable(IRON_INTENSITY_DELTA_MINUS_CDF_VAR_NAME, self.iron_intensity_delta_minus),
            DataProductVariable(IRON_INTENSITY_PA_CDF_VAR_NAME, self.iron_intensity_pa),
            DataProductVariable(IRON_INTENSITY_PA_DELTA_PLUS_CDF_VAR_NAME, self.iron_intensity_pa_delta_plus),
            DataProductVariable(IRON_INTENSITY_PA_DELTA_MINUS_CDF_VAR_NAME, self.iron_intensity_pa_delta_minus),
            DataProductVariable(IRON_ENERGY_CDF_VAR_NAME, self.iron_energies),
            DataProductVariable(IRON_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.iron_energy_delta_plus),
            DataProductVariable(IRON_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.iron_energy_delta_minus),
            DataProductVariable(MEASUREMENT_PITCH_ANGLE_VAR_NAME, self.measurement_pitch_angle),
            DataProductVariable(MEASUREMENT_GYROPHASE_VAR_NAME, self.measurement_gyrophase),
            DataProductVariable(PITCH_ANGLE_LABEL_VAR_NAME,
                                [f"Pitch Angle Label {str(i + 1)}" for i in range(len(self.pitch_angles))]),
            DataProductVariable(GYROPHASE_LABEL_VAR_NAME,
                                [f"Gyrophase Label {str(i + 1)}" for i in range(len(self.gyrophases))]),
            DataProductVariable(H_ENERGY_LABEL_VAR_NAME,
                                [f"H Energy Label {str(i + 1)}" for i in range(len(self.h_energies))]),
            DataProductVariable(HE4_ENERGY_LABEL_VAR_NAME,
                                [f"He4 Energy Label {str(i + 1)}" for i in range(len(self.he4_energies))]),
            DataProductVariable(CNO_ENERGY_LABEL_VAR_NAME,
                                [f"CNO Energy Label {str(i + 1)}" for i in range(len(self.cno_energies))]),
            DataProductVariable(NE_MG_SI_ENERGY_LABEL_VAR_NAME,
                                [f"NeMgSi Energy Label {str(i + 1)}" for i in range(len(self.ne_mg_si_energies))]),
            DataProductVariable(IRON_ENERGY_LABEL_VAR_NAME,
                                [f"Fe Energy Label {str(i + 1)}" for i in range(len(self.iron_energies))]),
            DataProductVariable(AZIMUTH_VAR_NAME, self.azimuth),
            DataProductVariable(ZENITH_VAR_NAME, self.zenith),
            DataProductVariable(AZIMUTH_LABEL_VAR_NAME, [str(float(azimuth)) for azimuth in self.azimuth]),
            DataProductVariable(ZENITH_LABEL_VAR_NAME, [str(float(zenith)) for zenith in self.zenith])
        ]
