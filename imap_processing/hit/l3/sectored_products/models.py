from spacepy import pycdf

from imap_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
GYROPHASE_CDF_VAR_NAME = "gyrophase"
GYROPHASE_DELTA_CDF_VAR_NAME = "gyrophase_delta"
PITCH_ANGLE_CDF_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_CDF_VAR_NAME = "pitch_angle_delta"
H_FLUX_CDF_VAR_NAME = "h_flux"
H_ENERGY_CDF_VAR_NAME = "h_energy"
H_ENERGY_DELTA_CDF_VAR_NAME = "h_energy_delta"
HE4_FLUX_CDF_VAR_NAME = "he4_flux"
HE4_ENERGY_CDF_VAR_NAME = "he4_energy"
HE4_ENERGY_DELTA_CDF_VAR_NAME = "he4_energy_delta"
CNO_FLUX_CDF_VAR_NAME = "cno_flux"
CNO_ENERGY_CDF_VAR_NAME = "cno_energy"
CNO_ENERGY_DELTA_CDF_VAR_NAME = "cno_energy_delta"
NE_MG_SI_FLUX_CDF_VAR_NAME = "ne_mg_si_flux"
NE_MG_SI_ENERGY_CDF_VAR_NAME = "ne_mg_si_energy"
NE_MG_SI_ENERGY_DELTA_CDF_VAR_NAME = "ne_mg_si_energy_delta"
IRON_FLUX_CDF_VAR_NAME = "iron_flux"
IRON_ENERGY_CDF_VAR_NAME = "iron_energy"
IRON_ENERGY_DELTA_CDF_VAR_NAME = "iron_energy_delta"


class HitPitchAngleDataProduct(DataProduct):
    def __init__(self, epochs,
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
                 iron_energy_deltas):
        self.epochs = epochs
        self.epoch_deltas = epoch_deltas
        self.pitch_angles = pitch_angles
        self.pitch_angle_deltas = pitch_angle_deltas
        self.gyrophases = gyrophases
        self.gyrophase_deltas = gyrophase_deltas
        self.h_fluxes = h_fluxes
        self.h_energies = h_energies
        self.h_energy_deltas = h_energy_deltas
        self.he4_fluxes = he4_fluxes
        self.he4_energies = he4_energies
        self.he4_energy_deltas = he4_energy_deltas
        self.cno_fluxes = cno_fluxes
        self.cno_energies = cno_energies
        self.cno_energy_deltas = cno_energy_deltas
        self.ne_mg_si_fluxes = ne_mg_si_fluxes
        self.ne_mg_si_energies = ne_mg_si_energies
        self.ne_mg_si_energy_deltas = ne_mg_si_energy_deltas
        self.iron_fluxes = iron_fluxes
        self.iron_energies = iron_energies
        self.iron_energy_deltas = iron_energy_deltas

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epochs, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_deltas),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, self.pitch_angles, record_varying=False),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, self.pitch_angle_deltas, record_varying=False),
            DataProductVariable(GYROPHASE_CDF_VAR_NAME, self.gyrophases, record_varying=False),
            DataProductVariable(GYROPHASE_DELTA_CDF_VAR_NAME, self.gyrophase_deltas, record_varying=False),
            DataProductVariable(H_FLUX_CDF_VAR_NAME, self.h_fluxes),
            DataProductVariable(H_ENERGY_CDF_VAR_NAME, self.h_energies, record_varying=False),
            DataProductVariable(H_ENERGY_DELTA_CDF_VAR_NAME, self.h_energy_deltas, record_varying=False),
            DataProductVariable(HE4_FLUX_CDF_VAR_NAME, self.he4_fluxes),
            DataProductVariable(HE4_ENERGY_CDF_VAR_NAME, self.he4_energies, record_varying=False),
            DataProductVariable(HE4_ENERGY_DELTA_CDF_VAR_NAME, self.he4_energy_deltas, record_varying=False),
            DataProductVariable(CNO_FLUX_CDF_VAR_NAME, self.cno_fluxes),
            DataProductVariable(CNO_ENERGY_CDF_VAR_NAME, self.cno_energies, record_varying=False),
            DataProductVariable(CNO_ENERGY_DELTA_CDF_VAR_NAME, self.cno_energy_deltas, record_varying=False),
            DataProductVariable(NE_MG_SI_FLUX_CDF_VAR_NAME, self.ne_mg_si_fluxes),
            DataProductVariable(NE_MG_SI_ENERGY_CDF_VAR_NAME, self.ne_mg_si_energies, record_varying=False),
            DataProductVariable(NE_MG_SI_ENERGY_DELTA_CDF_VAR_NAME, self.ne_mg_si_energy_deltas, record_varying=False),
            DataProductVariable(IRON_FLUX_CDF_VAR_NAME, self.iron_fluxes),
            DataProductVariable(IRON_ENERGY_CDF_VAR_NAME, self.iron_energies, record_varying=False),
            DataProductVariable(IRON_ENERGY_DELTA_CDF_VAR_NAME, self.iron_energy_deltas, record_varying=False),
        ]
