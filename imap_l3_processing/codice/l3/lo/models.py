from collections import namedtuple, Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection
from imap_l3_processing.models import DataProductVariable, DataProduct

CODICE_LO_L2_NUM_PRIORITIES = 7

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_PLUS_VAR_NAME = "epoch_delta_plus"
EPOCH_DELTA_MINUS_VAR_NAME = "epoch_delta_minus"
H_PARTIAL_DENSITY_VAR_NAME = "hplus_partial_density"
HE_PARTIAL_DENSITY_VAR_NAME = "heplusplus_partial_density"
C4_PARTIAL_DENSITY_VAR_NAME = "cplus4_partial_density"
C5_PARTIAL_DENSITY_VAR_NAME = "cplus5_partial_density"
C6_PARTIAL_DENSITY_VAR_NAME = "cplus6_partial_density"
O5_PARTIAL_DENSITY_VAR_NAME = "oplus5_partial_density"
O6_PARTIAL_DENSITY_VAR_NAME = "oplus6_partial_density"
O7_PARTIAL_DENSITY_VAR_NAME = "oplus7_partial_density"
O8_PARTIAL_DENSITY_VAR_NAME = "oplus8_partial_density"
MG_PARTIAL_DENSITY_VAR_NAME = "mg_partial_density"
SI_PARTIAL_DENSITY_VAR_NAME = "si_partial_density"
FE_LOW_PARTIAL_DENSITY_VAR_NAME = "fe_loq_partial_density"
FE_HIGH_PARTIAL_DENSITY_VAR_NAME = "fe_hiq_partial_density"


@dataclass
class CodiceLoL2SWSpeciesData:
    epoch: ndarray
    epoch_delta_minus: ndarray
    epoch_delta_plus: ndarray
    energy_table: ndarray
    hplus: ndarray
    heplusplus: ndarray
    heplus: ndarray
    ne: ndarray
    cplus4: ndarray
    cplus5: ndarray
    cplus6: ndarray
    oplus5: ndarray
    oplus6: ndarray
    oplus7: ndarray
    oplus8: ndarray
    cnoplus: ndarray
    mg: ndarray
    si: ndarray
    fe_loq: ndarray
    fe_hiq: ndarray
    data_quality: ndarray
    spin_sector_index: ndarray

    @classmethod
    def read_from_cdf(cls, l2_sectored_intensities_cdf: Path):
        with CDF(str(l2_sectored_intensities_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                epoch_delta_minus=cdf["epoch_delta_minus"][...],
                epoch_delta_plus=cdf["epoch_delta_plus"][...],
                energy_table=cdf["energy_table"][...],
                hplus=cdf["hplus"][...],
                heplusplus=cdf["heplusplus"][...],
                heplus=cdf["heplus"][...],
                ne=cdf["ne"][...],
                cplus4=cdf["cplus4"][...],
                cplus5=cdf["cplus5"][...],
                cplus6=cdf["cplus6"][...],
                oplus5=cdf["oplus5"][...],
                oplus6=cdf["oplus6"][...],
                oplus7=cdf["oplus7"][...],
                oplus8=cdf["oplus8"][...],
                cnoplus=cdf["cnoplus"][...],
                mg=cdf["mg"][...],
                si=cdf["si"][...],
                fe_loq=cdf["fe_loq"][...],
                fe_hiq=cdf["fe_hiq"][...],
                data_quality=cdf["data_quality"][...],
                spin_sector_index=cdf["spin_sector_index"][...],
            )


EnergyAndSpinAngle = namedtuple(typename="EnergyAndSpinAngle", field_names=["energy", "spin_angle"])


@dataclass
class PriorityEvent:
    apd_energy: ndarray
    apd_gain: ndarray
    apd_id: ndarray
    data_quality: ndarray
    energy_step: ndarray
    multi_flag: ndarray
    num_events: ndarray
    pha_type: ndarray
    spin_angle: ndarray
    tof: ndarray

    def total_events_binned_by_energy_step_and_spin_angle(self) -> [dict[EnergyAndSpinAngle, int]]:
        all_epoch_to_energy_and_spin_angle = np.stack((self.energy_step, self.spin_angle), axis=-1)
        total_events_by_energy_and_spin_angle = []

        for epoch_to_energy_and_spin_angle in all_epoch_to_energy_and_spin_angle:
            energy_step_and_spin_angle = [EnergyAndSpinAngle(*energy_step_and_spin_angle)
                                          for energy_step_and_spin_angle in epoch_to_energy_and_spin_angle
                                          if not any(np.isnan(energy_step_and_spin_angle))]

            total_events_by_energy_and_spin_angle.append(dict(Counter(energy_step_and_spin_angle)))

        return total_events_by_energy_and_spin_angle


@dataclass
class CodiceLoL2DirectEventData:
    epoch: ndarray
    event_num: ndarray
    priority_events: list[PriorityEvent]

    @classmethod
    def _read_priority_event(cls, cdf):
        priority_events = []
        for index in range(CODICE_LO_L2_NUM_PRIORITIES):
            priority_events.append(PriorityEvent(cdf[f"P{index}_APDEnergy"][...],
                                                 cdf[f"P{index}_APDGain"][...],
                                                 cdf[f"P{index}_APD_ID"][...],
                                                 cdf[f"P{index}_DataQuality"][...],
                                                 cdf[f"P{index}_EnergyStep"][...],
                                                 cdf[f"P{index}_MultiFlag"][...],
                                                 cdf[f"P{index}_NumEvents"][...],
                                                 cdf[f"P{index}_PHAType"][...],
                                                 cdf[f"P{index}_SpinAngle"][...],
                                                 cdf[f"P{index}_TOF"][...]
                                                 ))
        return priority_events

    @classmethod
    def read_from_cdf(cls, l2_direct_event_cdf: Path):
        with CDF(str(l2_direct_event_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                event_num=cdf["event_num"][...],
                priority_events=cls._read_priority_event(cdf)
            )


@dataclass
class CodiceLoL1aSWPriorityRates:
    epoch: np.ndarray
    epoch_delta_plus: np.ndarray
    epoch_delta_minus: np.ndarray
    energy_table: np.ndarray
    acquisition_time_per_step: np.ndarray
    spin_sector_index: np.ndarray
    rgfo_half_spin: np.ndarray
    nso_half_spin: np.ndarray
    sw_bias_gain_mode: np.ndarray
    st_bias_gain_mode: np.ndarray
    data_quality: np.ndarray
    spin_period: np.ndarray
    p0_tcrs: np.ndarray
    p1_hplus: np.ndarray
    p2_heplusplus: np.ndarray
    p3_heavies: np.ndarray
    p4_dcrs: np.ndarray

    @classmethod
    def read_from_cdf(cls, cdf_path: Path):
        with CDF(str(cdf_path)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                epoch_delta_plus=cdf["epoch_delta_plus"][...],
                epoch_delta_minus=cdf["epoch_delta_minus"][...],
                energy_table=cdf["energy_table"][...],
                acquisition_time_per_step=cdf["acquisition_time_per_step"][...],
                spin_sector_index=cdf["spin_sector_index"][...],
                rgfo_half_spin=cdf["rgfo_half_spin"][...],
                nso_half_spin=cdf["nso_half_spin"][...],
                sw_bias_gain_mode=cdf["sw_bias_gain_mode"][...],
                st_bias_gain_mode=cdf["st_bias_gain_mode"][...],
                data_quality=cdf["data_quality"][...],
                spin_period=cdf["spin_period"][...],
                p0_tcrs=cdf["p0_tcrs"][...],
                p1_hplus=cdf["p1_hplus"][...],
                p2_heplusplus=cdf["p2_heplusplus"][...],
                p3_heavies=cdf["p3_heavies"][...],
                p4_dcrs=cdf["p4_dcrs"][...]
            )


@dataclass
class CodiceLoL1aNSWPriorityRates:
    energy_table: np.ndarray
    acquisition_time_per_step: np.ndarray
    epoch: np.ndarray
    epoch_delta_plus: np.ndarray
    epoch_delta_minus: np.ndarray
    spin_sector_index: np.ndarray
    rgfo_half_spin: np.ndarray
    data_quality: np.ndarray
    p5_heavies: np.ndarray
    p6_hplus_heplusplus: np.ndarray
    nso_half_spin: np.ndarray
    sw_bias_gain_mode: np.ndarray
    st_bias_gain_mode: np.ndarray
    spin_period: np.ndarray

    @classmethod
    def read_from_cdf(cls, cdf_path: Path):
        with CDF(str(cdf_path)) as cdf:
            return cls(
                energy_table=cdf["energy_table"][...],
                acquisition_time_per_step=cdf["acquisition_time_per_step"][...],
                epoch=cdf["epoch"][...],
                epoch_delta_plus=cdf["epoch_delta_plus"][...],
                epoch_delta_minus=cdf["epoch_delta_minus"][...],
                spin_sector_index=cdf["spin_sector_index"][...],
                rgfo_half_spin=cdf["rgfo_half_spin"][...],
                data_quality=cdf["data_quality"][...],
                p5_heavies=cdf["p5_heavies"][...],
                p6_hplus_heplusplus=cdf["p6_hplus_heplusplus"][...],
                nso_half_spin=cdf["nso_half_spin"][...],
                sw_bias_gain_mode=cdf["sw_bias_gain_mode"][...],
                st_bias_gain_mode=cdf["st_bias_gain_mode"][...],
                spin_period=cdf["spin_period"][...],
            )


@dataclass
class CodiceLoL3aPartialDensityDataProduct(DataProduct):
    epoch: ndarray
    epoch_delta_plus: ndarray
    epoch_delta_minus: ndarray
    hplus_partial_density: ndarray
    heplusplus_partial_density: ndarray
    cplus4_partial_density: ndarray
    cplus5_partial_density: ndarray
    cplus6_partial_density: ndarray
    oplus5_partial_density: ndarray
    oplus6_partial_density: ndarray
    oplus7_partial_density: ndarray
    oplus8_partial_density: ndarray
    mg_partial_density: ndarray
    si_partial_density: ndarray
    fe_loq_partial_density: ndarray
    fe_hiq_partial_density: ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_PLUS_VAR_NAME, self.epoch_delta_plus),
            DataProductVariable(EPOCH_DELTA_MINUS_VAR_NAME, self.epoch_delta_minus),
            DataProductVariable(H_PARTIAL_DENSITY_VAR_NAME, self.hplus_partial_density),
            DataProductVariable(HE_PARTIAL_DENSITY_VAR_NAME, self.heplusplus_partial_density),
            DataProductVariable(C4_PARTIAL_DENSITY_VAR_NAME, self.cplus4_partial_density),
            DataProductVariable(C5_PARTIAL_DENSITY_VAR_NAME, self.cplus5_partial_density),
            DataProductVariable(C6_PARTIAL_DENSITY_VAR_NAME, self.cplus6_partial_density),
            DataProductVariable(O5_PARTIAL_DENSITY_VAR_NAME, self.oplus5_partial_density),
            DataProductVariable(O6_PARTIAL_DENSITY_VAR_NAME, self.oplus6_partial_density),
            DataProductVariable(O7_PARTIAL_DENSITY_VAR_NAME, self.oplus7_partial_density),
            DataProductVariable(O8_PARTIAL_DENSITY_VAR_NAME, self.oplus8_partial_density),
            DataProductVariable(MG_PARTIAL_DENSITY_VAR_NAME, self.mg_partial_density),
            DataProductVariable(SI_PARTIAL_DENSITY_VAR_NAME, self.si_partial_density),
            DataProductVariable(FE_LOW_PARTIAL_DENSITY_VAR_NAME, self.fe_loq_partial_density),
            DataProductVariable(FE_HIGH_PARTIAL_DENSITY_VAR_NAME, self.fe_hiq_partial_density),
        ]


EVENT_NUM_VAR_NAME = "event_num"
SPIN_ANGLE_VAR_NAME = "spin_angle"
ENERGY_STEP_VAR_NAME = "energy_step"
PRIORITY_VAR_NAME = "priority"
NORMALIZATION_VAR_NAME = "normalization"
MASS_PER_CHARGE_VAR_NAME = "mass_per_charge"
MASS_VAR_NAME = "mass"
ENERGY_VAR_NAME = "energy"
GAIN_VAR_NAME = "gain"
APD_ID_VAR_NAME = "apd_id"
MULTI_FLAG_VAR_NAME = "multi_flag"
NUM_EVENTS_VAR_NAME = "num_events"
PHA_TYPE_VAR_NAME = "pha_type"
DATA_QUALITY_VAR_NAME = "data_quality"
TOF_VAR_NAME = "tof"


@dataclass
class CodiceLoL3aDirectEventDataProduct(DataProduct):
    epoch: ndarray
    event_num: np.ndarray
    normalization: ndarray
    mass_per_charge: np.ndarray
    mass: np.ndarray
    energy: np.ndarray
    gain: np.ndarray
    apd_id: np.ndarray
    multi_flag: np.ndarray
    num_events: np.ndarray
    pha_type: np.ndarray
    data_quality: np.ndarray
    tof: np.ndarray
    spin_angle: np.ndarray = field(init=False)
    energy_step: np.ndarray = field(init=False)
    priority: np.ndarray = field(init=False)

    def __post_init__(self):
        self.spin_angle = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        self.energy_step = np.arange(128)
        self.priority = np.arange(8)

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EVENT_NUM_VAR_NAME, self.event_num),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(ENERGY_STEP_VAR_NAME, self.energy_step),
            DataProductVariable(PRIORITY_VAR_NAME, self.priority),
            DataProductVariable(NORMALIZATION_VAR_NAME, self.normalization),
            DataProductVariable(MASS_PER_CHARGE_VAR_NAME, self.mass_per_charge),
            DataProductVariable(MASS_VAR_NAME, self.mass),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(GAIN_VAR_NAME, self.gain),
            DataProductVariable(APD_ID_VAR_NAME, self.apd_id),
            DataProductVariable(MULTI_FLAG_VAR_NAME, self.multi_flag),
            DataProductVariable(NUM_EVENTS_VAR_NAME, self.num_events),
            DataProductVariable(PHA_TYPE_VAR_NAME, self.pha_type),
            DataProductVariable(DATA_QUALITY_VAR_NAME, self.data_quality),
            DataProductVariable(TOF_VAR_NAME, self.tof),
        ]


@dataclass
class CodiceLo3dData:
    data_in_3d_bins: np.ndarray
    mass_bin_lookup: MassSpeciesBinLookup

    def get_3d_distribution(self, species: str, event_direction: EventDirection) -> np.ndarray:
        species_index = self.mass_bin_lookup.get_species_index(species, event_direction)
        return self.data_in_3d_bins[:, species_index, ...]
