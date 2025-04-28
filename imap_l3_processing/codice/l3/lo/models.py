from collections import namedtuple, Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.models import DataProductVariable, DataProduct

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
H_PARTIAL_DENSITY_VAR_NAME = "h_partial_density"
HE_PARTIAL_DENSITY_VAR_NAME = "he_partial_density"
C4_PARTIAL_DENSITY_VAR_NAME = "c4_partial_density"
C5_PARTIAL_DENSITY_VAR_NAME = "c5_partial_density"
C6_PARTIAL_DENSITY_VAR_NAME = "c6_partial_density"
O5_PARTIAL_DENSITY_VAR_NAME = "o5_partial_density"
O6_PARTIAL_DENSITY_VAR_NAME = "o6_partial_density"
O7_PARTIAL_DENSITY_VAR_NAME = "o7_partial_density"
O8_PARTIAL_DENSITY_VAR_NAME = "o8_partial_density"
MG_PARTIAL_DENSITY_VAR_NAME = "mg_partial_density"
SI_PARTIAL_DENSITY_VAR_NAME = "si_partial_density"
FE_LOW_PARTIAL_DENSITY_VAR_NAME = "fe_low_partial_density"
FE_HIGH_PARTIAL_DENSITY_VAR_NAME = "fe_high_partial_density"


@dataclass
class CodiceLoL2Data:
    epoch: ndarray
    epoch_delta_minus: ndarray
    epoch_delta_plus: ndarray
    energy_table: ndarray
    spin_sector: ndarray
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
                spin_sector=cdf["spin_sector"][...],
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

    def get_species_intensities(self) -> dict:
        return {
            "H+": self.hplus,
            "He++": self.heplusplus,
            "He+": self.heplus,
            "Ne": self.ne,
            "C+4": self.cplus4,
            "C+5": self.cplus5,
            "C+6": self.cplus6,
            "O+5": self.oplus5,
            "O+6": self.oplus6,
            "O+7": self.oplus7,
            "O+8": self.oplus8,
            "CNO+": self.cnoplus,
            "Mg": self.mg,
            "Si": self.si,
            "Fe (low Q)": self.fe_loq,
            "Fe (high Q)": self.fe_hiq,
        }


@dataclass
class CodiceLoL2bPriorityRates:
    epoch: ndarray
    energy: ndarray
    inst_az: ndarray
    spin_sector: ndarray
    energy_label: ndarray
    acquisition_times: ndarray
    counters: ndarray
    esa_sweep: ndarray
    hi_counters_aggregated_aggregated: ndarray
    hi_counters_singles_tcr: ndarray
    hi_counters_singles_ssdo: ndarray
    hi_counters_singles_stssd: ndarray
    hi_omni_h: ndarray
    hi_omni_he3: ndarray
    hi_omni_he4: ndarray
    hi_omni_c: ndarray
    hi_omni_o: ndarray
    hi_omni_ne_mg_si: ndarray
    hi_omni_fe: ndarray
    hi_omni_uh: ndarray
    hi_sectored_h: ndarray
    hi_sectored_he3he4: ndarray
    hi_sectored_cno: ndarray
    hi_sectored_fe: ndarray
    lo_counters_aggregated_aggregated: ndarray
    lo_counters_singles_apd_singles: ndarray
    lo_sw_angular_hplus: ndarray
    lo_sw_angular_heplusplus: ndarray
    lo_sw_angular_oplus6: ndarray
    lo_sw_angular_fe_loq: ndarray
    lo_nsw_angular_heplusplus: ndarray
    lo_sw_priority_p0_tcrs: ndarray
    lo_sw_priority_p1_hplus: ndarray
    lo_sw_priority_p2_heplusplus: ndarray
    lo_sw_priority_p3_heavies: ndarray
    lo_sw_priority_p4_dcrs: ndarray
    lo_nsw_priority_p5_heavies: ndarray
    lo_nsw_priority_p6_hplus_heplusplus: ndarray
    # TODO - there is no p7 variable in the l1a cdf. Update when we know what the name is
    lo_nsw_priority_p7_missing: ndarray
    lo_sw_species_hplus: ndarray
    lo_sw_species_heplusplus: ndarray
    lo_sw_species_cplus4: ndarray
    lo_sw_species_cplus5: ndarray
    lo_sw_species_cplus6: ndarray
    lo_sw_species_oplus5: ndarray
    lo_sw_species_oplus6: ndarray
    lo_sw_species_oplus7: ndarray
    lo_sw_species_oplus8: ndarray
    lo_sw_species_ne: ndarray
    lo_sw_species_mg: ndarray
    lo_sw_species_si: ndarray
    lo_sw_species_fe_loq: ndarray
    lo_sw_species_fe_hiq: ndarray
    lo_sw_species_heplus: ndarray
    lo_sw_species_cnoplus: ndarray
    lo_nsw_species_hplus: ndarray
    lo_nsw_species_heplusplus: ndarray
    lo_nsw_species_c: ndarray
    lo_nsw_species_o: ndarray
    lo_nsw_species_ne_si_mg: ndarray
    lo_nsw_species_fe: ndarray
    lo_nsw_species_heplus: ndarray
    lo_nsw_species_cnoplus: ndarray

    @classmethod
    def read_from_cdf(cls, codice_l1b_cdf: Path):
        with CDF(str(codice_l1b_cdf)) as cdf_file:
            return cls(
                epoch=cdf_file["epoch"][...],
                energy=cdf_file["energy"][...],
                inst_az=cdf_file["inst_az"][...],
                spin_sector=cdf_file["spin_sector"][...],
                energy_label=cdf_file["energy_label"][...],
                acquisition_times=cdf_file["acquisition_times"][...],
                counters=cdf_file["counters"][...],
                esa_sweep=cdf_file["esa_sweep"][...],
                hi_counters_aggregated_aggregated=cdf_file["hi_counters_aggregated_aggregated"][...],
                hi_counters_singles_tcr=cdf_file["hi_counters_singles_tcr"][...],
                hi_counters_singles_ssdo=cdf_file["hi_counters_singles_ssdo"][...],
                hi_counters_singles_stssd=cdf_file["hi_counters_singles_stssd"][...],
                hi_omni_h=cdf_file["hi_omni_h"][...],
                hi_omni_he3=cdf_file["hi_omni_he3"][...],
                hi_omni_he4=cdf_file["hi_omni_he4"][...],
                hi_omni_c=cdf_file["hi_omni_c"][...],
                hi_omni_o=cdf_file["hi_omni_o"][...],
                hi_omni_ne_mg_si=cdf_file["hi_omni_ne_mg_si"][...],
                hi_omni_fe=cdf_file["hi_omni_fe"][...],
                hi_omni_uh=cdf_file["hi_omni_uh"][...],
                hi_sectored_h=cdf_file["hi_sectored_h"][...],
                hi_sectored_he3he4=cdf_file["hi_sectored_he3he4"][...],
                hi_sectored_cno=cdf_file["hi_sectored_cno"][...],
                hi_sectored_fe=cdf_file["hi_sectored_fe"][...],
                lo_counters_aggregated_aggregated=cdf_file["lo_counters_aggregated_aggregated"][...],
                lo_counters_singles_apd_singles=cdf_file["lo_counters_singles_apd_singles"][...],
                lo_sw_angular_hplus=cdf_file["lo_sw_angular_hplus"][...],
                lo_sw_angular_heplusplus=cdf_file["lo_sw_angular_heplusplus"][...],
                lo_sw_angular_oplus6=cdf_file["lo_sw_angular_oplus6"][...],
                lo_sw_angular_fe_loq=cdf_file["lo_sw_angular_fe_loq"][...],
                lo_nsw_angular_heplusplus=cdf_file["lo_nsw_angular_heplusplus"][...],
                lo_sw_priority_p0_tcrs=cdf_file["lo_sw_priority_p0_tcrs"][...],
                lo_sw_priority_p1_hplus=cdf_file["lo_sw_priority_p1_hplus"][...],
                lo_sw_priority_p2_heplusplus=cdf_file["lo_sw_priority_p2_heplusplus"][...],
                lo_sw_priority_p3_heavies=cdf_file["lo_sw_priority_p3_heavies"][...],
                lo_sw_priority_p4_dcrs=cdf_file["lo_sw_priority_p4_dcrs"][...],
                lo_nsw_priority_p5_heavies=cdf_file["lo_nsw_priority_p5_heavies"][...],
                lo_nsw_priority_p6_hplus_heplusplus=cdf_file["lo_nsw_priority_p6_hplus_heplusplus"][...],
                lo_nsw_priority_p7_missing=cdf_file["lo_nsw_priority_p7_missing"][...],
                lo_sw_species_hplus=cdf_file["lo_sw_species_hplus"][...],
                lo_sw_species_heplusplus=cdf_file["lo_sw_species_heplusplus"][...],
                lo_sw_species_cplus4=cdf_file["lo_sw_species_cplus4"][...],
                lo_sw_species_cplus5=cdf_file["lo_sw_species_cplus5"][...],
                lo_sw_species_cplus6=cdf_file["lo_sw_species_cplus6"][...],
                lo_sw_species_oplus5=cdf_file["lo_sw_species_oplus5"][...],
                lo_sw_species_oplus6=cdf_file["lo_sw_species_oplus6"][...],
                lo_sw_species_oplus7=cdf_file["lo_sw_species_oplus7"][...],
                lo_sw_species_oplus8=cdf_file["lo_sw_species_oplus8"][...],
                lo_sw_species_ne=cdf_file["lo_sw_species_ne"][...],
                lo_sw_species_mg=cdf_file["lo_sw_species_mg"][...],
                lo_sw_species_si=cdf_file["lo_sw_species_si"][...],
                lo_sw_species_fe_loq=cdf_file["lo_sw_species_fe_loq"][...],
                lo_sw_species_fe_hiq=cdf_file["lo_sw_species_fe_hiq"][...],
                lo_sw_species_heplus=cdf_file["lo_sw_species_heplus"][...],
                lo_sw_species_cnoplus=cdf_file["lo_sw_species_cnoplus"][...],
                lo_nsw_species_hplus=cdf_file["lo_nsw_species_hplus"][...],
                lo_nsw_species_heplusplus=cdf_file["lo_nsw_species_heplusplus"][...],
                lo_nsw_species_c=cdf_file["lo_nsw_species_c"][...],
                lo_nsw_species_o=cdf_file["lo_nsw_species_o"][...],
                lo_nsw_species_ne_si_mg=cdf_file["lo_nsw_species_ne_si_mg"][...],
                lo_nsw_species_fe=cdf_file["lo_nsw_species_fe"][...],
                lo_nsw_species_heplus=cdf_file["lo_nsw_species_heplus"][...],
                lo_nsw_species_cnoplus=cdf_file["lo_nsw_species_cnoplus"][...]
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
    priority_event_0: PriorityEvent
    priority_event_1: PriorityEvent
    priority_event_2: PriorityEvent
    priority_event_3: PriorityEvent
    priority_event_4: PriorityEvent
    priority_event_5: PriorityEvent
    priority_event_6: PriorityEvent
    priority_event_7: PriorityEvent

    @classmethod
    def _read_priority_event(cls, cdf):
        values = {}
        for index in range(8):
            priority_event = PriorityEvent(cdf[f"P{index}_APDEnergy"][...],
                                           cdf[f"P{index}_APDGain"][...],
                                           cdf[f"P{index}_APD_ID"][...],
                                           cdf[f"P{index}_DataQuality"][...],
                                           cdf[f"P{index}_EnergyStep"][...],
                                           cdf[f"P{index}_MultiFlag"][...],
                                           cdf[f"P{index}_NumEvents"][...],
                                           cdf[f"P{index}_PHAType"][...],
                                           cdf[f"P{index}_SpinAngle"][...],
                                           cdf[f"P{index}_TOF"][...]
                                           )

            values.update({f"priority_event_{index}": priority_event})
        return values

    @classmethod
    def read_from_cdf(cls, l2_direct_event_cdf: Path):
        with CDF(str(l2_direct_event_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                event_num=cdf["event_num"][...],
                **cls._read_priority_event(cdf)

            )

    @property
    def priority_events(self):
        return [
            self.priority_event_0, self.priority_event_1, self.priority_event_2, self.priority_event_3,
            self.priority_event_4, self.priority_event_5, self.priority_event_6, self.priority_event_7
        ]


@dataclass
class CodiceLoL3aPartialDensityDataProduct:
    epoch: ndarray
    epoch_delta: ndarray
    h_partial_density: ndarray
    he_partial_density: ndarray
    c4_partial_density: ndarray
    c5_partial_density: ndarray
    c6_partial_density: ndarray
    o5_partial_density: ndarray
    o6_partial_density: ndarray
    o7_partial_density: ndarray
    o8_partial_density: ndarray
    mg_partial_density: ndarray
    si_partial_density: ndarray
    fe_low_partial_density: ndarray
    fe_high_partial_density: ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(H_PARTIAL_DENSITY_VAR_NAME, self.h_partial_density),
            DataProductVariable(HE_PARTIAL_DENSITY_VAR_NAME, self.he_partial_density),
            DataProductVariable(C4_PARTIAL_DENSITY_VAR_NAME, self.c4_partial_density),
            DataProductVariable(C5_PARTIAL_DENSITY_VAR_NAME, self.c5_partial_density),
            DataProductVariable(C6_PARTIAL_DENSITY_VAR_NAME, self.c6_partial_density),
            DataProductVariable(O5_PARTIAL_DENSITY_VAR_NAME, self.o5_partial_density),
            DataProductVariable(O6_PARTIAL_DENSITY_VAR_NAME, self.o6_partial_density),
            DataProductVariable(O7_PARTIAL_DENSITY_VAR_NAME, self.o7_partial_density),
            DataProductVariable(O8_PARTIAL_DENSITY_VAR_NAME, self.o8_partial_density),
            DataProductVariable(MG_PARTIAL_DENSITY_VAR_NAME, self.mg_partial_density),
            DataProductVariable(SI_PARTIAL_DENSITY_VAR_NAME, self.si_partial_density),
            DataProductVariable(FE_LOW_PARTIAL_DENSITY_VAR_NAME, self.fe_low_partial_density),
            DataProductVariable(FE_HIGH_PARTIAL_DENSITY_VAR_NAME, self.fe_high_partial_density),
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
