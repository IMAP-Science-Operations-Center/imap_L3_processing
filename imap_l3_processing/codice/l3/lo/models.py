from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable, read_variable_and_mask_fill_values
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.models import DataProductVariable, DataProduct

CODICE_LO_L2_NUM_PRIORITIES = 7

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
H_PARTIAL_DENSITY_VAR_NAME = "hplus_partial_density"
HE_PARTIAL_DENSITY_VAR_NAME = "heplusplus_partial_density"
C4_PARTIAL_DENSITY_VAR_NAME = "cplus4_partial_density"
C5_PARTIAL_DENSITY_VAR_NAME = "cplus5_partial_density"
C6_PARTIAL_DENSITY_VAR_NAME = "cplus6_partial_density"
O5_PARTIAL_DENSITY_VAR_NAME = "oplus5_partial_density"
O6_PARTIAL_DENSITY_VAR_NAME = "oplus6_partial_density"
O7_PARTIAL_DENSITY_VAR_NAME = "oplus7_partial_density"
O8_PARTIAL_DENSITY_VAR_NAME = "oplus8_partial_density"
NE_PARTIAL_DENSITY_VAR_NAME = "ne_partial_density"
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
    spin_angle: ndarray
    elevation: ndarray
    position: ndarray
    tof: ndarray


@dataclass
class CodiceLoL2DirectEventData:
    epoch: ndarray
    epoch_delta_plus: ndarray
    epoch_delta_minus: ndarray
    priority_events: list[PriorityEvent]

    @classmethod
    def read_from_cdf(cls, l2_direct_event_cdf: Path):
        with CDF(str(l2_direct_event_cdf)) as cdf:
            priority_events = []
            for index in range(CODICE_LO_L2_NUM_PRIORITIES):
                priority_events.append(PriorityEvent(
                    apd_energy=read_numeric_variable(cdf[f"p{index}_apd_energy"]),
                    apd_gain=read_variable_and_mask_fill_values(cdf[f"p{index}_gain"]),
                    apd_id=read_variable_and_mask_fill_values(cdf[f"p{index}_apd_id"]),
                    data_quality=read_variable_and_mask_fill_values(cdf[f"p{index}_data_quality"]),
                    energy_step=read_numeric_variable(cdf[f"p{index}_energy_step"]),
                    multi_flag=read_variable_and_mask_fill_values(cdf[f"p{index}_multi_flag"]),
                    num_events=read_variable_and_mask_fill_values(cdf[f"p{index}_num_events"]),
                    spin_angle=read_numeric_variable(cdf[f"p{index}_spin_sector"]),
                    elevation=read_numeric_variable(cdf[f"p{index}_elevation"]),
                    tof=read_variable_and_mask_fill_values(cdf[f"p{index}_tof"]),
                    position=read_variable_and_mask_fill_values(cdf[f"p{index}_position"]),
                )

                )

            return cls(
                epoch=cdf["epoch"][...],
                epoch_delta_plus=cdf["epoch_delta_plus"][...],
                epoch_delta_minus=cdf["epoch_delta_minus"][...],
                priority_events=priority_events
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
                rgfo_half_spin=read_variable_and_mask_fill_values(cdf["rgfo_half_spin"]),
                nso_half_spin=read_variable_and_mask_fill_values(cdf["nso_half_spin"]),
                sw_bias_gain_mode=read_variable_and_mask_fill_values(cdf["sw_bias_gain_mode"]),
                st_bias_gain_mode=read_variable_and_mask_fill_values(cdf["st_bias_gain_mode"]),
                data_quality=read_variable_and_mask_fill_values(cdf["data_quality"]),
                spin_period=read_numeric_variable(cdf["spin_period"]),
                p0_tcrs=read_variable_and_mask_fill_values(cdf["p0_tcrs"]),
                p1_hplus=read_variable_and_mask_fill_values(cdf["p1_hplus"]),
                p2_heplusplus=read_variable_and_mask_fill_values(cdf["p2_heplusplus"]),
                p3_heavies=read_variable_and_mask_fill_values(cdf["p3_heavies"]),
                p4_dcrs=read_variable_and_mask_fill_values(cdf["p4_dcrs"])
            )


@dataclass
class CodiceLoL1aNSWPriorityRates:
    epoch: np.ndarray
    epoch_delta_plus: np.ndarray
    epoch_delta_minus: np.ndarray
    energy_table: np.ndarray
    acquisition_time_per_step: np.ndarray
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
                epoch=cdf["epoch"][...],
                epoch_delta_plus=cdf["epoch_delta_plus"][...],
                epoch_delta_minus=cdf["epoch_delta_minus"][...],
                energy_table=cdf["energy_table"][...],
                acquisition_time_per_step=cdf["acquisition_time_per_step"][...],
                spin_sector_index=cdf["spin_sector_index"][...],
                rgfo_half_spin=read_variable_and_mask_fill_values(cdf["rgfo_half_spin"]),
                data_quality=read_variable_and_mask_fill_values(cdf["data_quality"]),
                p5_heavies=read_variable_and_mask_fill_values(cdf["p5_heavies"]),
                p6_hplus_heplusplus=read_variable_and_mask_fill_values(cdf["p6_hplus_heplusplus"]),
                nso_half_spin=read_variable_and_mask_fill_values(cdf["nso_half_spin"]),
                sw_bias_gain_mode=read_variable_and_mask_fill_values(cdf["sw_bias_gain_mode"]),
                st_bias_gain_mode=read_variable_and_mask_fill_values(cdf["st_bias_gain_mode"]),
                spin_period=read_numeric_variable(cdf["spin_period"]),
            )


@dataclass
class CodiceLoPartialDensityData:
    epoch: ndarray
    epoch_delta: ndarray
    hplus_partial_density: ndarray
    heplusplus_partial_density: ndarray
    cplus4_partial_density: ndarray
    cplus5_partial_density: ndarray
    cplus6_partial_density: ndarray
    oplus5_partial_density: ndarray
    oplus6_partial_density: ndarray
    oplus7_partial_density: ndarray
    oplus8_partial_density: ndarray
    ne_partial_density: ndarray
    mg_partial_density: ndarray
    si_partial_density: ndarray
    fe_loq_partial_density: ndarray
    fe_hiq_partial_density: ndarray

    @classmethod
    def read_from_cdf(cls, cdf_path: Path | str):
        with CDF(str(cdf_path)) as cdf:
            return cls(
                epoch=cdf[EPOCH_VAR_NAME][...],
                epoch_delta=cdf[EPOCH_DELTA_VAR_NAME][...],
                hplus_partial_density=read_numeric_variable(cdf[H_PARTIAL_DENSITY_VAR_NAME]),
                heplusplus_partial_density=read_numeric_variable(cdf[HE_PARTIAL_DENSITY_VAR_NAME]),
                cplus4_partial_density=read_numeric_variable(cdf[C4_PARTIAL_DENSITY_VAR_NAME]),
                cplus5_partial_density=read_numeric_variable(cdf[C5_PARTIAL_DENSITY_VAR_NAME]),
                cplus6_partial_density=read_numeric_variable(cdf[C6_PARTIAL_DENSITY_VAR_NAME]),
                oplus5_partial_density=read_numeric_variable(cdf[O5_PARTIAL_DENSITY_VAR_NAME]),
                oplus6_partial_density=read_numeric_variable(cdf[O6_PARTIAL_DENSITY_VAR_NAME]),
                oplus7_partial_density=read_numeric_variable(cdf[O7_PARTIAL_DENSITY_VAR_NAME]),
                oplus8_partial_density=read_numeric_variable(cdf[O8_PARTIAL_DENSITY_VAR_NAME]),
                ne_partial_density=read_numeric_variable(cdf[NE_PARTIAL_DENSITY_VAR_NAME]),
                mg_partial_density=read_numeric_variable(cdf[MG_PARTIAL_DENSITY_VAR_NAME]),
                si_partial_density=read_numeric_variable(cdf[SI_PARTIAL_DENSITY_VAR_NAME]),
                fe_loq_partial_density=read_numeric_variable(cdf[FE_LOW_PARTIAL_DENSITY_VAR_NAME]),
                fe_hiq_partial_density=read_numeric_variable(cdf[FE_HIGH_PARTIAL_DENSITY_VAR_NAME]),
            )


@dataclass
class CodiceLoL3aPartialDensityDataProduct(DataProduct):
    data: CodiceLoPartialDensityData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.data.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.data.epoch_delta),
            DataProductVariable(H_PARTIAL_DENSITY_VAR_NAME, self.data.hplus_partial_density),
            DataProductVariable(HE_PARTIAL_DENSITY_VAR_NAME, self.data.heplusplus_partial_density),
            DataProductVariable(C4_PARTIAL_DENSITY_VAR_NAME, self.data.cplus4_partial_density),
            DataProductVariable(C5_PARTIAL_DENSITY_VAR_NAME, self.data.cplus5_partial_density),
            DataProductVariable(C6_PARTIAL_DENSITY_VAR_NAME, self.data.cplus6_partial_density),
            DataProductVariable(O5_PARTIAL_DENSITY_VAR_NAME, self.data.oplus5_partial_density),
            DataProductVariable(O6_PARTIAL_DENSITY_VAR_NAME, self.data.oplus6_partial_density),
            DataProductVariable(O7_PARTIAL_DENSITY_VAR_NAME, self.data.oplus7_partial_density),
            DataProductVariable(O8_PARTIAL_DENSITY_VAR_NAME, self.data.oplus8_partial_density),
            DataProductVariable(NE_PARTIAL_DENSITY_VAR_NAME, self.data.ne_partial_density),
            DataProductVariable(MG_PARTIAL_DENSITY_VAR_NAME, self.data.mg_partial_density),
            DataProductVariable(SI_PARTIAL_DENSITY_VAR_NAME, self.data.si_partial_density),
            DataProductVariable(FE_LOW_PARTIAL_DENSITY_VAR_NAME, self.data.fe_loq_partial_density),
            DataProductVariable(FE_HIGH_PARTIAL_DENSITY_VAR_NAME, self.data.fe_hiq_partial_density)
        ]


EVENT_INDEX_VAR_NAME = "event_index"
SPIN_ANGLE_DELTA_BIN_VAR_NAME = "spin_angle_bin_delta"
SPIN_ANGLE_BIN_VAR_NAME = "spin_angle_bin"
ENERGY_BIN_VAR_NAME = "energy_bin"
ENERGY_BIN_DELTA_PLUS_VAR_NAME = "energy_bin_delta_plus"
ENERGY_BIN_DELTA_MINUS_VAR_NAME = "energy_bin_delta_minus"
PRIORITY_INDEX_VAR_NAME = "priority_index"
NORMALIZATION_VAR_NAME = "normalization"
MASS_PER_CHARGE_VAR_NAME = "mass_per_charge"
MASS_VAR_NAME = "mass"
APD_ENERGY_VAR_NAME = "apd_energy"
ENERGY_STEP_VAR_NAME = "energy_step"
GAIN_VAR_NAME = "gain"
APD_ID_VAR_NAME = "apd_id"
MULTI_FLAG_VAR_NAME = "multi_flag"
NUM_EVENTS_VAR_NAME = "num_events"
DATA_QUALITY_VAR_NAME = "data_quality"
TOF_VAR_NAME = "tof"
SPIN_ANGLE_VAR_NAME = "spin_angle"
ELEVATION_VAR_NAME = "elevation"
POSITION_VAR_NAME = "position"
PRIORITY_INDEX_LABEL_VAR_NAME = "priority_index_label"
EVENT_INDEX_LABEL_VAR_NAME = "event_index_label"
ENERGY_BIN_LABEL_VAR_NAME = "energy_bin_label"
SPIN_ANGLE_BIN_LABEL_VAR_NAME = "spin_angle_bin_label"


@dataclass
class CodiceLoDirectEventData:
    epoch: ndarray
    epoch_delta: ndarray
    normalization: ndarray
    mass_per_charge: np.ndarray
    mass: np.ndarray
    apd_energy: np.ndarray
    energy_step: np.ndarray
    gain: np.ndarray
    apd_id: np.ndarray
    multi_flag: np.ndarray
    num_events: np.ndarray
    data_quality: np.ndarray
    tof: np.ndarray
    spin_angle: np.ndarray
    elevation: np.ndarray
    position: np.ndarray

    @classmethod
    def read_from_cdf(cls, cdf_path: Path | str):
        with CDF(str(cdf_path)) as cdf:
            return cls(epoch=cdf[EPOCH_VAR_NAME][...],
                       epoch_delta=cdf[EPOCH_DELTA_VAR_NAME][...],
                       normalization=read_numeric_variable(cdf[NORMALIZATION_VAR_NAME]),
                       mass_per_charge=read_numeric_variable(cdf[MASS_PER_CHARGE_VAR_NAME]),
                       mass=read_numeric_variable(cdf[MASS_VAR_NAME]),
                       apd_energy=read_numeric_variable(cdf[APD_ENERGY_VAR_NAME]),
                       energy_step=read_numeric_variable(cdf[ENERGY_STEP_VAR_NAME]),
                       gain=read_variable_and_mask_fill_values(cdf[GAIN_VAR_NAME]),
                       apd_id=read_variable_and_mask_fill_values(cdf[APD_ID_VAR_NAME]),
                       multi_flag=read_variable_and_mask_fill_values(cdf[MULTI_FLAG_VAR_NAME]),
                       num_events=read_variable_and_mask_fill_values(cdf[NUM_EVENTS_VAR_NAME]),
                       data_quality=read_variable_and_mask_fill_values(cdf[DATA_QUALITY_VAR_NAME]),
                       tof=read_variable_and_mask_fill_values(cdf[TOF_VAR_NAME]),
                       spin_angle=read_numeric_variable(cdf[SPIN_ANGLE_VAR_NAME]),
                       elevation=read_numeric_variable(cdf[ELEVATION_VAR_NAME]),
                       position=read_variable_and_mask_fill_values(cdf[POSITION_VAR_NAME]), )


@dataclass
class CodiceLoL3aDirectEventDataProduct(CodiceLoDirectEventData, DataProduct):
    energy_bin: np.ndarray
    energy_bin_delta_plus: np.ndarray
    energy_bin_delta_minus: np.ndarray
    spin_angle_bin_delta: np.ndarray
    spin_angle_bin: np.ndarray
    priority_index: np.ndarray = field(init=False)
    event_index: np.ndarray = field(init=False)
    priority_index_label: np.ndarray = field(init=False)
    event_index_label: np.ndarray = field(init=False)
    energy_bin_label: np.ndarray = field(init=False)
    spin_angle_bin_label: np.ndarray = field(init=False)

    def __post_init__(self):
        self.priority_index = np.arange(CODICE_LO_L2_NUM_PRIORITIES)
        self.event_index = np.arange(self.mass_per_charge.shape[-1])
        self.priority_index_label = self.priority_index.astype(str)
        self.event_index_label = self.event_index.astype(str)
        self.energy_bin_label = self.energy_bin.astype(str)
        self.spin_angle_bin_label = self.spin_angle_bin.astype(str)

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),

            DataProductVariable(NORMALIZATION_VAR_NAME, self.normalization),

            DataProductVariable(ENERGY_BIN_VAR_NAME, self.energy_bin),
            DataProductVariable(ENERGY_BIN_DELTA_MINUS_VAR_NAME, self.energy_bin_delta_minus),
            DataProductVariable(ENERGY_BIN_DELTA_PLUS_VAR_NAME, self.energy_bin_delta_plus),
            DataProductVariable(ENERGY_BIN_LABEL_VAR_NAME, self.energy_bin_label),

            DataProductVariable(SPIN_ANGLE_BIN_VAR_NAME, self.spin_angle_bin),
            DataProductVariable(SPIN_ANGLE_DELTA_BIN_VAR_NAME, self.spin_angle_bin_delta),
            DataProductVariable(SPIN_ANGLE_BIN_LABEL_VAR_NAME, self.spin_angle_bin_label),

            DataProductVariable(MASS_PER_CHARGE_VAR_NAME, self.mass_per_charge),
            DataProductVariable(APD_ENERGY_VAR_NAME, self.apd_energy),
            DataProductVariable(MASS_VAR_NAME, self.mass),
            DataProductVariable(GAIN_VAR_NAME, self.gain),
            DataProductVariable(APD_ID_VAR_NAME, self.apd_id),
            DataProductVariable(MULTI_FLAG_VAR_NAME, self.multi_flag),
            DataProductVariable(NUM_EVENTS_VAR_NAME, self.num_events),
            DataProductVariable(DATA_QUALITY_VAR_NAME, self.data_quality),
            DataProductVariable(TOF_VAR_NAME, self.tof),
            DataProductVariable(ELEVATION_VAR_NAME, self.elevation),
            DataProductVariable(POSITION_VAR_NAME, self.position),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(ENERGY_STEP_VAR_NAME, self.energy_step),

            DataProductVariable(PRIORITY_INDEX_VAR_NAME, self.priority_index),
            DataProductVariable(PRIORITY_INDEX_LABEL_VAR_NAME, self.priority_index_label),
            DataProductVariable(EVENT_INDEX_VAR_NAME, self.event_index),
            DataProductVariable(EVENT_INDEX_LABEL_VAR_NAME, self.event_index_label),
        ]


C_TO_O_RATIO_VAR_NAME = "c_to_o_ratio"
MG_TO_O_RATIO_VAR_NAME = "mg_to_o_ratio"
FE_TO_O_RATIO_VAR_NAME = "fe_to_o_ratio"
C6_TO_C5_RATIO_VAR_NAME = "c6_to_c5_ratio"
C6_TO_C4_RATIO_VAR_NAME = "c6_to_c4_ratio"
O7_TO_O6_RATIO_VAR_NAME = "o7_to_o6_ratio"
FELO_TO_FEHI_RATIO_VAR_NAME = "felo_to_fehi_ratio"


@dataclass
class CodiceLoL3aRatiosDataProduct(DataProduct):
    epoch: ndarray
    epoch_delta: ndarray
    c_to_o_ratio: ndarray
    mg_to_o_ratio: ndarray
    fe_to_o_ratio: ndarray
    c6_to_c5_ratio: ndarray
    c6_to_c4_ratio: ndarray
    o7_to_o6_ratio: ndarray
    felo_to_fehi_ratio: ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(C_TO_O_RATIO_VAR_NAME, self.c_to_o_ratio),
            DataProductVariable(MG_TO_O_RATIO_VAR_NAME, self.mg_to_o_ratio),
            DataProductVariable(FE_TO_O_RATIO_VAR_NAME, self.fe_to_o_ratio),
            DataProductVariable(C6_TO_C5_RATIO_VAR_NAME, self.c6_to_c5_ratio),
            DataProductVariable(C6_TO_C4_RATIO_VAR_NAME, self.c6_to_c4_ratio),
            DataProductVariable(O7_TO_O6_RATIO_VAR_NAME, self.o7_to_o6_ratio),
            DataProductVariable(FELO_TO_FEHI_RATIO_VAR_NAME, self.felo_to_fehi_ratio)
        ]


@dataclass
class CodiceLoL3ChargeStateDistributionsDataProduct(DataProduct):
    epoch: np.ndarray
    epoch_delta: np.ndarray
    oxygen_charge_state_distribution: np.ndarray
    carbon_charge_state_distribution: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable("epoch", self.epoch),
            DataProductVariable("epoch_delta", self.epoch_delta),
            DataProductVariable("oxygen_charge_state_distribution", self.oxygen_charge_state_distribution),
            DataProductVariable("carbon_charge_state_distribution", self.carbon_charge_state_distribution),
            DataProductVariable("oxygen_charge_state", np.array([5, 6, 7, 8])),
            DataProductVariable("carbon_charge_state", np.array([4, 5, 6])),
        ]


EPOCH = TypeVar("EPOCH")
PRIORITY = TypeVar("PRIORITY")
SPECIES = TypeVar("SPECIES")
AZIMUTH = TypeVar("AZIMUTH")
SPIN_ANGLE = TypeVar("SPIN_ANGLE")
ENERGY = TypeVar("ENERGY")

AZIMUTH_OR_ELEVATION = TypeVar("AZIMUTH_OR_ELEVATION")


@dataclass
class CodiceLo3dData:
    data_in_3d_bins: np.ndarray[(SPECIES, EPOCH, PRIORITY, AZIMUTH_OR_ELEVATION, SPIN_ANGLE, ENERGY)] | \
                     np.ndarray[(SPECIES, EPOCH, AZIMUTH_OR_ELEVATION, SPIN_ANGLE, ENERGY)]
    mass_bin_lookup: MassSpeciesBinLookup
    energy_per_charge: np.ndarray[(ENERGY,)]
    spin_angle: np.ndarray[(SPIN_ANGLE,)]
    azimuth_or_elevation: np.ndarray[(AZIMUTH_OR_ELEVATION,)]

    def get_3d_distribution(self, species: str) -> np.ndarray:
        species_index = self.mass_bin_lookup.get_species_index(species)
        return self.data_in_3d_bins[species_index, ...]


ENERGY_VAR_NAME = "energy"
ENERGY_DELTA_PLUS_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_VAR_NAME = "energy_delta_minus"
SPIN_ANGLE_DELTA_VAR_NAME = "spin_angle_delta"
ELEVATION_DELTA_VAR_NAME = "elevation_delta"

ENERGY_LABEL_VAR_NAME = "energy_label"
SPIN_ANGLE_LABEL_VAR_NAME = "spin_angle_label"
ELEVATION_ANGLE_LABEL_VAR_NAME = "elevation_label"


@dataclass
class CodiceLoL3a3dDistributionDataProduct(DataProduct):
    epoch: np.ndarray
    epoch_delta: np.ndarray
    elevation: np.ndarray
    elevation_delta: np.ndarray
    spin_angle: np.ndarray
    spin_angle_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    species: str
    species_data: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(ELEVATION_VAR_NAME, self.elevation),
            DataProductVariable(ELEVATION_DELTA_VAR_NAME, self.elevation_delta),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(SPIN_ANGLE_DELTA_VAR_NAME, self.spin_angle_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.energy_delta_minus),
            DataProductVariable(self.species, self.species_data),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, self.energy.astype(str)),
            DataProductVariable(SPIN_ANGLE_LABEL_VAR_NAME, self.spin_angle.astype(str)),
            DataProductVariable(ELEVATION_ANGLE_LABEL_VAR_NAME, self.elevation.astype(str)),
        ]
