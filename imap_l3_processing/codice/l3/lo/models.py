from dataclasses import dataclass
from pathlib import Path

from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.models import DataProductVariable

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
    epoch_delta: ndarray
    energy: ndarray
    spin_sector: ndarray
    ssd_id: ndarray
    h_intensities: ndarray
    he_intensities: ndarray
    c4_intensities: ndarray
    c5_intensities: ndarray
    c6_intensities: ndarray
    o5_intensities: ndarray
    o6_intensities: ndarray
    o7_intensities: ndarray
    o8_intensities: ndarray
    mg_intensities: ndarray
    si_intensities: ndarray
    fe_low_intensities: ndarray
    fe_high_intensities: ndarray

    @classmethod
    def read_from_cdf(cls, l2_sectored_intensities_cdf: Path):
        with CDF(str(l2_sectored_intensities_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                epoch_delta=cdf["epoch_delta"][...],
                energy=cdf["energy"][...],
                spin_sector=cdf["spin_sector"][...],
                ssd_id=cdf["ssd_id"][...],
                h_intensities=cdf["h_intensities"][...],
                he_intensities=cdf["he_intensities"][...],
                c4_intensities=cdf["c4_intensities"][...],
                c5_intensities=cdf["c5_intensities"][...],
                c6_intensities=cdf["c6_intensities"][...],
                o5_intensities=cdf["o5_intensities"][...],
                o6_intensities=cdf["o6_intensities"][...],
                o7_intensities=cdf["o7_intensities"][...],
                o8_intensities=cdf["o8_intensities"][...],
                mg_intensities=cdf["mg_intensities"][...],
                si_intensities=cdf["si_intensities"][...],
                fe_low_intensities=cdf["fe_low_intensities"][...],
                fe_high_intensities=cdf["fe_high_intensities"][...],
            )

    def get_species_intensities(self) -> dict:
        return {
            "H+": self.h_intensities,
            "He++": self.he_intensities,
            "C+4": self.c4_intensities,
            "C+5": self.c5_intensities,
            "C+6": self.c6_intensities,
            "O+5": self.o5_intensities,
            "O+6": self.o6_intensities,
            "O+7": self.o7_intensities,
            "O+8": self.o8_intensities,
            "Mg": self.mg_intensities,
            "Si": self.si_intensities,
            "Fe (low Q)": self.fe_low_intensities,
            "Fe (high Q)": self.fe_high_intensities,
        }


@dataclass
class CodiceLoL2DirectEventData:
    epoch: ndarray
    event_num: ndarray
    p0_apdenergy: ndarray
    p0_apdgain: ndarray
    p0_apd_id: ndarray
    p0_dataquality: ndarray
    p0_energystep: ndarray
    p0_multiflag: ndarray
    p0_numevents: ndarray
    p0_phatype: ndarray
    p0_spinangle: ndarray
    p0_tof: ndarray
    p1_apdenergy: ndarray
    p1_apdgain: ndarray
    p1_apd_id: ndarray
    p1_dataquality: ndarray
    p1_energystep: ndarray
    p1_multiflag: ndarray
    p1_numevents: ndarray
    p1_phatype: ndarray
    p1_spinangle: ndarray
    p1_tof: ndarray
    p2_apdenergy: ndarray
    p2_apdgain: ndarray
    p2_apd_id: ndarray
    p2_dataquality: ndarray
    p2_energystep: ndarray
    p2_multiflag: ndarray
    p2_numevents: ndarray
    p2_phatype: ndarray
    p2_spinangle: ndarray
    p2_tof: ndarray
    p3_apdenergy: ndarray
    p3_apdgain: ndarray
    p3_apd_id: ndarray
    p3_dataquality: ndarray
    p3_energystep: ndarray
    p3_multiflag: ndarray
    p3_numevents: ndarray
    p3_phatype: ndarray
    p3_spinangle: ndarray
    p3_tof: ndarray
    p4_apdenergy: ndarray
    p4_apdgain: ndarray
    p4_apd_id: ndarray
    p4_dataquality: ndarray
    p4_energystep: ndarray
    p4_multiflag: ndarray
    p4_numevents: ndarray
    p4_phatype: ndarray
    p4_spinangle: ndarray
    p4_tof: ndarray
    p5_apdenergy: ndarray
    p5_apdgain: ndarray
    p5_apd_id: ndarray
    p5_dataquality: ndarray
    p5_energystep: ndarray
    p5_multiflag: ndarray
    p5_numevents: ndarray
    p5_phatype: ndarray
    p5_spinangle: ndarray
    p5_tof: ndarray
    p6_apdenergy: ndarray
    p6_apdgain: ndarray
    p6_apd_id: ndarray
    p6_dataquality: ndarray
    p6_energystep: ndarray
    p6_multiflag: ndarray
    p6_numevents: ndarray
    p6_phatype: ndarray
    p6_spinangle: ndarray
    p6_tof: ndarray
    p7_apdenergy: ndarray
    p7_apdgain: ndarray
    p7_apd_id: ndarray
    p7_dataquality: ndarray
    p7_energystep: ndarray
    p7_multiflag: ndarray
    p7_numevents: ndarray
    p7_phatype: ndarray
    p7_spinangle: ndarray
    p7_tof: ndarray

    @classmethod
    def read_from_cdf(cls, l2_direct_event_cdf: Path):
        with CDF(str(l2_direct_event_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                event_num=cdf["event_num"][...],
                p0_apdenergy=cdf["P0_APDEnergy"][...],
                p0_apdgain=cdf["P0_APDGain"][...],
                p0_apd_id=cdf["P0_APD_ID"][...],
                p0_dataquality=cdf["P0_DataQuality"][...],
                p0_energystep=cdf["P0_EnergyStep"][...],
                p0_multiflag=cdf["P0_MultiFlag"][...],
                p0_numevents=cdf["P0_NumEvents"][...],
                p0_phatype=cdf["P0_PHAType"][...],
                p0_spinangle=cdf["P0_SpinAngle"][...],
                p0_tof=cdf["P0_TOF"][...],
                p1_apdenergy=cdf["P1_APDEnergy"][...],
                p1_apdgain=cdf["P1_APDGain"][...],
                p1_apd_id=cdf["P1_APD_ID"][...],
                p1_dataquality=cdf["P1_DataQuality"][...],
                p1_energystep=cdf["P1_EnergyStep"][...],
                p1_multiflag=cdf["P1_MultiFlag"][...],
                p1_numevents=cdf["P1_NumEvents"][...],
                p1_phatype=cdf["P1_PHAType"][...],
                p1_spinangle=cdf["P1_SpinAngle"][...],
                p1_tof=cdf["P1_TOF"][...],
                p2_apdenergy=cdf["P2_APDEnergy"][...],
                p2_apdgain=cdf["P2_APDGain"][...],
                p2_apd_id=cdf["P2_APD_ID"][...],
                p2_dataquality=cdf["P2_DataQuality"][...],
                p2_energystep=cdf["P2_EnergyStep"][...],
                p2_multiflag=cdf["P2_MultiFlag"][...],
                p2_numevents=cdf["P2_NumEvents"][...],
                p2_phatype=cdf["P2_PHAType"][...],
                p2_spinangle=cdf["P2_SpinAngle"][...],
                p2_tof=cdf["P2_TOF"][...],
                p3_apdenergy=cdf["P3_APDEnergy"][...],
                p3_apdgain=cdf["P3_APDGain"][...],
                p3_apd_id=cdf["P3_APD_ID"][...],
                p3_dataquality=cdf["P3_DataQuality"][...],
                p3_energystep=cdf["P3_EnergyStep"][...],
                p3_multiflag=cdf["P3_MultiFlag"][...],
                p3_numevents=cdf["P3_NumEvents"][...],
                p3_phatype=cdf["P3_PHAType"][...],
                p3_spinangle=cdf["P3_SpinAngle"][...],
                p3_tof=cdf["P3_TOF"][...],
                p4_apdenergy=cdf["P4_APDEnergy"][...],
                p4_apdgain=cdf["P4_APDGain"][...],
                p4_apd_id=cdf["P4_APD_ID"][...],
                p4_dataquality=cdf["P4_DataQuality"][...],
                p4_energystep=cdf["P4_EnergyStep"][...],
                p4_multiflag=cdf["P4_MultiFlag"][...],
                p4_numevents=cdf["P4_NumEvents"][...],
                p4_phatype=cdf["P4_PHAType"][...],
                p4_spinangle=cdf["P4_SpinAngle"][...],
                p4_tof=cdf["P4_TOF"][...],
                p5_apdenergy=cdf["P5_APDEnergy"][...],
                p5_apdgain=cdf["P5_APDGain"][...],
                p5_apd_id=cdf["P5_APD_ID"][...],
                p5_dataquality=cdf["P5_DataQuality"][...],
                p5_energystep=cdf["P5_EnergyStep"][...],
                p5_multiflag=cdf["P5_MultiFlag"][...],
                p5_numevents=cdf["P5_NumEvents"][...],
                p5_phatype=cdf["P5_PHAType"][...],
                p5_spinangle=cdf["P5_SpinAngle"][...],
                p5_tof=cdf["P5_TOF"][...],
                p6_apdenergy=cdf["P6_APDEnergy"][...],
                p6_apdgain=cdf["P6_APDGain"][...],
                p6_apd_id=cdf["P6_APD_ID"][...],
                p6_dataquality=cdf["P6_DataQuality"][...],
                p6_energystep=cdf["P6_EnergyStep"][...],
                p6_multiflag=cdf["P6_MultiFlag"][...],
                p6_numevents=cdf["P6_NumEvents"][...],
                p6_phatype=cdf["P6_PHAType"][...],
                p6_spinangle=cdf["P6_SpinAngle"][...],
                p6_tof=cdf["P6_TOF"][...],
                p7_apdenergy=cdf["P7_APDEnergy"][...],
                p7_apdgain=cdf["P7_APDGain"][...],
                p7_apd_id=cdf["P7_APD_ID"][...],
                p7_dataquality=cdf["P7_DataQuality"][...],
                p7_energystep=cdf["P7_EnergyStep"][...],
                p7_multiflag=cdf["P7_MultiFlag"][...],
                p7_numevents=cdf["P7_NumEvents"][...],
                p7_phatype=cdf["P7_PHAType"][...],
                p7_spinangle=cdf["P7_SpinAngle"][...],
                p7_tof=cdf["P7_TOF"][...],

            )


@dataclass
class CodiceLoL3aDataProduct:
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
