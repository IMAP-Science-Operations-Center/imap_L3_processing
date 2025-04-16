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
