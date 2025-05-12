import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.hi.l3.models import HiL1cData, HiGlowsL3eData


def read_hi_l1c_data(path: Union[Path, str]) -> HiL1cData:
    with CDF(str(path)) as cdf:
        return HiL1cData(epoch=cdf["epoch"][0], epoch_j2000=cdf.raw_var("epoch")[...],
                         exposure_times=read_numeric_variable(cdf["exposure_times"]),
                         esa_energy_step=cdf["esa_energy_step"][...])


def read_glows_l3e_data(cdf_path: Union[Path, str]) -> HiGlowsL3eData:
    with CDF(str(cdf_path)) as cdf:
        return HiGlowsL3eData(epoch=cdf["epoch"][0],
                              energy=read_numeric_variable(cdf["energy"]),
                              spin_angle=read_numeric_variable(cdf["spin_angle"]),
                              probability_of_survival=read_numeric_variable(cdf["probability_of_survival"]))


class Sensor(enum.Enum):
    Hi45 = "45"
    Hi90 = "90"
    Combined = "Combined"

    @staticmethod
    def get_sensor_angle(sensor_name):
        sensor_angles = {Sensor.Hi45.value: -45, Sensor.Hi90.value: 0}
        return sensor_angles[sensor_name]


class ReferenceFrame(enum.Enum):
    Heliospheric = "Heliospheric"
    Spacecraft = "Spacecraft"
    HeliosphericKinematic = "HeliosphericKinematic"


class SurvivalCorrection(enum.Enum):
    SurvivalCorrected = "SurvivalCorrected"
    NotSurvivalCorrected = "NotSurvivalCorrected"


class SpinPhase(enum.Enum):
    RamOnly = "RamOnly"
    AntiRamOnly = "AntiRamOnly"
    FullSpin = "FullSpin"


class Duration(enum.Enum):
    ThreeMonths = "ThreeMonths"
    SixMonths = "SixMonths"
    OneYear = "OneYear"


class PixelSize(enum.IntEnum):
    FourDegrees = 4
    SixDegrees = 6


class MapQuantity(enum.Enum):
    Intensity = "Intensity"
    SpectralIndex = "SpectralIndex"


@dataclass
class MapDescriptorParts:
    sensor: Sensor
    reference_frame: ReferenceFrame
    survival_correction: SurvivalCorrection
    spin_phase: SpinPhase
    grid: PixelSize
    duration: Duration
    quantity: MapQuantity


def parse_map_descriptor(descriptor: str) -> Optional[MapDescriptorParts]:
    descriptor_regex = """
        (?P<sensor>hic|h45|h90)-
        (?P<quantity>ena|spx)-
        (?P<species>h)-
        (?P<frame>sf|hf)-
        (?P<survival_corrected>sp|nsp)-
        (?P<spin_phase>ram|anti|full)-
        (?P<coord>hae)-
        (?P<grid>4deg|6deg)-
        (?P<duration>3mo|6mo|1yr)
        """

    descriptor_part_match = re.fullmatch(descriptor_regex, descriptor, flags=re.VERBOSE)
    if descriptor_part_match is None:
        return None

    sensors = {"hic": Sensor.Combined, "h45": Sensor.Hi45, "h90": Sensor.Hi90}
    cg_corrections = {"sf": ReferenceFrame.Spacecraft, "hf": ReferenceFrame.Heliospheric}
    survival_corrections = {"sp": SurvivalCorrection.SurvivalCorrected, "nsp": SurvivalCorrection.NotSurvivalCorrected}
    spin_phases = {"ram": SpinPhase.RamOnly, "anti": SpinPhase.AntiRamOnly, "full": SpinPhase.FullSpin}
    durations = {"3mo": Duration.ThreeMonths, "6mo": Duration.SixMonths, "1yr": Duration.OneYear}
    grid_sizes = {"4deg": PixelSize.FourDegrees, "6deg": PixelSize.SixDegrees}
    quantities = {"spx": MapQuantity.SpectralIndex, "ena": MapQuantity.Intensity}

    return MapDescriptorParts(
        sensor=sensors[descriptor_part_match["sensor"]],
        quantity=quantities[descriptor_part_match["quantity"]],
        reference_frame=cg_corrections[descriptor_part_match["frame"]],
        survival_correction=survival_corrections[descriptor_part_match["survival_corrected"]],
        spin_phase=spin_phases[descriptor_part_match["spin_phase"]],
        grid=grid_sizes[descriptor_part_match["grid"]],
        duration=durations[descriptor_part_match["duration"]],
    )
