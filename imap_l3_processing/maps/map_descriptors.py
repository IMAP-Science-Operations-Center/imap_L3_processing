import enum
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional


class Sensor(enum.Enum):
    Hi45 = "Hi45"
    Hi90 = "Hi90"
    Lo90 = "Lo90"
    Lo = "Lo"
    HiCombined = "HiCombined"
    Ultra45 = "Ultra45"
    Ultra90 = "Ultra90"
    UltraCombined = "UltraCombined"

    @staticmethod
    def get_sensor_angle(sensor_name):
        sensor_angles = {Sensor.Hi45: -45,
                         Sensor.Hi90: 0,
                         Sensor.Lo90: 0,
                         Sensor.Lo: 0,
                         Sensor.Ultra45: -45,
                         Sensor.Ultra90: 0}
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


class PixelSize(enum.IntEnum):
    TwoDegrees = 2
    FourDegrees = 4
    SixDegrees = 6
    Nside8 = 8
    Nside16 = 16


class MapQuantity(enum.Enum):
    Intensity = "Intensity"
    SpectralIndex = "SpectralIndex"
    ISNBackgroundSubtracted = "ISNBackgroundSubtracted"


@dataclass
class MapDescriptorParts:
    sensor: Sensor
    reference_frame: ReferenceFrame
    survival_correction: SurvivalCorrection
    spin_phase: SpinPhase
    coord: str
    grid: PixelSize
    duration: str
    quantity: MapQuantity
    quantity_suffix: str
    spectral_index_energy_range: Optional[tuple[int,int]] = None


sensor_mapping = [
    ("hic", Sensor.HiCombined),
    ("h45", Sensor.Hi45),
    ("h90", Sensor.Hi90),
    ("l090", Sensor.Lo90),
    ("ilo", Sensor.Lo),
    ("ulc", Sensor.UltraCombined),
    ("u45", Sensor.Ultra45),
    ("u90", Sensor.Ultra90)
]

quantity_mapping = [
    ("spx", MapQuantity.SpectralIndex),
    ("ena", MapQuantity.Intensity),
    ("isn", MapQuantity.ISNBackgroundSubtracted)
]

cg_correction_mapping = [
    ("sf", ReferenceFrame.Spacecraft),
    ("hf", ReferenceFrame.Heliospheric),
    ("hk", ReferenceFrame.HeliosphericKinematic)
]

sp_correction_mapping = [
    ("sp", SurvivalCorrection.SurvivalCorrected),
    ("nsp", SurvivalCorrection.NotSurvivalCorrected)
]

spin_phase_mapping = [
    ("ram", SpinPhase.RamOnly),
    ("anti", SpinPhase.AntiRamOnly),
    ("full", SpinPhase.FullSpin)
]

grid_size_mapping = [
    ("2deg", PixelSize.TwoDegrees),
    ("4deg", PixelSize.FourDegrees),
    ("6deg", PixelSize.SixDegrees),
    ("nside8", PixelSize.Nside8),
    ("nside16", PixelSize.Nside16)
]


def parse_map_descriptor(descriptor: str) -> Optional[MapDescriptorParts]:
    descriptor_regex = """
        (?P<sensor>hic|h45|h90|l090|ulc|u45|u90|ilo)-
        (?P<quantity>ena|spx|isn)(?P<spectral_index_range>[0-9]{4})?(?P<quantity_suffix>[a-zA-Z]*)-
        (?P<species>h|o)-
        (?P<frame>sf|hf|hk)-
        (?P<survival_corrected>sp|nsp)-
        (?P<spin_phase>ram|anti|full)-
        (?P<coord>[a-zA-Z0-9]*)-
        (?P<grid>2deg|4deg|6deg|nside8|nside16)-
        (?P<duration>[0-9]+(?:mo|yr))
        """

    descriptor_part_match = re.fullmatch(descriptor_regex, descriptor, flags=re.VERBOSE)
    if descriptor_part_match is None:
        return None

    sensor_part_from_str = {desc: sensor_part for desc, sensor_part in sensor_mapping}
    quantity_part_from_str = {desc: quantity_part for desc, quantity_part in quantity_mapping}
    cg_correction_part_from_str = {desc: cg_correction_part for desc, cg_correction_part in cg_correction_mapping}
    sp_correction_part_from_str = {desc: sp_correction_part for desc, sp_correction_part in sp_correction_mapping}
    spin_phase_part_from_str = {desc: spin_phase_part for desc, spin_phase_part in spin_phase_mapping}
    grid_size_part_from_str = {desc: grid_size_part for desc, grid_size_part in grid_size_mapping}
    spectral_index_energy_range = None
    if descriptor_part_match["spectral_index_range"] is not None:
        spectral_index_energy_range = (int(descriptor_part_match["spectral_index_range"][:2]), int(descriptor_part_match["spectral_index_range"][2:]))

    return MapDescriptorParts(
        sensor=sensor_part_from_str[descriptor_part_match["sensor"]],
        quantity=quantity_part_from_str[descriptor_part_match["quantity"]],
        quantity_suffix=descriptor_part_match["quantity_suffix"],
        reference_frame=cg_correction_part_from_str[descriptor_part_match["frame"]],
        survival_correction=sp_correction_part_from_str[descriptor_part_match["survival_corrected"]],
        spin_phase=spin_phase_part_from_str[descriptor_part_match["spin_phase"]],
        coord=descriptor_part_match["coord"],
        grid=grid_size_part_from_str[descriptor_part_match["grid"]],
        duration=descriptor_part_match["duration"],
        spectral_index_energy_range=spectral_index_energy_range
    )


def map_descriptor_parts_to_string(descriptor_parts: MapDescriptorParts) -> str:
    sensor_part_to_str = {sensor_part: desc for desc, sensor_part in sensor_mapping}
    quantity_part_to_str = {quantity_part: desc for desc, quantity_part in quantity_mapping}
    cg_correction_part_to_str = {cg_correction_part: desc for desc, cg_correction_part in cg_correction_mapping}
    sp_correction_part_to_str = {sp_correction_part: desc for desc, sp_correction_part in sp_correction_mapping}
    spin_phase_part_to_str = {spin_phase_part: desc for desc, spin_phase_part in spin_phase_mapping}
    grid_size_part_to_str = {grid_size_part: desc for desc, grid_size_part in grid_size_mapping}
    spectral_index_energy_range = ""
    if descriptor_parts.spectral_index_energy_range is not None:
        spectral_index_energy_range = f"{descriptor_parts.spectral_index_energy_range[0]:02}{descriptor_parts.spectral_index_energy_range[1]:02}"

    return "-".join([
        sensor_part_to_str[descriptor_parts.sensor],
        quantity_part_to_str[descriptor_parts.quantity] + spectral_index_energy_range + descriptor_parts.quantity_suffix,
        "h",
        cg_correction_part_to_str[descriptor_parts.reference_frame],
        sp_correction_part_to_str[descriptor_parts.survival_correction],
        spin_phase_part_to_str[descriptor_parts.spin_phase],
        descriptor_parts.coord,
        grid_size_part_to_str[descriptor_parts.grid],
        descriptor_parts.duration,
    ])


def get_duration_from_map_descriptor(descriptor: MapDescriptorParts) -> timedelta:
    match descriptor:
        case MapDescriptorParts(duration="1mo"):
            return timedelta(days=365.25) / 12
        case MapDescriptorParts(duration="3mo"):
            return timedelta(days=365.25) / 4
        case MapDescriptorParts(duration="6mo"):
            return timedelta(days=365.25) / 2
        case MapDescriptorParts(duration="1yr" | "12mo"):
            return timedelta(days=365.25)
        case _:
            raise ValueError(f"Expected a duration in the map descriptor, got: {descriptor} (e.g., '1mo', '3mo')")
