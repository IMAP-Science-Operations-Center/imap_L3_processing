import enum

from imap_processing.ena_maps.ena_maps import UltraPointingSet

from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData


class Sensor(enum.Enum):
    Ultra45 = "45"
    Ultra90 = "90"

    @staticmethod
    def get_sensor_angle(sensor_name):
        raise NotImplementedError


class UltraSurvivalProbability(UltraPointingSet):
    def __init__(self, l1c_pset: UltraL1CPSet, l3e_glows: UltraGlowsL3eData):
        return NotImplementedError


class UltraSurvivalProbabilitySkyMap():
    def __init__(self, sp: list[UltraSurvivalProbability]):
        return NotImplementedError
