import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames

from imap_l3_processing.ena_maps.new_map_types import DerivedPointingSet, AbstractPointingSetOperation
from imap_l3_processing.hi.l3.utils import SpinPhase


class MaskVarBySpinPhase(AbstractPointingSetOperation):
    def __init__(self, var_to_mask: str, spin_phase: SpinPhase):
        assert spin_phase in [SpinPhase.RamOnly, SpinPhase.AntiRamOnly]

        self.var_to_mask = var_to_mask
        self.spin_phase = spin_phase

    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        assert CoordNames.AZIMUTH_L1C.value in pointing_set

        pset_azimuths = pointing_set.data[CoordNames.AZIMUTH_L1C.value].values
        ram_mask = np.less(0, pset_azimuths) & np.less(pset_azimuths, 180)
        antiram_mask = ~ram_mask

        phase_to_mask = {SpinPhase.RamOnly: ram_mask, SpinPhase.AntiRamOnly: antiram_mask}
        masked_data = pointing_set.data[self.var_to_mask].values * phase_to_mask[self.spin_phase]
        return DerivedPointingSet(dataset=pointing_set.data.assign(
            {self.var_to_mask: (pointing_set.data[self.var_to_mask].dims, masked_data)}),
            spice_reference_frame=pointing_set.spice_reference_frame)
